import ipaddress
import logging
import os
from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from oai_agent_server.exceptions import AuthenticationException
from oai_agent_server.middleware.request_context import get_original_environ
from oai_platform_core.security import is_saml_token
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError

logger = logging.getLogger(__name__)


# Default to loopback only. Override via env var TRUSTED_CIDRS=10.0.0.0/8,127.0.0.1/32
DEFAULT_TRUSTED_CIDRS = "127.0.0.0/8,::1/128"


@lru_cache(maxsize=1)
def _trusted_networks():
    raw = os.environ.get("TRUSTED_CIDRS", DEFAULT_TRUSTED_CIDRS)
    nets = []
    for cidr in raw.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            logger.warning("Invalid CIDR in TRUSTED_CIDRS: %r", cidr)
    return nets


def _client_in_trusted_network(request: Request) -> bool:
    """Check whether the actual TCP peer of *request* is in a trusted CIDR.

    Uses request.client.host (the peer IP) — never the URL string, which
    can be spoofed via Host headers or shared with non-trusted pod IPs.
    """
    if not request.client:
        return False
    try:
        ip = ipaddress.ip_address(request.client.host)
    except (ValueError, AttributeError):
        return False
    return any(ip in net for net in _trusted_networks())


def _is_localhost_request(request: Request) -> bool:
    """Return True when request peer is a loopback address or localhost."""
    if not request.client:
        return False

    host = getattr(request.client, "host", None)
    if not isinstance(host, str) or not host:
        return False

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


@lru_cache(maxsize=1)
def _get_token_manager():
    """Lazy module-level singleton for TokenManager (re-uses Redis connection)."""
    from oai_platform_core.security import TokenManager
    return TokenManager(db_path_name="agent_server_tokens.db")

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def verify_api_key(
        request: Request,
        api_token: Optional[str] = Header(None, alias="api-token"),
        api_token_underscore: Optional[str] = Header(None, alias="api_token"),
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
        authorization: Optional[str] = Header(None, alias="authorization")
):
    """
    Dependency to validate API tokens.

    Checks for tokens in various headers (api-token, api_token, x-api-key, Authorization).
    Validates the token using TokenManager if authentication is enabled.

    Args:
        request: The FastAPI request object.
        api_token: Token from 'api-token' header.
        api_token_underscore: Token from 'api_token' header.
        x_api_key: Token from 'x-api-key' header.
        authorization: Token from 'Authorization' header.

    Returns:
        True if authentication is successful or disabled.

    Raises:
        AuthenticationException: If token is missing or invalid.
        HTTPException: If authentication service is unavailable.
    """
    # Bypass authentication for /health, /ready, /status and /metrics endpoints
    # /metrics must be unauthenticated so Prometheus can scrape it without credentials
    if request.url.path in ["/health", "/ready", "/status", "/metrics"]:
        return True

    # Always allow local loopback calls for local tooling and health checks.
    if _is_localhost_request(request):
        return True

    # 1. Check if Auth is globally enabled
    original_environ = get_original_environ()
    auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'

    if not auth_enabled:
        return True

    # 2. Optional dev-mode trusted-network bypass.
    # Default FORCE_AUTH=true → auth is required unless operators explicitly opt
    # out AND the connection's actual peer IP is in TRUSTED_CIDRS.
    force_auth = original_environ.get('FORCE_AUTH', 'true').lower() != 'false'
    if not force_auth and _client_in_trusted_network(request):
        return True

    # 3. Extract Token (Support api-token, api_token, x-api-key header or Authorization: Bearer)
    token = api_token or api_token_underscore or x_api_key

    # If not found in specific headers, check Authorization header
    if not token and authorization:
        if authorization.lower().startswith('bearer '):
            token = authorization[7:]
        else:
            token = authorization

    # Fallback: Check headers directly from request object (case-insensitive)
    if not token:
        token = (
                request.headers.get('api-token') or
                request.headers.get('api_token') or
                request.headers.get('x-api-key')
        )

    if not token:
        raise AuthenticationException(reason="API token required")

    # 4. Validate Token
    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH", None))
            validation_result = validator.validate_token_and_get_role(token)
            if validation_result.is_valid:
                # Store user info in request state if needed
                request.state.user_role = validation_result.role
                request.state.user_email = validation_result.email
                return True
            else:
                raise AuthenticationException(reason=validation_result.error_message or "Invalid SAML token")
        except TokenValidationError as e:
            raise AuthenticationException(reason=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="SAML token validation service unavailable")
    else:
        try:
            token_manager = _get_token_manager()

            agent_name = getattr(request.app.state, "agent_name", "unknown")

            user_info = token_manager.validate_token(agent_name, token)

            if user_info:
                request.state.user_id = user_info.get("user_id")
                request.state.user_role = user_info.get("role_id")
                return True
            else:
                raise AuthenticationException(reason="Invalid or expired API token")

        except ImportError:
            raise HTTPException(status_code=500, detail="Authentication service unavailable")


# Define security scheme for Swagger UI
security = HTTPBearer(auto_error=False)


async def verify_jwt_token(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Dependency to validate JWT tokens.

    Args:
        request: The FastAPI request object.
        credentials: The HTTP Bearer credentials.

    Returns:
        Decoded token payload if valid.

    Raises:
        AuthenticationException: If token is missing or invalid.
    """
    original_environ = get_original_environ()
    auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'

    if not auth_enabled:
        return {}

    if _is_localhost_request(request):
        return True

    force_auth = original_environ.get('FORCE_AUTH', 'true').lower() != 'false'
    if not force_auth and _client_in_trusted_network(request):
        return True

    if not credentials:
        raise AuthenticationException(reason="Authorization header required")

    token = credentials.credentials
    validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH", None))

    if is_saml_token(str(token)):
        try:
            validation_result = validator.validate_token_and_get_role(str(token))
            if validation_result.is_valid:
                request.state.user_role = validation_result.role
                request.state.user_email = validation_result.email
                return True
            raise AuthenticationException(reason=validation_result.error_message or "Invalid SAML token")
        except TokenValidationError as e:
            raise AuthenticationException(reason=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="SAML token validation service unavailable")
    else:
        try:
            import jwt

            default_path = validator._get_default_key_path()
            public_key_path = os.environ.get("JWT_PUBLIC_KEY_PATH", default_path)
            public_key_content = os.environ.get("JWT_PUBLIC_KEY")

            key = None
            algorithms = []

            if public_key_path and os.path.exists(public_key_path):
                with open(public_key_path, "r") as f:
                    key = f.read()
                algorithms = ["RS256"]
            elif public_key_content:
                key = public_key_content
                algorithms = ["RS256"]
            else:
                # Fallback to Secret Key (HMAC)
                key = os.environ.get("JWT_SECRET_KEY")
                algorithms = ["HS256"]

            if not key:
                raise AuthenticationException(reason="JWT configuration missing (Public Key or Secret Key required)")

            payload = jwt.decode(token, key, algorithms=algorithms)

            if payload:
                role_id = payload.get('role', 'unknown')
                if role_id == 'admin':
                    return payload
            raise AuthenticationException(reason="Invalid role")
        except ImportError:
            raise HTTPException(status_code=500, detail="JWT library not installed")
        except jwt.ExpiredSignatureError:
            raise AuthenticationException(reason="Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationException(reason="Invalid token")


async def verify_api_key_strict(
        request: Request,
        api_token: Optional[str] = Header(None, alias="api-token"),
        api_token_underscore: Optional[str] = Header(None, alias="api_token"),
        x_api_key: Optional[str] = Header(None, alias="x-api-key"),
        authorization: Optional[str] = Header(None, alias="authorization")
):
    """Strict API-key validator for destructive endpoints (e.g. /restart, /kill).

    Always validates the token for non-local peers: it does NOT honor the
    FORCE_AUTH / TRUSTED_CIDRS bypass. Loopback callers (localhost,
    127.0.0.1, ::1) are allowed for local-only operations.
    AGENT_AUTH_ENABLED=false still disables auth globally (intentional,
    matches the rest of the system).
    """
    original_environ = get_original_environ()
    auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'

    if not auth_enabled:
        return True

    if _is_localhost_request(request):
        return True

    token = api_token or api_token_underscore or x_api_key

    if not token and authorization:
        if authorization.lower().startswith('bearer '):
            token = authorization[7:]
        else:
            token = authorization

    if not token:
        token = (
            request.headers.get('api-token') or
            request.headers.get('api_token') or
            request.headers.get('x-api-key')
        )

    if not token:
        raise AuthenticationException(reason="API token required")

    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH", None))
            validation_result = validator.validate_token_and_get_role(token)
            if validation_result.is_valid:
                request.state.user_role = validation_result.role
                request.state.user_email = validation_result.email
                return True
            raise AuthenticationException(reason=validation_result.error_message or "Invalid SAML token")
        except TokenValidationError as e:
            raise AuthenticationException(reason=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="SAML token validation service unavailable")
    else:
        try:
            token_manager = _get_token_manager()
            agent_name = getattr(request.app.state, "agent_name", "unknown")
            user_info = token_manager.validate_token(agent_name, token)
            if user_info:
                request.state.user_id = user_info.get("user_id")
                request.state.user_role = user_info.get("role_id")
                return True
            raise AuthenticationException(reason="Invalid or expired API token")
        except ImportError:
            raise HTTPException(status_code=500, detail="Authentication service unavailable")