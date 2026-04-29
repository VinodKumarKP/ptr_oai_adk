import base64
import os
from typing import Optional

from fastapi import Header, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from oai_agent_server.exceptions import AuthenticationException
from oai_agent_server.middleware.request_context import get_original_environ
from oai_agent_server.utils.saml_token_validation import TokenValidator, TokenValidationError

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def is_saml_token(token: str) -> bool:
    """
    Checks if a token is likely a SAML token by checking if it's base64 encoded XML.
    """
    if not token or not isinstance(token, str) or len(token) % 4 != 0:
        return False
    try:
        decoded_token = base64.b64decode(token, validate=True)
        # Check for SAML or SAMLP tags, without requiring the XML declaration
        return b'<saml:' in decoded_token or b'<samlp:' in decoded_token
    except (ValueError, TypeError):
        return False


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
    # 1. Check if Auth is globally enabled
    original_environ = get_original_environ()
    auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'

    if not auth_enabled:
        return True

    # 2. Check for Localhost exception
    if ('localhost' in str(request.url) or
            '0.0.0.0' in str(request.url) or
            '127.0.0.1' in str(request.url) or
            '::1' in str(request.url) or
            'host.docker.internal' in str(request.url)) and original_environ.get('FORCE_AUTH',
                                                                                    'false').lower() == 'false':
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
            from oai_agent_server.utils.token_manager import TokenManager
            token_manager = TokenManager()

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

    if ('localhost' in str(request.url) or
        '0.0.0.0' in str(request.url) or
        '127.0.0.1' in str(request.url) or
        '::1' in str(request.url) or
        'host.docker.internal' in str(request.url)) and original_environ.get('FORCE_AUTH',
                                                                             'false').lower() == 'false':
        return True

    if not credentials:
        raise AuthenticationException(reason="Authorization header required")

    token = credentials.credentials

    try:
        import jwt

        # Check for Public Key (RSA)
        public_key_path = os.environ.get("JWT_PUBLIC_KEY_PATH")
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