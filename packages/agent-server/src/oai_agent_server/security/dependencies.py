from typing import Optional
import os

from fastapi import Header, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from oai_agent_server.exceptions import AuthenticationException
from oai_agent_server.middleware.request_context import get_original_environ


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
    if 'localhost' in str(request.url) and original_environ.get('FORCE_AUTH', 'false').lower() == 'false':
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
    try:
        from oai_agent_server.utils.token_manager import TokenManager
        token_manager = TokenManager()

        # We need the agent name. In a dependency, we can try to get it from the app state 
        # or assume the token manager handles validation generically.
        # Based on previous code, it needed agent_name. 
        # We can access the agent_name from the request.app if stored there, 
        # or we can pass it if we use a class-based dependency.

        # For now, let's try to get agent_name from the request state or app title parsing
        # A more robust way is to store agent_name in app.state
        agent_name = getattr(request.app.state, "agent_name", None)

        # Fallback: If agent_name isn't in state, we might need to rethink how we pass it.
        # However, looking at main.py, we can store it in app.state.

        if not agent_name:
            # Fallback for now, though this should be set in main.py
            agent_name = "unknown"

        is_valid = token_manager.validate_token(agent_name, token)

        if not is_valid:
            raise AuthenticationException(reason="Invalid or expired API token")

        return True

    except ImportError:
        # If TokenManager is missing, and auth is enabled, we should probably fail safe
        # or log an error. The previous middleware returned 500.
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

    if 'localhost' in str(request.url) and original_environ.get('FORCE_AUTH', 'false').lower() == 'false':
        return {}

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
