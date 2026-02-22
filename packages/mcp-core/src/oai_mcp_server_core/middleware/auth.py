import os
from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from oai_mcp_server_core.utils.logger_utils import get_logger
from oai_mcp_server_core.core.context import RequestAwareEnviron
from oai_mcp_server_core.core.exceptions import AuthenticationError, DependencyError


class AuthenticationMiddleware(Middleware):
    """Middleware to validate API tokens when AUTH_ENABLED is true."""

    def __init__(self, server_name: str):
        """
        Initialize the authentication middleware.
        
        Args:
            server_name: Name of the server
        """
        self.server_name = server_name
        self.logger = get_logger()

        # Get auth_enabled from original environ (not request-scoped)
        original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
        self.auth_enabled = original_environ.get('AUTH_ENABLED', '').lower() == 'true'

        # Only import TokenManager if auth is enabled
        if self.auth_enabled:
            try:
                from oai_mcp_server_core.utils.token_manager import TokenManager
                self.token_manager = TokenManager()
            except ImportError as e:
                self.logger.error(f"Failed to import TokenManager: {e}")
                self.token_manager = None

    async def __call__(self, context: MiddlewareContext, call_next):
        """
        Middleware execution logic.
        
        Args:
            context: Middleware context
            call_next: Next middleware in chain
            
        Returns:
            Response from next middleware
        """
        # Skip authentication if not enabled
        if not self.auth_enabled:
            return await call_next(context)

        # If there's no request context, we're likely in a non-HTTP transport (e.g., stdio).
        # In this case, we bypass header-based authentication.
        if not context.fastmcp_context.request_context:
            self.logger.info("No HTTP request context found, bypassing authentication (stdio mode?).")
            return await call_next(context)

        # Skip authentication for health and info endpoints
        request = context.fastmcp_context.request_context.request
        if hasattr(request, 'url') and request.url.path in ['/health', '/info', '/', '/debug/env']:
            return await call_next(context)

        # Get original environ for FORCE_AUTH check
        original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ

        if hasattr(request, 'url') and 'localhost' in str(request.url) and original_environ.get('FORCE_AUTH',
                                                                                                'false').lower() == 'false':
            self.logger.info("Skipping authentication for localhost request")
            return await call_next(context)

        # Get headers
        headers = get_http_headers()
        api_token = headers.get('api-token') or headers.get('api_token') or headers.get('authorization')

        # If authorization header, strip "Bearer " prefix if present
        if api_token and api_token.lower().startswith('bearer '):
            api_token = api_token[7:]

        # Validate token
        if not api_token:
            self.logger.warning(f"Authentication failed: No API token provided for {self.server_name}")
            # We use Starlette's HTTPException here because it's caught by the server framework to return a proper 401 response
            from starlette.exceptions import HTTPException
            raise HTTPException(status_code=401, detail="API token required")

        # Validate token using TokenManager
        if not self.token_manager:
            self.logger.error("TokenManager not available")
            from starlette.exceptions import HTTPException
            raise HTTPException(status_code=500, detail="Authentication service unavailable")

        is_valid = self.token_manager.validate_token(self.server_name, api_token)

        if not is_valid:
            self.logger.warning(f"Authentication failed: Invalid or expired token for {self.server_name}")
            from starlette.exceptions import HTTPException
            raise HTTPException(status_code=401, detail="Invalid or expired API token")

        # Token is valid, proceed with request
        self.logger.info(f"Authentication successful for {self.server_name}")
        return await call_next(context)
