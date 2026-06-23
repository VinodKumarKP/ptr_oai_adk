import os
from typing import Dict
from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from oai_mcp_server_core.utils.logger_utils import get_logger
from oai_mcp_server_core.core.context import request_env


class HeaderCaptureMiddleware(Middleware):
    """
    Middleware that captures HTTP headers and makes them available as
    environment variables in a request-scoped context.

    This prevents race conditions when handling concurrent requests with
    different authentication tokens or configuration values.
    """

    def __init__(self, header_prefix: str = "", transform_keys: bool = True):
        """
        Initialize the middleware.

        Args:
            header_prefix: Only capture headers with this prefix (e.g., "X-Custom-")
            transform_keys: Convert header names to env-style (uppercase, - to _)
        """
        super().__init__()
        self.header_prefix = header_prefix.lower()
        self.transform_keys = transform_keys
        self.logger = get_logger()

    def _transform_header_key(self, key: str) -> str:
        """
        Transform header name to environment variable style.
        
        Args:
            key: Header name
            
        Returns:
            Transformed key (uppercase, hyphens replaced with underscores)
        """
        if self.transform_keys:
            # Convert to uppercase and replace hyphens with underscores
            return key.upper().replace('-', '_')
        return key

    def _sanitize_for_logging(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize sensitive values for logging.
        
        Args:
            headers: Dictionary of headers
            
        Returns:
            Sanitized dictionary
        """
        sensitive_keys = {'AUTHORIZATION', 'API_KEY', 'API_TOKEN', 'TOKEN', 'SECRET', 'PASSWORD'}
        sanitized = {}
        for key, value in headers.items():
            key_upper = key.upper()
            if any(sensitive in key_upper for sensitive in sensitive_keys):
                # Show only first/last 4 chars for sensitive values
                if len(value) > 8:
                    sanitized[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = value
        return sanitized

    async def __call__(self, context: MiddlewareContext, call_next):
        """
        Middleware execution logic.
        
        Args:
            context: Middleware context
            call_next: Next middleware in chain
            
        Returns:
            Response from next middleware
        """
        headers = get_http_headers()

        # Filter and transform headers
        env_dict = {}
        self.logger.info(f"Capturing environment variables from {len(headers)} HTTP header(s)")
        for key, value in headers.items():
            # Apply prefix filter if specified
            if self.header_prefix and not key.lower().startswith(self.header_prefix):
                continue

            transformed_key = self._transform_header_key(key)
            # if value contains ${VAR} or $VAR, resolve from original environ
            if ('$' in value) and (('{' in value and '}' in value) or ' ' not in value):
                try:
                    resolved_value = os.path.expandvars(value)
                    env_dict[transformed_key] = resolved_value
                except Exception as e:
                    env_dict[transformed_key] = value
            else:
                env_dict[transformed_key] = value

        # Log the request context (sanitized)
        if env_dict:
            sanitized = self._sanitize_for_logging(env_dict)
            self.logger.info(f"MCP Request context: {sanitized}")

        # Set in request-scoped context
        token = request_env.set(env_dict)

        try:
            result = await call_next(context)
            return result
        finally:
            # Clean up context after request completes
            request_env.reset(token)
            self.logger.info("MCP Request context cleaned up")
