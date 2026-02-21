import os
from contextvars import ContextVar
from typing import Dict, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable for request-specific environment (async-safe)
request_env: ContextVar[Dict[str, str]] = ContextVar('request_env', default={})


class RequestAwareEnviron:
    """
    Wrapper around os.environ that checks request context first.
    This allows request-specific headers to override environment variables
    without affecting other concurrent requests.
    """

    def __init__(self, original_environ):
        self._original = original_environ
        self._is_wrapped = True

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with request context priority"""
        # Check request context first
        req_env = request_env.get()
        if key in req_env or key.upper() in req_env:
            return req_env.get(key) or req_env.get(key.upper())
        # Fallback to original os.environ
        return self._original.get(key, default)

    def __getitem__(self, key: str) -> str:
        """Dictionary-style access to environment variables"""
        req_env = request_env.get()
        if key in req_env:
            return req_env[key]
        if str(key).upper() in req_env:
            return req_env[str(key).upper()]
        return self._original[str(key)]

    def __setitem__(self, key: str, value: str):
        """Set environment variable in original environ"""
        self._original[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in request context or original environ"""
        req_env = request_env.get()
        return key in req_env or key.upper() in req_env or key in self._original

    def __delitem__(self, key: str):
        """Delete from original environ"""
        del self._original[key]

    def __iter__(self):
        """Return iterator over all keys (merged from request context and original)"""
        req_env = request_env.get()
        all_keys = set(self._original.keys()) | set(req_env.keys())
        return iter(all_keys)

    def __len__(self):
        """Return total number of environment variables"""
        req_env = request_env.get()
        all_keys = set(self._original.keys()) | set(req_env.keys())
        return len(all_keys)

    def keys(self):
        """Return all keys (merged from request context and original)"""
        req_env = request_env.get()
        all_keys = set(self._original.keys()) | set(req_env.keys())
        return all_keys

    def items(self):
        """Return all items (merged from request context and original)"""
        req_env = request_env.get()
        merged = dict(self._original)
        merged.update(req_env)
        return merged.items()

    def values(self):
        """Return all values"""
        return [v for k, v in self.items()]

    # Delegate other methods to original
    def __getattr__(self, name):
        return getattr(self._original, name)

    def __repr__(self):
        return f"RequestAwareEnviron(request_vars={len(request_env.get())}, total_vars={len(self.keys())})"


def setup_request_isolation(logger):
    """
    Setup request-aware environment wrapper.
    This makes os.environ automatically use request-scoped values.
    """
    # Only wrap once
    if not isinstance(os.environ, RequestAwareEnviron):
        original_environ = os.environ
        os.environ = RequestAwareEnviron(original_environ)
        logger.info("Request isolation enabled: os.environ is now request-aware")
    else:
        logger.info("Request isolation already enabled")


def get_original_environ():
    """Get the original os.environ, unwrapping if needed"""
    if isinstance(os.environ, RequestAwareEnviron):
        return os.environ._original
    return os.environ


def transform_header_key(key: str) -> str:
    """Transform header name to environment variable style"""
    return key.upper().replace('-', '_')


def sanitize_for_logging(headers: Dict[str, str]) -> Dict[str, str]:
    """Sanitize sensitive values for logging"""
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


class HeaderCaptureMiddleware(BaseHTTPMiddleware):
    """Middleware to capture headers and inject them into request context."""

    def __init__(self, app, logger, enable_request_isolation=True):
        super().__init__(app)
        self.logger = logger
        self.enable_request_isolation = enable_request_isolation

    async def dispatch(self, request: Request, call_next):
        """Capture HTTP headers and make them available in request context"""
        headers = dict(request.headers)

        # Transform headers to environment variable format
        env_dict = {}
        for key, value in headers.items():
            transformed_key = transform_header_key(key)
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
        if env_dict and self.enable_request_isolation:
            sanitized = sanitize_for_logging(env_dict)
            self.logger.info(f"Agent Request context: {sanitized}")

            # if env_dict contains anything other than standard headers, set agent_reinitialize to true
            env_dict['AGENT_REINITIALIZE'] = 'false'
            standard_headers = {
                'HOST', 'USER_AGENT', 'ACCEPT', 'ACCEPT_ENCODING',
                'ACCEPT_LANGUAGE', 'CONNECTION', 'CACHE_CONTROL',
                'AGENT_REINITIALIZE', 'CONTENT_LENGTH', 'CONTENT_TYPE',
                'POSTMAN_TOKEN', 'AUTHORIZATION'
            }
            if any(k not in standard_headers for k in env_dict.keys()):
                self.logger.info(f"Agent Request context contains non-standard headers: {env_dict}")
                env_dict['AGENT_REINITIALIZE'] = 'true'

        # Set in request-scoped context
        token = request_env.set(env_dict)

        try:
            response = await call_next(request)
            return response
        finally:
            # Clean up context after request completes
            if self.enable_request_isolation:
                request_env.reset(token)
                self.logger.info("Agent Request context cleaned up")
