import os
from typing import Optional, Dict
from oai_mcp_server_core.core.context import request_env, RequestAwareEnviron


def get_request_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Explicitly get a request-scoped environment variable.

    This is optional since os.environ.get() will automatically check
    request context when using the RequestAwareEnviron wrapper.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Value from request context, or os.environ, or default
    """
    req_env = request_env.get()
    if key in req_env or key.upper() in req_env:
        return req_env.get(key) or req_env.get(key.upper())

    # Access original environ
    original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
    return original_environ.get(key, default)


def get_all_request_env() -> Dict[str, str]:
    """
    Get all environment variables merged with request context.
    Request context values override os.environ values.

    Returns:
        Dictionary of all environment variables
    """
    original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
    base_env = dict(original_environ)
    req_env = request_env.get()
    base_env.update(req_env)
    return base_env
