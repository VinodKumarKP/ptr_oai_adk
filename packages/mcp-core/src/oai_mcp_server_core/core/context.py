import os
from typing import Dict, Optional, Iterator, KeysView, ItemsView, ValuesView
from contextvars import ContextVar

# ============================================================================
# REQUEST-AWARE ENVIRONMENT VARIABLES
# ============================================================================

# Context variable for request-specific environment (async-safe)
request_env: ContextVar[Dict[str, str]] = ContextVar('request_env', default={})


class RequestAwareEnviron:
    """
    Wrapper around os.environ that checks request context first.
    This allows request-specific headers to override environment variables
    without affecting other concurrent requests.
    """

    def __init__(self, original_environ: os._Environ):
        self._original = original_environ
        self._is_wrapped = True

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable with request context priority.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Value from request context, or os.environ, or default
        """
        # Check request context first
        req_env = request_env.get()
        if key in req_env or key.upper() in req_env:
            return req_env.get(key) or req_env.get(key.upper())
        # Fallback to original os.environ
        return self._original.get(key, default)

    def __getitem__(self, key: str) -> str:
        """
        Dictionary-style access to environment variables.
        
        Args:
            key: Environment variable name
            
        Returns:
            Value from request context or os.environ
            
        Raises:
            KeyError: If key is not found
        """
        req_env = request_env.get()
        if key in req_env:
            return req_env[key]
        if str(key).upper() in req_env:
            return req_env[str(key).upper()]
        return self._original[str(key)]

    def __setitem__(self, key: str, value: str):
        """
        Set environment variable in original environ.
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        self._original[key] = value

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in request context or original environ.
        
        Args:
            key: Environment variable name
            
        Returns:
            True if key exists, False otherwise
        """
        req_env = request_env.get()
        return key in req_env or key.upper() in req_env or key in self._original

    def __delitem__(self, key: str):
        """
        Delete from original environ.
        
        Args:
            key: Environment variable name
        """
        del self._original[key]

    def __iter__(self) -> Iterator[str]:
        """
        Return iterator over all keys (merged from request context and original).
        
        Returns:
            Iterator over keys
        """
        req_env = request_env.get()
        all_keys = set(self._original.keys()) | set(req_env.keys())
        return iter(all_keys)

    def __len__(self) -> int:
        """
        Return total number of environment variables.
        
        Returns:
            Total count of environment variables
        """
        req_env = request_env.get()
        all_keys = set(self._original.keys()) | set(req_env.keys())
        return len(all_keys)

    def keys(self) -> KeysView[str]:
        """
        Return all keys (merged from request context and original).

        Returns:
            View of all keys
        """
        req_env = request_env.get()
        merged = dict(self._original)
        merged.update(req_env)
        return merged.keys()

    def items(self) -> ItemsView[str, str]:
        """
        Return all items (merged from request context and original).
        
        Returns:
            View of all items
        """
        req_env = request_env.get()
        merged = dict(self._original)
        merged.update(req_env)
        return merged.items()

    def values(self) -> ValuesView[str]:
        """
        Return all values.

        Returns:
            View of all values
        """
        req_env = request_env.get()
        merged = dict(self._original)
        merged.update(req_env)
        return merged.values()

    # Delegate other methods to original
    def __getattr__(self, name):
        return getattr(self._original, name)

    def __repr__(self):
        return f"RequestAwareEnviron(request_vars={len(request_env.get())}, total_vars={len(self.keys())})"
