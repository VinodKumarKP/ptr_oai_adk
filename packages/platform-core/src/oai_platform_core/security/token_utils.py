"""
oai_platform_core.security.token_utils — token extraction helpers.

:func:`extract_bearer_token` inspects a headers mapping for the set of
header names that OAI services accept as API tokens.  It is framework-
agnostic (accepts any ``Mapping[str, str]``) so it works with FastAPI's
``request.headers``, raw WSGI/ASGI environ dicts, and test stubs alike.
"""

from __future__ import annotations

from typing import Mapping, Optional

__all__ = ["extract_bearer_token"]

# Header names checked in priority order.  All are case-insensitive when
# accessed through Starlette/FastAPI Headers but we use lower-case to
# match the stored form.
_TOKEN_HEADERS = ("api-token", "api_token", "x-api-key")


def extract_bearer_token(headers: Mapping[str, str]) -> Optional[str]:
    """Extract an API token from an HTTP headers mapping.

    Checks, in order:
    1. ``api-token`` header
    2. ``api_token`` header
    3. ``x-api-key`` header
    4. ``authorization`` header — strips the ``Bearer `` prefix when present

    Args:
        headers: Any ``Mapping[str, str]`` of HTTP headers.  FastAPI's
                 ``request.headers`` satisfies this interface directly.

    Returns:
        The raw token string, or ``None`` if no token header was found.
    """
    for name in _TOKEN_HEADERS:
        value = headers.get(name)
        if value:
            return value

    authorization = headers.get("authorization")
    if authorization:
        if authorization.lower().startswith("bearer "):
            return authorization[7:]
        return authorization

    return None
