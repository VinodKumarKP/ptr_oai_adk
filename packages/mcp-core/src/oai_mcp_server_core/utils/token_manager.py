"""
mcp-server-core token store — thin compatibility shim.

All token-management logic lives in ``oai_platform_core.security.TokenManager``.
This subclass exists solely to pin the service-specific redislite DB file name
so that existing call-sites (``TokenManager()``) continue to work unchanged.

Migration note
--------------
The previous mcp-core implementation required a live Redis/Valkey instance and
had no fallback.  This shim inherits the automatic redislite fallback from the
canonical TokenManager, making the MCP server work out-of-the-box for
development/demo without a Redis server.

validate_token() return type change
------------------------------------
The old implementation returned ``bool``.  The canonical implementation returns
``Optional[Dict[str, str]]`` (``None`` on failure, ``{"user_id": …, "role_id": …}``
on success).  All existing callers that do ``if token_manager.validate_token(…)``
continue to work correctly — ``None`` is falsy, a non-empty dict is truthy.

generate_token() return type change
-------------------------------------
The old implementation returned ``Dict[str, Any]`` with token, user_id, role_id
and ttl_seconds.  The canonical implementation returns only the token ``str``.
Call-sites in routes.py have been updated to reconstruct the full response dict.

Canonical source: packages/platform-core/src/oai_platform_core/security/token_manager.py
"""

from typing import Optional

from oai_platform_core.security import TokenManager as _CoreTokenManager

__all__ = ["TokenManager"]


class TokenManager(_CoreTokenManager):
    """mcp-server-core token store.

    Delegates entirely to :class:`oai_platform_core.security.TokenManager`.
    The only customisation is the default redislite database file name,
    which keeps tokens isolated from other services on the same machine.

    Default redislite path: ``~/.oai/mcp_server_tokens.db``
    (overridable via ``REDISLITE_DB_PATH`` env var or ``redislite_path`` arg)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redislite_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
            db=db,
            redislite_path=redislite_path,
            db_path_name="mcp_server_tokens.db",
        )
