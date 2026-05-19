"""
agent-registry token store — thin compatibility shim.

All token-management logic lives in ``oai_platform_core.security.TokenManager``.
This subclass exists solely to pin the service-specific redislite DB file name
so that existing call-sites (``TokenManager()``) continue to work unchanged.

Canonical source: packages/platform-core/src/oai_platform_core/security/token_manager.py
"""

from typing import Optional

from oai_platform_core.security import TokenManager as _CoreTokenManager

__all__ = ["TokenManager"]


class TokenManager(_CoreTokenManager):
    """agent-registry token store.

    Delegates entirely to :class:`oai_platform_core.security.TokenManager`.
    The only customisation is the default redislite database file name,
    which keeps tokens isolated from other services on the same machine.

    Default redislite path: ``~/.oai/agent_registry_tokens.db``
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
            db_path_name="agent_registry_tokens.db",
        )
