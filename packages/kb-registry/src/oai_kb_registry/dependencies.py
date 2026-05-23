"""
FastAPI dependency injection for KB Registry.

Infra auto-start configuration uses a three-level precedence chain:

  1. Module-level variables (set directly by cli.py before uvicorn starts)
  2. Environment variables  (AUTO_START_INFRA, INFRA_COMPOSE_FILE, INFRA_STARTUP_TIMEOUT)
  3. Built-in defaults      (False / bundled compose file / 60 s)

This means the CLI flag always wins over a pre-set env var, and both win
over the hard-coded default.  The env var path still works for deployments
that never use the CLI (e.g. Docker / docker-compose).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, Request

from oai_kb_registry.services.db.database_logger import KBDatabaseLogger
from oai_kb_registry.services.kb_registry import KBRegistry
from oai_kb_registry.security.dependencies import _validate_token

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_kb_registry: Optional[KBRegistry] = None
_db_logger:   Optional[KBDatabaseLogger] = None
_logger:      Optional[logging.Logger] = None

# ---------------------------------------------------------------------------
# Infra config — set by cli.py before uvicorn starts.
# None means "not set by code; fall back to env var / default".
# ---------------------------------------------------------------------------

_auto_start_infra:     Optional[bool] = None   # cli --auto-start-infra
_infra_compose_file:   Optional[str]  = None   # cli --infra-compose-file
_infra_startup_timeout: Optional[int] = None   # cli --infra-startup-timeout

# ---------------------------------------------------------------------------
# Infra shutdown tracking — populated during initialize_registry() so that
# close_registry() can mirror the exact compose file + project used at startup.
# ---------------------------------------------------------------------------

_infra_started:           bool          = False
_infra_compose_file_used: Optional[Path] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def initialize_registry(log: Optional[logging.Logger] = None) -> KBRegistry:
    """Initialise the KB registry on application startup.

    If auto-start infra is enabled (via CLI flag or AUTO_START_INFRA env var),
    brings up the infra docker-compose project (postgres + valkey, and
    optionally pgvector + chromadb) before the database logger attempts to
    connect.  Falls back to SQLite automatically if Postgres is still
    unreachable.

    Precedence for each setting:
        module-level variable  >  environment variable  >  built-in default
    """
    global _kb_registry, _db_logger, _logger, _infra_started, _infra_compose_file_used

    _logger = log or logging.getLogger(__name__)

    # --- Resolve infra settings (module-level > env var > default) -------
    auto_start: bool = (
        _auto_start_infra
        if _auto_start_infra is not None
        else os.environ.get("AUTO_START_INFRA", "false").lower() == "true"
    )

    _logger.info(
        "auto_start_infra: %s  "
        "(source: %s | to enable: --auto-start-infra flag or AUTO_START_INFRA=true env var)",
        "ENABLED" if auto_start else "DISABLED",
        "code" if _auto_start_infra is not None
        else "env"  if os.environ.get("AUTO_START_INFRA")
        else "default",
    )

    # --- Optional: auto-start infra Docker services before DB init -------
    #
    # Sequence:
    #  1. start_services() — runs `docker compose up -d --wait <services>`
    #     using the bundled (or user-provided) static compose file.
    #  2. wait_for_postgres() — async TCP safety-net for Compose < v2.4.
    #  3. Only then initialize the DB logger so asyncpg finds Postgres ready.
    #
    if auto_start:
        try:
            from oai_kb_registry.services.infra_manager import InfraManager, _BUNDLED_COMPOSE  # noqa: PLC0415
        except ImportError as exc:
            _logger.error("auto_start_infra: failed to import InfraManager: %s", exc)
            InfraManager   = None  # type: ignore[assignment]
            _BUNDLED_COMPOSE = None  # type: ignore[assignment]

        if InfraManager is not None:
            # Resolve compose file: module-level > env var > bundled default
            compose_path: Optional[str] = (
                _infra_compose_file
                if _infra_compose_file is not None
                else os.environ.get("INFRA_COMPOSE_FILE")
            )
            compose_file = Path(compose_path) if compose_path else _BUNDLED_COMPOSE

            # Resolve timeout: module-level > env var > 60 s
            timeout: int = (
                _infra_startup_timeout
                if _infra_startup_timeout is not None
                else int(os.environ.get("INFRA_STARTUP_TIMEOUT", "60"))
            )

            _logger.info(
                "auto_start_infra: starting postgres + valkey via %s (timeout: %ds)",
                compose_file, timeout,
            )

            try:
                InfraManager.start_services(compose_file=compose_file)
                _logger.info("auto_start_infra: docker compose up completed successfully")
                _infra_started          = True
                _infra_compose_file_used = compose_file
            except Exception as exc:
                _logger.error("auto_start_infra: docker compose startup failed: %s", exc)

            try:
                await InfraManager.wait_for_postgres(timeout=timeout)
            except TimeoutError as exc:
                _logger.error("auto_start_infra: Postgres did not become ready: %s", exc)
            except Exception as exc:
                _logger.warning("auto_start_infra: Postgres readiness check failed: %s", exc)

    # --- Initialise database logger ---------------------------------------
    _db_logger = KBDatabaseLogger(logger=_logger)
    initialized = await _db_logger.initialize()

    if not initialized:
        _logger.warning("KB database logger not initialised — running in no-persistence mode")

    # --- Initialise registry service -------------------------------------
    # Forward the resolved infra settings so KBRegistry can provision
    # vector store containers on demand at registration time.
    resolved_compose_file = (
        _infra_compose_file
        if _infra_compose_file is not None
        else os.environ.get("INFRA_COMPOSE_FILE")
    )
    resolved_timeout = (
        _infra_startup_timeout
        if _infra_startup_timeout is not None
        else int(os.environ.get("INFRA_STARTUP_TIMEOUT", "60"))
    )
    _kb_registry = KBRegistry(
        db_logger=_db_logger,
        log=_logger,
        auto_start_infra=auto_start,
        infra_compose_file=resolved_compose_file,
        infra_startup_timeout=resolved_timeout,
    )

    # --- Restore existing KBs from database ------------------------------
    # Re-initialises vector store containers and warms up the in-memory cache
    # for every KB that was registered before this restart.  Non-fatal: a
    # single failing KB is logged and skipped; healthy KBs still come up.
    try:
        await _kb_registry.restore_from_db()
    except Exception as exc:
        _logger.error("restore_from_db failed: %s", exc)

    return _kb_registry


async def close_registry() -> None:
    """Close the KB registry and tear down any infra this process started.

    Shutdown sequence (inverse of startup order)
    ---------------------------------------------
    1. ``registry.close()``
         a. Clear in-memory vector store cache
         b. Stop on-demand vector store containers (pgvector / chromadb)
            — Phase 2 infra started at KB registration time
         c. Close metadata DB connection
    2. Stop core infra (postgres + valkey)
         — Phase 1 infra started at process startup

    Mirrors the skills-registry / agent-registry pattern: each phase is only
    torn down when this process was the one that brought it up.
    """
    global _kb_registry, _infra_started, _infra_compose_file_used
    _log = _logger or logging.getLogger(__name__)

    # 1. Close the registry — this stops on-demand vector store containers
    #    (pgvector, chromadb) before closing the DB connection.
    if _kb_registry:
        _log.info("Shutting down KB Registry (vector stores → DB)…")
        await _kb_registry.close()
        _log.info("KB Registry closed.")

    # 2. Stop core infra (postgres, valkey) — only if we started it.
    if _infra_started:
        try:
            from oai_kb_registry.services.infra_manager import InfraManager  # noqa: PLC0415
            _log.info(
                "auto_stop_infra: shutting down core infra (postgres, valkey) "
                "(compose file: %s)",
                _infra_compose_file_used,
            )
            InfraManager.stop_services(compose_file=_infra_compose_file_used)
            _infra_started           = False
            _infra_compose_file_used = None
            _log.info("auto_stop_infra: core infra stopped successfully.")
        except Exception as exc:
            _log.error("auto_stop_infra: failed to stop core infra: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_registry() -> KBRegistry:
    """FastAPI dependency: return the initialised KBRegistry instance."""
    if _kb_registry is None:
        raise RuntimeError("KB Registry not initialised. Call initialize_registry() first.")
    return _kb_registry


async def verify_bearer_token(request: Request) -> bool:
    """FastAPI dependency — authenticate every protected endpoint.

    Delegates to the shared token validator in the security sub-package.
    Authentication can be disabled with ``KB_AUTH_ENABLED=false``.
    """
    _validate_token(request)
    return True


async def get_auth_user(
    request: Request,
    _auth: bool = Depends(verify_bearer_token),
) -> str:
    """Return the authenticated user identifier from ``request.state``."""
    email   = getattr(request.state, "user_email", None)
    user_id = getattr(request.state, "user_id",    None)
    return email or user_id or "authenticated_user"
