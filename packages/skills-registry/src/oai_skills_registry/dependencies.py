"""
FastAPI dependency injection for Skills Registry.

Infra auto-start configuration uses a three-level precedence chain:

  1. Module-level variables (set directly by cli.py before uvicorn starts)
  2. Environment variables  (AUTO_START_INFRA, INFRA_COMPOSE_FILE, INFRA_STARTUP_TIMEOUT)
  3. Built-in defaults      (False / bundled compose file / 60 s)

This means the CLI flag always wins over a pre-set env var, and both win
over the hard-coded default.  The env var path still works for deployments
that never use the CLI (e.g. Docker / docker-compose).
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.skills_registry import SkillsRegistry

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_skills_registry: Optional[SkillsRegistry] = None
_db_logger: Optional[SkillsDatabaseLogger] = None
_logger: Optional[logging.Logger] = None

# ---------------------------------------------------------------------------
# Infra config — set by cli.py before uvicorn starts.
# None means "not set by code; fall back to env var / default".
# ---------------------------------------------------------------------------

_auto_start_infra: Optional[bool] = None      # cli --auto-start-infra
_infra_compose_file: Optional[str] = None     # cli --infra-compose-file
_infra_startup_timeout: Optional[int] = None  # cli --infra-startup-timeout

security = HTTPBearer()


async def initialize_registry(logger: Optional[logging.Logger] = None) -> SkillsRegistry:
    """Initialize the skills registry.

    If auto-start infra is enabled (via CLI flag or AUTO_START_INFRA env var),
    brings up the infra docker-compose project (postgres + valkey) before the
    database logger attempts to connect.
    Falls back to SQLite automatically if Postgres is still unreachable.

    Precedence for each setting:
        module-level variable  >  environment variable  >  built-in default
    """
    global _skills_registry, _db_logger, _logger

    _logger = logger or logging.getLogger(__name__)

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
        "code" if _auto_start_infra is not None else "env" if os.environ.get("AUTO_START_INFRA") else "default",
    )

    # --- Optional: auto-start infra Docker services before DB init -------
    #
    # Sequence:
    #  1. start_services() — runs `docker compose up -d --wait postgres valkey`
    #     using the bundled (or user-provided) static compose file.
    #  2. wait_for_postgres() — async TCP safety-net for Compose < v2.4.
    #  3. Only then initialize the DB logger so asyncpg finds Postgres ready.
    #
    if auto_start:
        try:
            from oai_skills_registry.services.infra_manager import InfraManager, _BUNDLED_COMPOSE
        except ImportError as exc:
            _logger.error("auto_start_infra: failed to import InfraManager: %s", exc)
            InfraManager = None  # type: ignore[assignment]
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
            except Exception as exc:
                _logger.error("auto_start_infra: docker compose startup failed: %s", exc)

            try:
                await InfraManager.wait_for_postgres(timeout=timeout)
            except TimeoutError as exc:
                _logger.error("auto_start_infra: Postgres did not become ready: %s", exc)
            except Exception as exc:
                _logger.warning("auto_start_infra: Postgres readiness check failed: %s", exc)

    # Initialize database logger
    _db_logger = SkillsDatabaseLogger(logger=_logger)
    initialized = await _db_logger.initialize()

    if not initialized:
        _logger.warning("Database logger not initialized - running in no-persistence mode")

    # Initialize skills registry
    _skills_registry = SkillsRegistry(db_logger=_db_logger, logger=_logger)

    return _skills_registry


def get_registry() -> SkillsRegistry:
    """Get the skills registry instance."""
    if _skills_registry is None:
        raise RuntimeError("Registry not initialized. Call initialize_registry() first.")
    return _skills_registry


async def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> HTTPAuthorizationCredentials:
    """Verify bearer token from Authorization header."""
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


async def get_auth_user(credentials: HTTPAuthorizationCredentials = Depends(verify_bearer_token)) -> str:
    """Extract username/identifier from token (simplified)."""
    # In production, this would validate the JWT and extract user info
    # For now, we'll just return a placeholder
    return "authenticated_user"


async def close_registry() -> None:
    """Close the skills registry."""
    global _skills_registry
    if _skills_registry:
        await _skills_registry.close()
