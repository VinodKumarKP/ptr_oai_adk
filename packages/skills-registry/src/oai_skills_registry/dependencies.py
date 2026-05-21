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

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.skills_registry import SkillsRegistry
from oai_skills_registry.security.dependencies import _validate_token

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

# ---------------------------------------------------------------------------
# Infra shutdown tracking — populated during initialize_registry() so that
# close_registry() can mirror the exact compose file + project used at startup.
# ---------------------------------------------------------------------------

_infra_started: bool = False               # True only when we actually brought infra up
_infra_compose_file_used: Optional[Path] = None   # the resolved compose path used at startup

security = HTTPBearer(auto_error=False)


async def initialize_registry(logger: Optional[logging.Logger] = None) -> SkillsRegistry:
    """Initialize the skills registry.

    If auto-start infra is enabled (via CLI flag or AUTO_START_INFRA env var),
    brings up the infra docker-compose project (postgres + valkey) before the
    database logger attempts to connect.
    Falls back to SQLite automatically if Postgres is still unreachable.

    Precedence for each setting:
        module-level variable  >  environment variable  >  built-in default
    """
    global _skills_registry, _db_logger, _logger, _infra_started, _infra_compose_file_used

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
                # Record that WE started infra so close_registry() can shut it down.
                _infra_started = True
                _infra_compose_file_used = compose_file
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


async def verify_bearer_token(request: Request) -> bool:
    """FastAPI dependency — authenticate every protected endpoint.

    Supports both SAML tokens and platform API tokens via
    :func:`oai_skills_registry.security.dependencies._validate_token`.

    On success, ``request.state`` is populated with:
    - ``user_id``    — username or e-mail
    - ``user_role``  — role string from token
    - ``user_email`` — e-mail (SAML tokens only)

    Authentication can be disabled by setting ``SKILLS_AUTH_ENABLED=false``.
    Trusted local peers (loopback / docker-internal) bypass auth unless
    ``FORCE_AUTH=true``.

    Raises:
        HTTPException 401: token missing or invalid.
        HTTPException 500: SAML validation service unavailable.
    """
    _validate_token(request)
    return True


async def get_auth_user(
    request: Request,
    _auth: bool = Depends(verify_bearer_token),
) -> str:
    """Return the authenticated user's identifier from ``request.state``.

    Prefers ``user_email`` (set by SAML flow) then falls back to ``user_id``
    (set by API-token flow).  Returns ``"authenticated_user"`` when auth is
    disabled or the request originates from a trusted peer.
    """
    email = getattr(request.state, "user_email", None)
    user_id = getattr(request.state, "user_id", None)
    return email or user_id or "authenticated_user"


async def close_registry() -> None:
    """Close the skills registry and tear down any infra we started.

    Mirrors the agent-registry / mcp-registry pattern: if this process brought
    up the Docker Compose infra on startup (``_infra_started`` is ``True``),
    it is responsible for bringing it back down on shutdown.
    """
    global _skills_registry, _infra_started, _infra_compose_file_used

    # 1. Close the registry (and its DB connection) first.
    if _skills_registry:
        await _skills_registry.close()

    # 2. Shut down Docker Compose infra only if this process started it.
    if _infra_started:
        try:
            from oai_skills_registry.services.infra_manager import InfraManager
            _log = _logger or logging.getLogger(__name__)
            _log.info(
                "auto_stop_infra: shutting down infra docker compose project "
                "(compose file: %s)",
                _infra_compose_file_used,
            )
            InfraManager.stop_services(compose_file=_infra_compose_file_used)
            _infra_started = False
            _infra_compose_file_used = None
        except Exception as exc:
            _log = _logger or logging.getLogger(__name__)
            _log.error("auto_stop_infra: failed to stop infra services: %s", exc)
