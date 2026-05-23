"""
KB-registry security dependencies.

Token validation for FastAPI endpoints.  Two token types are supported:

SAML tokens
    Base64-encoded SAML assertions issued by an Identity Provider.
    Validated by :class:`oai_platform_core.security.saml_token_validation.TokenValidator`.
    On success the request state receives ``user_role`` and ``user_email``.

API tokens
    Opaque tokens managed by :class:`oai_platform_core.security.TokenManager`.
    On success the request state receives ``user_id`` and ``user_role``.

Authentication behaviour is controlled by three environment variables:

``KB_AUTH_ENABLED``
    Set to ``"false"`` to disable all authentication (development / testing).
    Default: ``"true"``.

``FORCE_AUTH``
    When ``"false"`` (default), requests from trusted local peers (loopback,
    ``host.docker.internal``, Docker bridge subnets) skip authentication.
    Set to ``"true"`` to require a valid token even from local callers.

``SAML_PUBLIC_KEY_PATH``
    Optional explicit path to the RSA public key used for SAML signature
    verification.  Falls back to the bundled key shipped inside
    ``oai-platform-core`` when unset.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from oai_platform_core.security import (
    TokenManager,
    is_saml_token,
    extract_bearer_token,
    is_trusted_peer,
    TRUSTED_PEER_NAMES,
    trusted_subnets,
)
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError

logger = logging.getLogger(__name__)

__all__ = ["verify_api_key", "_validate_token"]

# OpenAPI security scheme declaration (shows padlock in Swagger UI)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Shared TokenManager — one connection pool per process.
_token_manager: Optional[TokenManager] = None


def _get_token_manager() -> TokenManager:
    """Return the shared TokenManager, creating it lazily on first call."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(db_path_name="kb_registry_tokens.db")
    return _token_manager


# ---------------------------------------------------------------------------
# Health / status bypass paths
# ---------------------------------------------------------------------------

_BYPASS_PATHS = ("/health", "/status", "/")


def _is_bypass_path(path: str) -> bool:
    """Return True for endpoints that never require authentication."""
    return any(path.endswith(bp) for bp in _BYPASS_PATHS)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def _validate_token(request: Request) -> None:
    """Validate the API token (SAML or platform) on the incoming request.

    Mutates ``request.state`` on success:
    - SAML tokens   → ``user_role``, ``user_email``, ``user_id`` (=email)
    - API tokens    → ``user_id``, ``user_role``

    Raises:
        HTTPException 401: token absent or invalid.
        HTTPException 500: SAML validation service unavailable.
    """
    # 1. Always let health/status through
    if _is_bypass_path(request.url.path):
        return

    # 2. Honour global auth toggle
    auth_enabled = os.environ.get("KB_AUTH_ENABLED", "true").lower() == "true"
    if not auth_enabled:
        return

    # 3. Trusted-peer bypass (loopback / Docker bridge) unless FORCE_AUTH=true
    peer = request.client.host if request.client else ""
    force_auth = os.environ.get("FORCE_AUTH", "false").lower() == "true"

    if is_trusted_peer(peer) and not force_auth:
        logger.debug("Auth skipped — peer %s is a trusted local/Docker address", peer)
        return

    if not is_trusted_peer(peer):
        logger.info(
            "Auth required — peer %s is not a trusted address  "
            "(trusted names: %s; trusted subnets: %s)  "
            "Set KB_AUTH_ENABLED=false to disable auth, or "
            "add extra CIDRs via TRUSTED_SUBNETS env var.",
            peer, sorted(TRUSTED_PEER_NAMES), [str(n) for n in trusted_subnets()],
        )

    # 4. Extract token from request headers
    token = extract_bearer_token(dict(request.headers))
    if not token:
        logger.warning(
            "Auth rejected — no Bearer token  method=%s path=%s client=%s  headers=%s",
            request.method, request.url.path, peer, list(request.headers.keys()),
        )
        raise HTTPException(status_code=401, detail="API token required")

    # 5a. SAML token path
    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
            result = validator.validate_token_and_get_role(token)
            if result.is_valid:
                logger.info("Auth OK (SAML) — user=%s role=%s client=%s", result.email, result.role, peer)
                request.state.user_role = result.role
                request.state.user_email = result.email
                request.state.user_id = result.email
                return
            logger.warning("Auth rejected (SAML) — %s  client=%s", result.error_message, peer)
            raise HTTPException(
                status_code=401,
                detail=result.error_message or "Invalid SAML token",
            )
        except TokenValidationError as exc:
            logger.warning("Auth rejected (SAML TokenValidationError) — %s  client=%s", exc, peer)
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("SAML validation service unavailable — %s  client=%s", exc, peer)
            raise HTTPException(
                status_code=500,
                detail="SAML token validation service unavailable",
            ) from exc

    # 5b. Platform API token path
    token_manager = _get_token_manager()
    user_info = token_manager.validate_token("kb-registry", token)
    if not user_info:
        logger.warning("Auth rejected (API key) — invalid/expired  client=%s", peer)
        raise HTTPException(status_code=401, detail="Invalid or expired API token")
    logger.info(
        "Auth OK (API key) — user_id=%s role=%s client=%s",
        user_info.get("user_id"), user_info.get("role_id"), peer,
    )
    request.state.user_id = user_info.get("user_id")
    request.state.user_role = user_info.get("role_id")


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request) -> bool:
    """FastAPI dependency — enforce authentication on every protected request.

    Returns:
        ``True`` when the request is permitted (token valid, auth disabled,
        or path is whitelisted).

    Raises:
        HTTPException 401/500: see :func:`_validate_token`.
    """
    if _is_bypass_path(request.url.path):
        return True
    _validate_token(request)
    return True
