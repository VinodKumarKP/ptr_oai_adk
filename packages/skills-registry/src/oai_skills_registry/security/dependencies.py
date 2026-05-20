"""
Skills-registry security dependencies.

Token validation for FastAPI endpoints.  Two token types are supported:

SAML tokens
    Base64-encoded SAML assertions issued by an Identity Provider.
    Validated by :class:`oai_platform_core.security.saml_token_validation.TokenValidator`.
    On success the request state receives ``user_role`` and ``user_email``.

API tokens
    Opaque tokens managed by :class:`oai_platform_core.security.TokenManager`.
    On success the request state receives ``user_id`` and ``user_role``.

Authentication behaviour is controlled by three environment variables:

``SKILLS_AUTH_ENABLED``
    Set to ``"false"`` to disable all authentication (development / testing).
    Default: ``"true"``.

``FORCE_AUTH``
    When ``"false"`` (default), requests from trusted local peers (loopback,
    ``host.docker.internal``) skip authentication.  Set to ``"true"`` to
    require a valid token even from local callers.

``SAML_PUBLIC_KEY_PATH``
    Optional explicit path to the RSA public key used for SAML signature
    verification.  Falls back to the bundled key shipped inside
    ``oai-platform-core`` when unset.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from oai_platform_core.security import TokenManager, is_saml_token, extract_bearer_token
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError

__all__ = ["verify_api_key", "_validate_token"]

# OpenAPI security scheme declaration (shows padlock in Swagger UI)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Shared TokenManager — one connection pool per process.
_token_manager: Optional[TokenManager] = None


def _get_token_manager() -> TokenManager:
    """Return the shared TokenManager, creating it lazily on first call."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(db_path_name="skills_registry_tokens.db")
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
    auth_enabled = os.environ.get("SKILLS_AUTH_ENABLED", "true").lower() == "true"
    if not auth_enabled:
        return

    # 3. Trusted-peer bypass (loopback / docker internal) unless FORCE_AUTH=true
    _LOCAL_PEERS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "host.docker.internal"}
    peer = request.client.host if request.client else ""
    if peer in _LOCAL_PEERS and os.environ.get("FORCE_AUTH", "false").lower() != "true":
        return

    # 4. Extract token from request headers (supports api-token, api_token,
    #    x-api-key, and Authorization: Bearer …)
    token = extract_bearer_token(dict(request.headers))
    if not token:
        raise HTTPException(status_code=401, detail="API token required")

    # 5a. SAML token path
    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
            result = validator.validate_token_and_get_role(token)
            if result.is_valid:
                request.state.user_role = result.role
                request.state.user_email = result.email
                request.state.user_id = result.email   # normalise to user_id
                return
            raise HTTPException(
                status_code=401,
                detail=result.error_message or "Invalid SAML token",
            )
        except TokenValidationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:                       # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail="SAML token validation service unavailable",
            ) from exc

    # 5b. Platform API token path
    token_manager = _get_token_manager()
    user_info = token_manager.validate_token("skills-registry", token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid or expired API token")
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
