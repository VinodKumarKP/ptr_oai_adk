"""
Agent-registry security dependencies.

Token validation for FastAPI endpoints.  The token extraction logic
(header parsing) is handled by the framework-agnostic
:func:`oai_platform_core.security.extract_bearer_token` helper; the
FastAPI-specific parts (HTTPException, request.state mutation) stay here.
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

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Shared TokenManager — one connection pool for the process lifetime.
_token_manager: Optional[TokenManager] = None



def _get_token_manager() -> TokenManager:
    """Return the shared TokenManager, creating it lazily on first call."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager(db_path_name="agent_registry_tokens.db")
    return _token_manager


def _validate_token(request: Request, agent_name: str) -> None:
    """Core token-validation logic for agent-registry endpoints.

    Args:
        request:    The FastAPI request being authenticated.
        agent_name: Name used to look up the token in the token store.

    Raises:
        HTTPException 401: token missing or invalid.
        HTTPException 500: auth service unavailable.
    """
    path = request.url.path
    peer = request.client.host if request.client else "<unknown>"
    method = request.method

    logger.debug("Auth check — %s %s  client=%s", method, path, peer)

    if "/status" in path:
        logger.debug("Auth skipped — status endpoint")
        return

    auth_enabled = os.environ.get("AGENT_AUTH_ENABLED", "true").lower() == "true"
    if not auth_enabled:
        logger.debug("Auth skipped — AGENT_AUTH_ENABLED=false")
        return

    force_auth = os.environ.get("FORCE_AUTH", "false").lower() != "false"

    if is_trusted_peer(peer) and not force_auth:
        logger.debug("Auth skipped — peer %s is a trusted local/Docker address", peer)
        return

    if not is_trusted_peer(peer):
        logger.info(
            "Auth required — peer %s is not a trusted address  "
            "(trusted names: %s; trusted subnets: %s)  "
            "Set AGENT_AUTH_ENABLED=false to disable auth, or "
            "add extra CIDRs via TRUSTED_SUBNETS env var.",
            peer,
            sorted(TRUSTED_PEER_NAMES),
            [str(n) for n in trusted_subnets()],
        )

    token = extract_bearer_token(request.headers)
    if not token:
        logger.warning(
            "Auth rejected — no Bearer token in request  "
            "method=%s path=%s client=%s  "
            "headers present: %s",
            method, path, peer,
            [k for k in request.headers.keys()],
        )
        raise HTTPException(status_code=401, detail="API token required")

    logger.debug("Token found — type=%s client=%s", "SAML" if is_saml_token(token) else "API-key", peer)

    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
            result = validator.validate_token_and_get_role(token)
            if result.is_valid:
                logger.info("Auth OK (SAML) — user=%s role=%s client=%s", result.email, result.role, peer)
                request.state.user_role = result.role
                request.state.user_email = result.email
                return
            logger.warning("Auth rejected (SAML) — %s  client=%s", result.error_message, peer)
            raise HTTPException(
                status_code=401,
                detail=result.error_message or "Invalid SAML token",
            )
        except TokenValidationError as exc:
            logger.warning("Auth rejected (SAML TokenValidationError) — %s  client=%s", exc, peer)
            raise HTTPException(status_code=401, detail=str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("SAML validation service unavailable — %s  client=%s", exc, peer)
            raise HTTPException(
                status_code=500, detail="SAML token validation service unavailable"
            )
    else:
        user_info = _get_token_manager().validate_token(agent_name, token)
        if not user_info:
            logger.warning(
                "Auth rejected (API key) — token invalid or expired  agent=%s client=%s",
                agent_name, peer,
            )
            raise HTTPException(status_code=401, detail="Invalid or expired API token")
        logger.info(
            "Auth OK (API key) — user_id=%s role=%s agent=%s client=%s",
            user_info.get("user_id"), user_info.get("role_id"), agent_name, peer,
        )
        request.state.user_id = user_info.get("user_id")
        request.state.user_role = user_info.get("role_id")


async def verify_api_key(request: Request) -> bool:
    """FastAPI dependency: validate the API token on every request."""
    if "/status" in request.url.path:
        return True

    from oai_agent_registry.dependencies import registry_instance  # noqa: PLC0415
    _validate_token(request, agent_name="agent-registry")
    return True
