"""
Agent-registry security dependencies.

Token validation for FastAPI endpoints.  The token extraction logic
(header parsing) is handled by the framework-agnostic
:func:`oai_platform_core.security.extract_bearer_token` helper; the
FastAPI-specific parts (HTTPException, request.state mutation) stay here.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from oai_platform_core.security import TokenManager, is_saml_token, extract_bearer_token
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError

logger = logging.getLogger(__name__)

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Shared TokenManager — one connection pool for the process lifetime.
_token_manager: Optional[TokenManager] = None

# ---------------------------------------------------------------------------
# Trusted-peer detection
# ---------------------------------------------------------------------------
# Exact hostnames / IPs that are always trusted (no token required).
_TRUSTED_PEER_NAMES: frozenset = frozenset({
    "127.0.0.1", "::1", "localhost", "0.0.0.0", "host.docker.internal",
})

# Docker containers calling the host via host.docker.internal arrive with the
# container's bridge IP as the *source*, not "host.docker.internal".  We trust
# the standard Docker bridge / overlay ranges by default.  Add more via the
# TRUSTED_SUBNETS env var (comma-separated CIDRs, e.g. "10.8.0.0/24").
_DEFAULT_TRUSTED_CIDRS = (
    "172.16.0.0/12",    # Docker default bridge: 172.17-31.x.x
    "192.168.65.0/24",  # Docker Desktop gateway on macOS / Windows
)


@functools.lru_cache(maxsize=1)
def _trusted_subnets() -> tuple:
    """Build and cache the list of trusted IP networks (evaluated once)."""
    cidrs = list(_DEFAULT_TRUSTED_CIDRS)
    extra = os.environ.get("TRUSTED_SUBNETS", "").strip()
    if extra:
        cidrs.extend(c.strip() for c in extra.split(",") if c.strip())
    nets = []
    for cidr in cidrs:
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid TRUSTED_SUBNETS entry: %r", cidr)
    logger.debug("Trusted subnets: %s", [str(n) for n in nets])
    return tuple(nets)


def _is_trusted_peer(peer: str) -> bool:
    """Return True if *peer* is a trusted loopback, hostname, or Docker-bridge address."""
    if peer in _TRUSTED_PEER_NAMES:
        return True
    try:
        addr = ipaddress.ip_address(peer)
        matched = next((n for n in _trusted_subnets() if addr in n), None)
        if matched:
            logger.debug("Peer %s matched trusted subnet %s", peer, matched)
            return True
    except ValueError:
        pass  # peer is a hostname we don't recognise — fall through to auth
    return False


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

    if _is_trusted_peer(peer) and not force_auth:
        logger.debug("Auth skipped — peer %s is a trusted local/Docker address", peer)
        return

    if not _is_trusted_peer(peer):
        logger.info(
            "Auth required — peer %s is not a trusted address  "
            "(trusted names: %s; trusted subnets: %s)  "
            "Set AGENT_AUTH_ENABLED=false to disable auth, or "
            "add extra CIDRs via TRUSTED_SUBNETS env var.",
            peer,
            sorted(_TRUSTED_PEER_NAMES),
            [str(n) for n in _trusted_subnets()],
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
