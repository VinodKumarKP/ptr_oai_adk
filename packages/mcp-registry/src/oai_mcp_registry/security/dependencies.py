"""
MCP-registry security dependencies.

Token validation for FastAPI endpoints.  The token extraction logic
(header parsing) is handled by the framework-agnostic
:func:`oai_platform_core.security.extract_bearer_token` helper; the
FastAPI-specific parts (HTTPException, request.state mutation) stay here.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

from oai_platform_core.security import TokenManager, is_saml_token, extract_bearer_token
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# Shared TokenManager instance — one Redis connection pool for the process lifetime.
_token_manager = TokenManager(db_path_name="mcp_registry_tokens.db")


def _validate_token(request: Request, server_name: str) -> None:
    """Core token-validation logic for MCP-registry endpoints.

    Args:
        request:     The FastAPI request being authenticated.
        server_name: Name used to look up the token in the token store.

    Raises:
        HTTPException 401: token missing or invalid.
        HTTPException 500: auth service unavailable.
    """
    if "/status" in request.url.path or "/health" in request.url.path:
        return

    auth_enabled = os.environ.get("MCP_AUTH_ENABLED", "true").lower() == "true"
    if not auth_enabled:
        return

    _LOCAL_PEERS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "host.docker.internal"}
    peer = request.client.host if request.client else ""
    if peer in _LOCAL_PEERS and os.environ.get("FORCE_AUTH", "false").lower() == "false":
        return

    token = extract_bearer_token(request.headers)
    if not token:
        raise HTTPException(status_code=401, detail="API token required")

    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
            result = validator.validate_token_and_get_role(token)
            if result.is_valid:
                request.state.user_role = result.role
                request.state.user_email = result.email
                return
            raise HTTPException(
                status_code=401,
                detail=result.error_message or "Invalid SAML token",
            )
        except TokenValidationError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=500, detail="SAML token validation service unavailable"
            )
    else:
        user_info = _token_manager.validate_token(server_name, token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid or expired API token")
        request.state.user_id = user_info.get("user_id")
        request.state.user_role = user_info.get("role_id")


async def verify_api_key(request: Request) -> bool:
    """FastAPI dependency: validate the API token on every request."""
    if "/status" in request.url.path or "/health" in request.url.path:
        return True

    from oai_mcp_registry.dependencies import registry_instance  # noqa: PLC0415
    _validate_token(request, server_name="mcp-registry")
    return True
