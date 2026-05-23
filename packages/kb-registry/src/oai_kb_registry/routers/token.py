"""
Token management routes — generate, list, revoke, and revoke-all.

All endpoints require a valid bearer token (the caller must already hold a
token to manage tokens).  Tokens are scoped to the ``"kb-registry"`` server key.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from oai_kb_registry.dependencies import verify_bearer_token
from oai_kb_registry.security.dependencies import _get_token_manager

router = APIRouter()
logger = logging.getLogger(__name__)

_SERVER_KEY = "kb-registry"
MAX_TOKENS = 10


@router.post("/tokens/generate", response_model=dict)
async def generate_token(
    user_id: Optional[str] = Query(None, description="User identifier"),
    role_id: Optional[str] = Query(None, description="Role identifier, defaults to 'default'"),
    ttl_seconds: Optional[int] = Query(3600, description="Lifetime in seconds; -1 for no expiration"),
    _auth: bool = Depends(verify_bearer_token),
):
    """Generate a new API token for the KB Registry."""
    try:
        tm = _get_token_manager()
        effective_ttl = None if ttl_seconds == -1 else ttl_seconds
        token = tm.generate_token(
            server_key=_SERVER_KEY,
            user_id=user_id,
            role_id=role_id,
            ttl_seconds=effective_ttl,
            max_tokens=MAX_TOKENS,
        )
        return {
            "token": token,
            "user_id": user_id or "anonymous",
            "role_id": role_id or "default",
            "ttl_seconds": ttl_seconds,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to generate token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token generation failed: {exc}",
        )


@router.get("/tokens", response_model=dict)
async def list_tokens(
    include_expired: bool = Query(False),
    _auth: bool = Depends(verify_bearer_token),
):
    """Return all active tokens for the KB Registry."""
    try:
        tm = _get_token_manager()
        tokens = tm.get_all_tokens(_SERVER_KEY, include_expired=include_expired)
        active_count = sum(1 for t in tokens if not t.get("is_expired", False))
        return {
            "tokens": tokens,
            "total": len(tokens),
            "max_tokens": MAX_TOKENS,
            "can_generate": active_count < MAX_TOKENS,
        }
    except Exception as exc:
        logger.error("Failed to list tokens: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tokens: {exc}",
        )


@router.delete("/tokens", response_model=dict)
async def revoke_all_tokens(_auth: bool = Depends(verify_bearer_token)):
    """Revoke every active token for the KB Registry."""
    try:
        tm = _get_token_manager()
        count = tm.revoke_all_tokens(_SERVER_KEY)
        return {"revoked": count, "message": f"Revoked {count} token(s)"}
    except Exception as exc:
        logger.error("Failed to revoke all tokens: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke tokens: {exc}",
        )


@router.delete("/tokens/{token}", response_model=dict)
async def revoke_token(token: str, _auth: bool = Depends(verify_bearer_token)):
    """Revoke a specific token."""
    try:
        tm = _get_token_manager()
        revoked = tm.revoke_token(token)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Token not found or already revoked",
            )
        return {"revoked": True, "message": "Token revoked successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to revoke token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {exc}",
        )
