from typing import Optional, List

from fastapi import APIRouter, Query, Security, Request
from fastapi.responses import JSONResponse
from starlette import status
from starlette.responses import Response

from oai_agent_server.security.dependencies import verify_jwt_token

# Maximum number of active tokens allowed per agent server.
# Per-user scoping will be added in a future release.
MAX_TOKENS_PER_SERVER = 10


def create_token_router(token_service, allowed_modes: Optional[List[str]] = None):
    """Create the token management router with configured endpoints."""
    # Use Security instead of Depends to ensure Swagger UI picks it up as a security scheme
    router = APIRouter(prefix="/token", tags=["token"], dependencies=[Security(verify_jwt_token)])

    if allowed_modes is None:
        allowed_modes = ["token"]

    if 'token' in allowed_modes:
        # ---------------------------------------------------------------
        # Generate endpoints
        # ---------------------------------------------------------------

        def _generate(request: Request, user_id, role_id, ttl_seconds):
            """Shared helper — enforces the per-server token limit."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            try:
                result = token_service.generate_token(
                    server_key, user_id, role_id, ttl_seconds,
                    max_tokens=MAX_TOKENS_PER_SERVER,
                )
            except Exception as exc:
                msg = str(exc)
                if "Token limit reached" in msg:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": msg, "max_tokens": MAX_TOKENS_PER_SERVER},
                    )
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": msg},
                )
            return JSONResponse(content=result)

        @router.post("/custom")
        def generate_token(request: Request,
                           user_id: Optional[str] = Query(None),
                           role_id: Optional[str] = Query(None),
                           ttl_seconds: Optional[int] = Query(3600)):
            """Generate a token with custom TTL (hard limit: MAX_TOKENS_PER_SERVER active tokens)."""
            return _generate(request, user_id, role_id, ttl_seconds)

        @router.post("/short-term")
        def generate_short_term_token(request: Request,
                                      user_id: Optional[str] = Query(None),
                                      role_id: Optional[str] = Query(None)):
            """Generate a short-term token (5 minutes)."""
            return _generate(request, user_id, role_id, 300)

        @router.post("/long-term")
        def generate_long_term_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a long-term token (30 days)."""
            return _generate(request, user_id, role_id, 2592000)

        @router.post("/permanent")
        def generate_permanent_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a permanent token (no expiration)."""
            return _generate(request, user_id, role_id, None)

        # ---------------------------------------------------------------
        # List endpoint
        # ---------------------------------------------------------------

        @router.get("/list")
        def list_tokens(request: Request,
                        include_expired: bool = Query(False, description="Include already-expired TTL tokens")):
            """List all active (and optionally expired) tokens for this agent.

            Response includes ``max_tokens`` and ``can_generate`` so the UI can
            enforce the quota without a separate request.
            """
            server_key = getattr(request.app.state, "agent_name", "unknown")
            tokens = token_service.get_all_tokens(server_key, include_expired=include_expired)
            active_count = sum(1 for t in tokens if not t.get("is_expired", False))
            return JSONResponse(content={
                "tokens": tokens,
                "total": len(tokens),
                "max_tokens": MAX_TOKENS_PER_SERVER,
                "can_generate": active_count < MAX_TOKENS_PER_SERVER,
            })

        # ---------------------------------------------------------------
        # Revoke all  (registered before revoke-single to avoid ambiguity)
        # ---------------------------------------------------------------

        @router.delete("/revoke-all")
        def revoke_all_tokens(request: Request):
            """Revoke every active token for this agent."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            count = token_service.revoke_all_tokens(server_key)
            return JSONResponse(content={"revoked": count, "message": f"Revoked {count} token(s)"})

        # ---------------------------------------------------------------
        # Revoke single
        # ---------------------------------------------------------------

        @router.delete("/revoke/{token_str}")
        def revoke_token(token_str: str, response: Response):
            """Revoke a specific token.  Returns 404 if not found."""
            revoked = token_service.revoke_token(token_str)
            if not revoked:
                response.status_code = status.HTTP_404_NOT_FOUND
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"detail": "Token not found or already revoked"},
                )
            return JSONResponse(content={"revoked": True, "message": "Token revoked successfully"})

    return router
