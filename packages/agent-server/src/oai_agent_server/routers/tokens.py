from typing import Optional, List

from fastapi import APIRouter, Query, Security, Request
from fastapi.responses import JSONResponse
from starlette import status
from starlette.responses import Response

from oai_agent_server.security.dependencies import verify_jwt_token


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

        @router.post("/custom")
        def generate_token(request: Request,
                           user_id: Optional[str] = Query(None),
                           role_id: Optional[str] = Query(None),
                           ttl_seconds: Optional[int] = Query(3600)):
            """Generate a token with embedded metadata."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds)
            return JSONResponse(content=result)

        @router.post("/short-term")
        def generate_short_term_token(request: Request,
                                      user_id: Optional[str] = Query(None),
                                      role_id: Optional[str] = Query(None)):
            """Generate a short-term token (5 minutes)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=300)
            return JSONResponse(content=result)

        @router.post("/long-term")
        def generate_long_term_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a long-term token (30 days)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=2592000)
            return JSONResponse(content=result)

        @router.post("/permanent")
        def generate_permanent_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a permanent token (no expiration)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=None)
            return JSONResponse(content=result)

        # ---------------------------------------------------------------
        # List endpoint
        # ---------------------------------------------------------------

        @router.get("/list")
        def list_tokens(request: Request,
                        include_expired: bool = Query(False, description="Include already-expired TTL tokens")):
            """List all active (and optionally expired) tokens for this agent."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            tokens = token_service.get_all_tokens(server_key, include_expired=include_expired)
            return JSONResponse(content={"tokens": tokens, "total": len(tokens)})

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
