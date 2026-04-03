from typing import Optional, List

from fastapi import APIRouter, Query, Security, Request
from fastapi.responses import JSONResponse

from oai_agent_server.security.dependencies import verify_jwt_token


def create_token_router(token_service, allowed_modes: Optional[List[str]] = None):
    """Create the logs router with configured endpoints."""
    # Use Security instead of Depends to ensure Swagger UI picks it up as a security scheme
    router = APIRouter(prefix="/token", tags=["token"], dependencies=[Security(verify_jwt_token)])

    if allowed_modes is None:
        allowed_modes = ["token"]

    if 'token' in allowed_modes:
        @router.post("/custom")
        def generate_token(request: Request,
                           user_id: Optional[str] = Query(None),
                           role_id: Optional[str] = Query(None),
                           ttl_seconds: Optional[int] = Query(3600)):
            """Generate a token with embedded metadata."""
            # Retrieve agent_name from app state (set in main.py)
            server_key = getattr(request.app.state, "agent_name", "unknown")

            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds)
            return JSONResponse(content=result)

        @router.post("/short-term")
        def generate_short_term_token(request: Request,
                                      user_id: Optional[str] = Query(None),
                                      role_id: Optional[str] = Query(None)):
            """Generate a short-term token (5 minutes)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            # Fixed TTL of 300 seconds (5 minutes)
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=300)
            return JSONResponse(content=result)

        @router.post("/long-term")
        def generate_long_term_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a long-term token (30 days)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            # Fixed TTL of 30 days (2592000 seconds)
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=2592000)
            return JSONResponse(content=result)

        @router.post("/permanent")
        def generate_permanent_token(request: Request,
                                     user_id: Optional[str] = Query(None),
                                     role_id: Optional[str] = Query(None)):
            """Generate a permanent token (no expiration)."""
            server_key = getattr(request.app.state, "agent_name", "unknown")
            # TTL None means permanent
            result = token_service.generate_token(server_key, user_id, role_id, ttl_seconds=None)
            return JSONResponse(content=result)

    return router
