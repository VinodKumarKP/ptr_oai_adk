import os
from typing import Optional
from fastapi import Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse
from starlette.exceptions import HTTPException

from oai_mcp_server_core.core.context import RequestAwareEnviron, request_env

# Maximum number of active tokens allowed per MCP server.
# Per-user scoping will be added in a future release.
MAX_TOKENS_PER_SERVER = 10


def register_server_routes(config_root, mcp_app, server_name: str, server_config, enable_request_isolation: bool, token_manager):
    """Register system and token routes for the MCP server."""

    @mcp_app.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        """
        Health check endpoint that returns "OK" when the server is healthy.
        """
        return PlainTextResponse("OK")

    @mcp_app.custom_route("/info", methods=["GET"])
    async def server_info(request: Request) -> JSONResponse:
        """Get server information including configuration and status."""
        original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
        auth_enabled = original_environ.get('AUTH_ENABLED', '').lower() == 'true'

        info = {
            "server_name": server_name,
            "status": "running",
            "server_config": server_config.model_dump(),
            "auth_enabled": auth_enabled,
            "request_isolation": enable_request_isolation
        }
        return JSONResponse(info)

    @mcp_app.custom_route("/", methods=["GET"])
    async def root(request: Request):
        """Root endpoint with API documentation."""
        original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
        auth_enabled = original_environ.get('AUTH_ENABLED', '').lower() == 'true'

        info = {
            "message": f"MCP Server: {server_name}",
            "endpoints": {
                "POST /mcp": "MCP server tools via streamable http",
                "GET /health": "Health check endpoint",
                "GET /info": "Get MCP Server information",
                "GET /debug/env": "Debug request environment (if enabled)",
                "GET /docs": "Swagger UI documentation",
                "GET /readme: Get the MCP Server readme"
                "POST /token/custom": "Generate a token with custom TTL",
                "POST /token/short-term": "Generate a short-term token (5 min)",
                "POST /token/long-term": "Generate a long-term token (30 days)",
                "POST /token/permanent": "Generate a permanent (non-expiring) token",
                "GET /token/list": "List all active tokens",
                "DELETE /token/revoke-all": "Revoke all tokens",
                "DELETE /token/revoke/{token}": "Revoke a specific token",
            },
            "auth_enabled": auth_enabled,
            "features": {
                "request_isolation": enable_request_isolation,
                "concurrent_requests": "supported" if enable_request_isolation else "not isolated"
            }
        }
        return JSONResponse(info)

    @mcp_app.custom_route("/debug/env", methods=["GET"])
    async def debug_env(request: Request) -> JSONResponse:
        """
        Debug endpoint to show current request environment.
        Useful for testing request isolation.
        """
        req_env = request_env.get()

        # Sanitize sensitive values
        sanitized = {}
        sensitive_keys = {'AUTHORIZATION', 'API_KEY', 'API_TOKEN', 'TOKEN', 'SECRET', 'PASSWORD'}

        for key, value in req_env.items():
            key_upper = key.upper()
            if any(sensitive in key_upper for sensitive in sensitive_keys):
                if len(value) > 8:
                    sanitized[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = value

        return JSONResponse({
            "request_env_count": len(req_env),
            "request_env": sanitized,
            "isolation_enabled": enable_request_isolation
        })

    @mcp_app.custom_route("/readme", methods=["GET"])
    async def get_agent_readme(request: Request) -> JSONResponse:
        """
        Reads and returns the content of the agent's README.md file.
        """

        readme_path = os.path.join(config_root, "servers", server_name, "README.md")

        if not os.path.exists(readme_path):
            raise HTTPException(status_code=404, detail=f"No README found at {readme_path}")

        try:
            with open(readme_path, "r") as f:
                content = f.read()
            return JSONResponse({
                "server_name": server_name,
                "content": content
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading README.md: {e}")

    def _token_response(token: str, user_id: Optional[str], role_id: Optional[str], ttl_seconds: Optional[int]) -> dict:
        """Build the HTTP response dict for a freshly generated token.

        generate_token() now returns only the token string (oai_platform_core
        canonical API).  This helper reconstructs the full response shape that
        clients expect, mirroring the normalisations applied inside TokenManager.
        """
        normalised_user = (
            user_id.split("@")[0] if user_id and "@" in user_id else user_id
        ) or "anonymous"
        normalised_role = role_id or "default"
        return {
            "token": token,
            "user_id": normalised_user,
            "role_id": normalised_role,
            "ttl_seconds": ttl_seconds,
        }

    def _generate_token_response(user_id, role_id, ttl_seconds):
        """Shared generate helper — enforces per-server token limit."""
        try:
            token = token_manager.generate_token(
                server_name, user_id, role_id, ttl_seconds,
                max_tokens=MAX_TOKENS_PER_SERVER,
            )
        except ValueError as exc:
            msg = str(exc)
            return JSONResponse(
                status_code=429,
                content={"detail": msg, "max_tokens": MAX_TOKENS_PER_SERVER},
            )
        return JSONResponse(content=_token_response(token, user_id, role_id, ttl_seconds))

    @mcp_app.custom_route("/token/custom", methods=["POST"])
    def generate_token(request: Request,
                       user_id: Optional[str] = None,
                       role_id: Optional[str] = None,
                       ttl_seconds: Optional[int] = 3600):
        """Generate a token with custom TTL (hard limit: MAX_TOKENS_PER_SERVER active tokens)."""
        return _generate_token_response(user_id, role_id, ttl_seconds)

    @mcp_app.custom_route("/token/short-term", methods=["POST"])
    def generate_short_term_token(request: Request,
                                  user_id: Optional[str] = None,
                                  role_id: Optional[str] = None):
        """Generate a short-term token (5 minutes)."""
        return _generate_token_response(user_id, role_id, 300)

    @mcp_app.custom_route("/token/long-term", methods=["POST"])
    def generate_long_term_token(request: Request,
                                 user_id: Optional[str] = None,
                                 role_id: Optional[str] = None):
        """Generate a long-term token (30 days)."""
        return _generate_token_response(user_id, role_id, 2592000)

    @mcp_app.custom_route("/token/permanent", methods=["POST"])
    def generate_permanent_token(request: Request,
                                 user_id: Optional[str] = None,
                                 role_id: Optional[str] = None):
        """Generate a permanent token (no expiration)."""
        return _generate_token_response(user_id, role_id, None)

    # -----------------------------------------------------------------------
    # Token management — list, revoke-all, revoke-single
    # -----------------------------------------------------------------------

    @mcp_app.custom_route("/token/list", methods=["GET"])
    def list_tokens(request: Request,
                    include_expired: bool = Query(False, description="Include already-expired TTL tokens")):
        """List all active (and optionally expired) tokens for this MCP server.

        Response includes ``max_tokens`` and ``can_generate`` so the UI can
        enforce the quota without a separate request.
        """
        tokens = token_manager.get_all_tokens(server_name, include_expired=include_expired)
        active_count = sum(1 for t in tokens if not t.get("is_expired", False))
        return JSONResponse(content={
            "tokens": tokens,
            "total": len(tokens),
            "max_tokens": MAX_TOKENS_PER_SERVER,
            "can_generate": active_count < MAX_TOKENS_PER_SERVER,
        })

    @mcp_app.custom_route("/token/revoke-all", methods=["DELETE"])
    def revoke_all_tokens(request: Request):
        """Revoke every active token for this MCP server."""
        count = token_manager.revoke_all_tokens(server_name)
        return JSONResponse(content={"revoked": count, "message": f"Revoked {count} token(s)"})

    @mcp_app.custom_route("/token/revoke/{token_str}", methods=["DELETE"])
    def revoke_token(request: Request):
        """Revoke a specific token.  Returns 404 if not found or already revoked."""
        revoked = token_manager.revoke_token(request.path_params.get('token_str'))
        if not revoked:
            return JSONResponse(
                status_code=404,
                content={"detail": "Token not found or already revoked"},
            )
        return JSONResponse(content={"revoked": True, "message": "Token revoked successfully"})