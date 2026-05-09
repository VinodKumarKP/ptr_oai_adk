import os
from typing import Optional
from fastapi import Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from oai_mcp_server_core.core.context import RequestAwareEnviron, request_env

def register_server_routes(mcp_app, server_name: str, server_config, enable_request_isolation: bool, token_manager):
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
                "GET /docs": "Swagger UI documentation"
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

    @mcp_app.custom_route("/token/custom", methods=["POST"])
    def generate_token(request: Request,
                       user_id: Optional[str] = None,
                       role_id: Optional[str] = None,
                       ttl_seconds: Optional[int] = 3600):
        """Generate a token with embedded metadata."""
        server_key = getattr(request.app.state, "agent_name", "unknown")
        result = token_manager.generate_token(server_key, user_id, role_id, ttl_seconds)
        return JSONResponse(content=result)

    @mcp_app.custom_route("/token/short-term", methods=["POST"])
    def generate_short_term_token(request: Request,
                                  user_id: Optional[str] = None,
                                  role_id: Optional[str] = None):
        """Generate a short-term token (5 minutes)."""
        server_key = getattr(request.app.state, "agent_name", "unknown")
        result = token_manager.generate_token(server_key, user_id, role_id, ttl_seconds=300)
        return JSONResponse(content=result)

    @mcp_app.custom_route("/token/long-term", methods=["POST"])
    def generate_long_term_token(request: Request,
                                 user_id: Optional[str] = None,
                                 role_id: Optional[str] = None):
        """Generate a long-term token (30 days)."""
        server_key = getattr(request.app.state, "agent_name", "unknown")
        result = token_manager.generate_token(server_key, user_id, role_id, ttl_seconds=2592000)
        return JSONResponse(content=result)

    @mcp_app.custom_route("/token/permanent", methods=["POST"])
    def generate_permanent_token(request: Request,
                                 user_id: Optional[str] = None,
                                 role_id: Optional[str] = None):
        """Generate a permanent token (no expiration)."""
        server_key = getattr(request.app.state, "agent_name", "unknown")
        result = token_manager.generate_token(server_key, user_id, role_id, ttl_seconds=None)
        return JSONResponse(content=result)
