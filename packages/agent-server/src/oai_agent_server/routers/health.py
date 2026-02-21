import os
from typing import List, Optional

import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from oai_agent_server.middleware.request_context import get_original_environ, request_env, sanitize_for_logging


def create_health_router(agent_name, server_state, enable_request_isolation, allowed_modes: Optional[List[str]] = None):
    """Create the health router with configured endpoints."""
    # Health router does NOT have the verify_api_key dependency
    router = APIRouter(tags=["health"],)

    if allowed_modes is None:
        allowed_modes = ["chat", "agent", "logs", "health"]

    @router.get("/")
    async def root():
        """Root endpoint returning server information and available endpoints."""
        original_environ = get_original_environ()
        auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', '').lower() == 'true'

        endpoints = {}

        if "chat" in allowed_modes:
            endpoints.update({
                "POST /chat": "Send a message and get a response",
                "POST /chat/stream": "Send a message and get a streaming response"
            })

        if "health" in allowed_modes:
            endpoints.update({
                "GET /health": "Health check endpoint",
                "GET /status": "Get server status",
                "GET /debug/env": "Debug request environment (if enabled)"
            })

        if "agent" in allowed_modes:
            endpoints.update({
                "GET /agent/info": "Get agent information"
            })

        return {
            "message": f"Agent HTTP Server for {agent_name}",
            "endpoints": endpoints,
            "auth_enabled": auth_enabled,
            "features": {
                "request_isolation": enable_request_isolation,
                "concurrent_requests": "supported" if enable_request_isolation else "not isolated"
            }
        }

    if "health" in allowed_modes:
        @router.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            return {"status": "healthy", "agent": agent_name}

        @router.get("/status")
        async def server_status():
            """Get current server status including active requests"""
            return {
                "agent_name": agent_name,
                "active_requests": server_state.active_requests,
                "is_shutting_down": server_state.is_shutting_down,
                "uptime": time.time() - getattr(server_state, 'start_time', time.time()),
                "status": "shutting_down" if server_state.is_shutting_down else "running"
            }

        @router.get("/check-env")
        async def check_environment():
            """Return all HTTP_ environment variables"""
            http_env_vars = {k: v for k, v in os.environ.items() if k.startswith('HTTP_')}
            return {"environment_variables": http_env_vars}

        @router.get("/debug/env")
        async def debug_env():
            """
            Debug endpoint to show current request environment.
            Useful for testing request isolation.
            """
            req_env = request_env.get()

            # Sanitize sensitive values
            sanitized = sanitize_for_logging(req_env)

            return JSONResponse({
                "request_env_count": len(req_env),
                "request_env": sanitized,
                "isolation_enabled": enable_request_isolation
            })

    return router
