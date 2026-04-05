from typing import List, Optional

from fastapi import APIRouter

from oai_agent_server.middleware.request_context import get_original_environ


def create_agent_router(agent_service, enable_request_isolation, allowed_modes: Optional[List[str]] = None):
    """Create the agent router with configured endpoints."""
    router = APIRouter(tags=["agent"])

    if allowed_modes is None:
        allowed_modes = ["agent"]

    if "agent" in allowed_modes:
        @router.post("/agent/initialize")
        async def initialize_agent():
            """Initialize the agent."""
            return await agent_service.initialize_agent()

        @router.post("/restart")
        async def restart_server():
            """Restart the server."""
            return await agent_service.restart_server()

        @router.post("/kill")
        async def kill_switch():
            """Kill the server process."""
            return await agent_service.kill_switch()

        @router.get("/agent/info")
        async def agent_info():
            """Get agent information."""
            original_environ = get_original_environ()
            auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', '').lower() == 'true'
            return await agent_service.get_agent_info(auth_enabled, enable_request_isolation)

        @router.get("/prompts")
        async def get_prompts():
            """Get agent prompts."""
            return await agent_service.get_prompts()

    return router