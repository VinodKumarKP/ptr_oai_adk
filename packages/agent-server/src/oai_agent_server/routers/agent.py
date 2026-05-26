import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException


from oai_agent_server.middleware.request_context import get_original_environ
from oai_agent_server.security.dependencies import verify_api_key_strict
from starlette.responses import JSONResponse


def create_agent_router(config_root, agent_name, agent_service, enable_request_isolation, allowed_modes: Optional[List[str]] = None, ):
    """Create the agent router with configured endpoints."""
    router = APIRouter(tags=["agent"])

    if allowed_modes is None:
        allowed_modes = ["agent"]

    if "agent" in allowed_modes:
        @router.post("/agent/initialize")
        async def initialize_agent():
            """Initialize the agent."""
            return await agent_service.initialize_agent()

        @router.post("/restart", dependencies=[Depends(verify_api_key_strict)])
        async def restart_server():
            """Restart the server."""
            return await agent_service.restart_server()

        @router.post("/kill", dependencies=[Depends(verify_api_key_strict)])
        async def kill_switch():
            """Kill the server process."""
            return await agent_service.kill_switch()

        @router.get("/info")
        async def agent_info():
            """Get agent information."""
            original_environ = get_original_environ()
            auth_enabled = original_environ.get('AGENT_AUTH_ENABLED', '').lower() == 'true'
            return await agent_service.get_agent_info(auth_enabled, enable_request_isolation)

        @router.get("/prompts")
        async def get_prompts():
            """Get agent prompts."""
            return await agent_service.get_prompts()

        @router.get("/readme", response_class=JSONResponse, summary="Get Agent Readme")
        async def get_agent_readme():
            """
            Reads and returns the content of the agent's README.md file.
            """
            if allowed_modes and "readme" not in allowed_modes:
                raise HTTPException(status_code=403, detail="Readme endpoint is not enabled.")

            readme_path = os.path.join(config_root, "agents", agent_name, "README.md")

            if not os.path.exists(readme_path):
                raise HTTPException(status_code=404, detail=f"No README found at {readme_path}")

            try:
                with open(readme_path, "r") as f:
                    content = f.read()
                return JSONResponse({
                    "agent_name": agent_name,
                    "content": content
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading README.md: {e}")

    return router