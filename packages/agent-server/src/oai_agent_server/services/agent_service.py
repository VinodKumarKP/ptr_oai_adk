import asyncio
import os
import time

from fastapi.responses import JSONResponse

from oai_agent_server.exceptions import AgentInitializationException


class AgentService:
    """Service for managing agent lifecycle and information."""

    def __init__(self, agent, server_state, logger):
        self.agent = agent
        self.server_state = server_state
        self.logger = logger
        self.agent_name = agent.agent_name

    async def initialize_agent(self):
        """Initialize the agent."""
        try:
            await self.agent.initialize()
            return {"message": "Agent initialized successfully", "initialized": self.agent._initialized}
        except Exception as e:
            raise AgentInitializationException(agent_name=self.agent_name, reason=str(e))

    async def restart_server(self):
        """Graceful restart - wait for current requests to finish before restarting"""
        try:
            self.logger.info(f"[RESTART] Graceful restart initiated for agent '{self.agent_name}'")

            # Set shutdown flag to stop accepting new requests
            self.server_state.is_shutting_down = True

            # Wait for active requests to complete (with timeout)
            start_time = time.time()
            while self.server_state.active_requests > 0 and (
                    time.time() - start_time) < self.server_state.shutdown_timeout:
                self.logger.info(
                    f"[RESTART] Waiting for {self.server_state.active_requests} active requests to complete...")
                await asyncio.sleep(1)

            if self.server_state.active_requests > 0:
                self.logger.info(
                    f"[RESTART] Timeout reached, forcing restart with {self.server_state.active_requests} active requests")
            else:
                self.logger.info("[RESTART] All requests completed, initiating restart")

            # Return success response before exit
            response_data = {
                "message": "Server restarting gracefully",
                "agent": self.agent_name,
                "remaining_requests": self.server_state.active_requests
            }

            # Schedule the actual restart after response is sent
            asyncio.create_task(self._delayed_restart())

            return JSONResponse(content=response_data)

        except Exception as e:
            self.logger.info(f"[RESTART] Error during restart: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Restart failed: {str(e)}"}
            )

    async def kill_switch(self):
        """Immediate kill switch - restart container immediately"""
        try:
            self.logger.info(f"[KILL] Immediate kill switch activated for agent '{self.agent_name}'")

            response_data = {
                "message": "Server killed immediately",
                "agent": self.agent_name
            }

            # Schedule immediate exit after response is sent
            asyncio.create_task(self._immediate_kill())

            return JSONResponse(content=response_data)

        except Exception as e:
            self.logger.info(f"[KILL] Error during kill: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Kill failed: {str(e)}"}
            )

    async def get_agent_info(self, auth_enabled, request_isolation):
        """Get information about the agent."""
        return {
            "agent_name": self.agent_name,
            "session_id": self.agent.session_id,
            "cloud_provider": self.agent.agent_config.get('cloud_provider', 'aws'),
            "description": self.agent.agent_config.get('description', 'No description available'),
            "initialized": self.agent._initialized,
            "agent_config": self.agent.agent_config,
            "auth_enabled": auth_enabled,
            "request_isolation": request_isolation,
            "endpoint": f"http://localhost:{self.agent.agent_config.get('port', 8081)}"
        }

    async def get_prompts(self):
        """Get agent prompts."""
        prompts = self.agent.agent_config.get("prompts", [])
        return JSONResponse(content={
            "total_prompts": len(prompts),
            "prompts": prompts
        })

    async def _immediate_kill(self):
        """Perform immediate kill after a small delay to allow response to be sent"""
        await asyncio.sleep(0.1)  # Small delay to ensure response is sent
        self.logger.info("[KILL] Force killing process for immediate Docker restart...")
        os._exit(1)

    async def _delayed_restart(self):
        """Perform restart after a small delay to allow response to be sent"""
        await asyncio.sleep(0.1)  # Small delay to ensure response is sent
        self.logger.info("[RESTART] Exiting process for Docker restart...")
        os._exit(0)  # Exit code 0 for normal restart
