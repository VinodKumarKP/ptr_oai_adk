import json
import logging
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

from oai_agent_registry.models import Config, AgentConfig, RegistryConfig, AgentRegistration, AgentDeregistration
from oai_agent_registry.security.dependencies import _validate_token
from oai_agent_registry.services.registry_database_logger import RegistryDatabaseLogger

logger = logging.getLogger(__name__)

class AgentRegistry:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('REGISTRY_CONFIG_PATH', './config/registry_config.json')
        self.config: Optional[Config] = None
        self.agents: Dict[str, AgentConfig] = {}
        self.registry_config: RegistryConfig = RegistryConfig()
        self.start_time = time.time()
        self.client: Optional[httpx.AsyncClient] = None
        self.public_ip: Optional[str] = None
        self.private_ip: Optional[str] = None
        self.db_logger: RegistryDatabaseLogger = RegistryDatabaseLogger(logger=logger)
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found at {self.config_path}, using defaults.")
                self.config = Config(agents={})
            else:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self.config = Config(**config_data)

            self.registry_config = self.config.registry
            self.agents = {}
            for agent_name, agent_data in self.config.agents.items():
                if isinstance(agent_data, str):
                    # Initialize agent as disabled until status check confirms it's active
                    self.agents[agent_name] = AgentConfig(endpoint=agent_data, name=agent_name, enabled=False)
                elif isinstance(agent_data, dict):
                    # Ensure 'enabled' is explicitly set to False if not provided, or override if provided
                    agent_data_copy = agent_data.copy()
                    if 'enabled' not in agent_data_copy:
                        agent_data_copy['enabled'] = False
                    self.agents[agent_name] = AgentConfig(**agent_data_copy)

            logger.info(f"Loaded configuration with {len(self.agents)} agents.")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    async def initialize(self):
        """Initializes the AgentRegistry, including the database logger and HTTP client."""
        self.client = httpx.AsyncClient() # Initialize httpx client here
        await self.db_logger.initialize()
        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            # Sync agents loaded from config to the database
            await self._sync_agents_to_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")
        
        await self._check_agent_statuses() # Call the new method here

    async def shutdown(self):
        """Shuts down the AgentRegistry, including closing the database logger and HTTP client."""
        if self.client:
            await self.client.aclose() # Close the httpx client
        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    async def _sync_agents_to_db(self):
        """Synchronizes agents loaded from config to the database."""
        if not self.db_logger.is_active:
            logger.warning("Database logger is not active, skipping agent sync to DB.")
            return

        for agent_name, agent_config in self.agents.items():
            try:
                await self.db_logger.log_agent_registration(
                    agent_name=agent_name,
                    endpoint_url=agent_config.endpoint,
                    port=agent_config.port,
                    git_source_url=agent_config.git_source_url,
                    active=agent_config.enabled
                )
                logger.debug(f"Synced agent '{agent_name}' from config to DB.")
            except Exception as e:
                logger.error(f"Failed to sync agent '{agent_name}' to DB from config: {e}")

    async def _check_agent_statuses(self):
        """Checks the /status endpoint of each agent and updates its enabled status."""
        logger.info("Checking agent statuses...")
        if not self.client:
            logger.error("HTTP client not initialized. Cannot check agent statuses.")
            return

        for agent_name, agent_config in self.agents.items():
            status_url = f"{agent_config.endpoint}/status"
            try:
                response = await self.client.get(status_url, timeout=agent_config.timeout)
                if response.status_code == 200:
                    agent_config.enabled = True
                    logger.info(f"Agent '{agent_name}' at {agent_config.endpoint} is active.")
                else:
                    agent_config.enabled = False
                    logger.warning(f"Agent '{agent_name}' at {agent_config.endpoint} returned status {response.status_code}. Marking as inactive.")
            except httpx.RequestError as e:
                agent_config.enabled = False
                logger.warning(f"Could not reach agent '{agent_name}' at {agent_config.endpoint} ({e}). Marking as inactive.")
            except Exception as e:
                agent_config.enabled = False
                logger.error(f"Error checking status for agent '{agent_name}' at {agent_config.endpoint}: {e}. Marking as inactive.")

    async def discover_agents(self):
        """Discover agents by scanning a range of ports."""
        start = self.registry_config.start_port
        end = self.registry_config.end_port
        host = self.registry_config.host or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        logger.info(f"Starting auto-discovery of agents in port range {start}-{end} on host {host}...")

        for port in range(start, end + 1):
            endpoint = f"http://{host}:{port}"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{endpoint}/info", timeout=1.0)
                    if response.status_code == 200:
                        agent_info = response.json()
                        agent_name = agent_info.get("agent_name")
                        if agent_name and agent_name not in self.agents:
                            agent_config = AgentConfig(
                                name=agent_name,
                                endpoint=endpoint,
                                description=agent_info.get("description", "Auto-discovered agent")
                            )
                            self.agents[agent_name] = agent_config
                            logger.info(f"Discovered agent '{agent_name}' at {endpoint}")
                            # Log the discovered agent
                            await self.db_logger.log_agent_registration(
                                agent_name=agent_name,
                                endpoint_url=endpoint,
                                port=port,
                                git_source_url=agent_info.get("git_source_url"), # Assuming agent_info might contain this
                                active=True
                            )
            except (httpx.RequestError, json.JSONDecodeError) as e:
                pass

    async def get_info(self) -> JSONResponse:
        """Returns information about the registry and its agents."""
        enabled_agents = {name for name, agent in self.agents.items() if agent.enabled}
        endpoint = self.private_ip if os.environ.get('USE_PRIVATE_IP', 'false').lower() == 'true' else \
            os.environ.get('AGENT_BASE_URL', "localhost")

        info = {
            "message": "Agent Registry",
            "uptime_seconds": int(time.time() - self.start_time),
            "total_agents": len(self.agents),
            "enabled_agents": len(enabled_agents),
            "agents": {
                name: {
                    "description": agent.description,
                    "endpoint": f"{endpoint}:{self.registry_config.port}/{name}",
                    "status": "active" if agent.enabled else "inactive"
                }
                for name, agent in self.agents.items()
            }
        }
        return JSONResponse(info)

    async def health_check(self) -> JSONResponse:
        """Performs a health check on all enabled agents."""
        health_status = {}
        all_healthy = True
        for agent_name, agent_config in self.agents.items():
            if not agent_config.enabled:
                health_status[agent_name] = {"status": "disabled"}
                continue
            try:
                response = await self.client.get(f"{agent_config.endpoint}/health")
                if response.status_code == 200:
                    health_status[agent_name] = {"status": "healthy", "response": response.json()}
                else:
                    health_status[agent_name] = {"status": "unhealthy", "code": response.status_code}
                    all_healthy = False
            except Exception as e:
                health_status[agent_name] = {"status": "unreachable", "error": str(e)}
                all_healthy = False
        
        return JSONResponse({
            "registry_status": "healthy",
            "all_agents_healthy": all_healthy,
            "agents": health_status,
            "timestamp": datetime.now().isoformat()
        })

    async def reload_config(self) -> JSONResponse:
        """Reloads the configuration from the config file."""
        try:
            self.load_config()
            # After reloading config, re-sync to DB
            await self._sync_agents_to_db()
            await self._check_agent_statuses() # Re-check statuses after reload
            return JSONResponse({"message": "Configuration reloaded", "agents": list(self.agents.keys())})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def register_agent(self, agent_registration: AgentRegistration) -> JSONResponse:
        """Registers a new agent."""
        agent_name = agent_registration.name
        if agent_name in self.agents:
            logger.info(f"Agent '{agent_name}' is already registered. Updating its configuration.")
        
        agent_config = AgentConfig(**agent_registration.dict())
        self.agents[agent_name] = agent_config
        
        logger.info(f"Registered agent '{agent_name}' with endpoint {agent_config.endpoint}")
        
        # Log the agent registration
        await self.db_logger.log_agent_registration(
            agent_name=agent_name,
            endpoint_url=agent_registration.endpoint,
            port=agent_registration.port,
            git_source_url=agent_registration.git_source_url,
            active=True
        )
        
        return JSONResponse({"message": f"Agent '{agent_name}' registered successfully."})

    async def deregister_agent(self, agent_deregistration: AgentDeregistration) -> JSONResponse:
        """Deregisters an agent by setting its active flag to False."""
        agent_name = agent_deregistration.name
        
        if agent_name in self.agents:
            # Update the in-memory agent config
            self.agents[agent_name].enabled = False
            logger.info(f"Deactivating agent '{agent_name}'.")
            
            # Update the active flag in the database
            await self.db_logger.deregister_agent(agent_name=agent_name)

            return JSONResponse({"message": f"Agent '{agent_name}' deactivated successfully."})
        else:
            logger.warning(f"Attempted to deregister agent '{agent_name}', but it was not found.")
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    async def proxy_request(self, agent_name: str, path: str, request: Request) -> Response:
        """Proxies a request to the specified agent."""
        _validate_token(request, self.registry_config, agent_name)

        if agent_name not in self.agents:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
        
        agent_config = self.agents[agent_name]
        if not agent_config.enabled:
            raise HTTPException(status_code=503, detail=f"Agent '{agent_name}' is disabled.")

        target_url = f"{agent_config.endpoint}/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        headers = dict(request.headers)
        headers.pop("host", None)

        body = await request.body()

        if "stream" in path:
            return await self._proxy_streaming_request(target_url, request.method, headers, body, agent_config.timeout)
        
        return await self._proxy_regular_request(agent_name, target_url, request.method, headers, body, agent_config.timeout, path)

    async def _proxy_regular_request(self, agent_name: str, url: str, method: str, headers: dict, body: bytes, timeout: int, path: str) -> Response:
        try:
            response = await self.client.request(method, url, headers=headers, content=body, timeout=timeout)
            
            response_headers = self._clean_response_headers(response.headers)
            content = response.content

            if path and path.rstrip("/") in ["docs", "redoc"] and "text/html" in response.headers.get("content-type", ""):
                try:
                    text = response.text
                    text = text.replace('"/openapi.json"', f'"/{agent_name}/openapi.json"')
                    text = text.replace("'/openapi.json'", f"'/{agent_name}/openapi.json'")
                    content = text.encode("utf-8")
                    if "content-length" in response_headers:
                        del response_headers["content-length"]
                except Exception as e:
                    logger.warning(f"Failed to rewrite docs for {agent_name}: {e}")

            elif path and path.rstrip("/") == "openapi.json" and "application/json" in response.headers.get("content-type", ""):
                try:
                    data = response.json()
                    if "paths" in data:
                        new_paths = {}
                        for p, methods in data["paths"].items():
                            new_path = f"/{agent_name}{p}" if not p.startswith(f"/{agent_name}") else p
                            new_paths[new_path] = methods
                        data["paths"] = new_paths
                    content = json.dumps(data).encode("utf-8")
                    if "content-length" in response_headers:
                        del response_headers["content-length"]
                except Exception as e:
                    logger.warning(f"Failed to rewrite openapi.json for {agent_name}: {e}")

            return Response(content=content, status_code=response.status_code, headers=response_headers, media_type=response.headers.get("content-type"))
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Request to agent '{agent_name}' timed out.")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Proxy error: {e}")

    async def _proxy_streaming_request(self, url: str, method: str, headers: dict, body: bytes, timeout: int) -> StreamingResponse:
        async def stream_generator():
            try:
                async with httpx.AsyncClient(timeout=timeout) as stream_client:
                    async with stream_client.stream(method, url, headers=headers, content=body) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    def _clean_response_headers(self, headers: httpx.Headers) -> dict:
        excluded = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
        return {k: v for k, v in headers.items() if k.lower() not in excluded}
