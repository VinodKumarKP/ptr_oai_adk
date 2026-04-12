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

from oai_agent_registry.models import Config, AgentConfig, RegistryConfig
from oai_agent_registry.security.dependencies import _validate_token

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
                    self.agents[agent_name] = AgentConfig(endpoint=agent_data, name=agent_name)
                elif isinstance(agent_data, dict):
                    self.agents[agent_name] = AgentConfig(**agent_data)

            logger.info(f"Loaded configuration with {len(self.agents)} agents.")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

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
                    response = await client.get(f"{endpoint}/agent/info", timeout=1.0)
                    if response.status_code == 200:
                        agent_info = response.json()
                        agent_name = agent_info.get("agent_name")
                        if agent_name and agent_name not in self.agents:
                            self.agents[agent_name] = AgentConfig(
                                name=agent_name,
                                endpoint=endpoint,
                                description=agent_info.get("description", "Auto-discovered agent")
                            )
                            logger.info(f"Discovered agent '{agent_name}' at {endpoint}")
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
            return JSONResponse({"message": "Configuration reloaded", "agents": list(self.agents.keys())})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
