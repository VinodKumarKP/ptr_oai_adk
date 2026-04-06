import json
import logging
import os
import httpx
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP

from oai_mcp_registry.models import AppConfig, ServerConfig, RegistryConfig
from oai_mcp_registry.utils.util import get_local_ip, get_public_ip

logger = logging.getLogger("MCPRegistry")

class MCPRegistry:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv("MCP_CONFIG_PATH", './config/proxy_config.json')
        self.host_ip = get_local_ip()
        self.sub_apps: Dict[str, FastAPI] = {}
        self.config: Optional[AppConfig] = None
        self.registry_config: RegistryConfig = RegistryConfig()
        self.public_ip = get_public_ip()

        try:
            self.load_configuration()
            self.initialize_proxies()
        except (FileNotFoundError, Exception) as e:
            logger.critical(f"Failed to initialize server due to config error: {e}")
            self.config = AppConfig(servers={}, registry=self.registry_config)

    def load_configuration(self):
        if not os.path.exists(self.config_path):
            self.config = AppConfig(servers={}, registry=self.registry_config)
            return

        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)
                self.config = AppConfig(**config_data)
                self.registry_config = self.config.registry
            logger.info(f"Successfully loaded configuration from {self.config_path}")
        except (json.JSONDecodeError, Exception) as e:
            raise Exception(f"Configuration Invalid: {e}") from e

    async def discover_servers(self, host: str = None):
        start = self.registry_config.start_port
        end = self.registry_config.end_port
        host = host or self.registry_config.host or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        logger.info(f"Starting auto-discovery of MCP servers in port range {start}-{end} on host {host}...")

        for port in range(start, end + 1):
            endpoint =  f"http://localhost:{port}/info"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(endpoint, timeout=1.0)
                    if response.status_code == 200:
                        server_info = response.json()
                        server_name = server_info.get("server_name")
                        if server_name and server_name not in self.config.servers:
                            self.config.servers[server_name] = ServerConfig(
                                endpoint=f"http://{host}:{port}",
                                description=server_info.get("server_config", {}).get("description", "Auto-discovered MCP server")
                            )
                            logger.info(f"Discovered MCP server '{server_name}' at http://{host}:{port}")
            except (httpx.RequestError, json.JSONDecodeError) as e:
                pass
        self.initialize_proxies()

    def _build_upstream_url(self, server_info: ServerConfig) -> str:
        url = ""
        if server_info.endpoint:
            url = server_info.endpoint
        elif server_info.port:
            url = f"http://{self.host_ip}:{server_info.port}/mcp"

        if url and not (url.endswith("/sse") or url.endswith("/mcp")):
            url = f"{url.rstrip('/')}/mcp"
        return url

    def initialize_proxies(self):
        if not self.config:
            return

        logger.info(f"Initializing {len(self.config.servers)} upstream servers...")
        self.sub_apps = {}

        for name, info in self.config.servers.items():
            try:
                url = self._build_upstream_url(info)
                if not url:
                    logger.warning(f"Skipping '{name}': No endpoint or port defined.")
                    continue

                logger.info(f"  [Registering] {name} -> {url}")
                mcp = FastMCP.as_proxy(url, name=name)
                self.sub_apps[name] = mcp.http_app()
            except Exception as e:
                logger.error(f"  [Failed] Could not initialize '{name}': {e}")

    def reload_config(self):
        try:
            self.load_configuration()
            self.initialize_proxies()
            return {"message": "Configuration reloaded", "servers": list(self.config.servers.keys())}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
