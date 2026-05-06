import json
import logging
import os
import httpx
from typing import Dict, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastmcp import FastMCP

from oai_mcp_registry.models import AppConfig, ServerConfig, RegistryConfig, ServerRegistration, ServerDeregistration
from oai_mcp_registry.utils.util import get_local_ip, get_public_ip
from oai_mcp_registry.services.db.database_logger import RegistryDatabaseLogger

logger = logging.getLogger("MCPRegistry")

class MCPRegistry:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv("MCP_CONFIG_PATH", './config/proxy_config.json')
        self.host_ip = get_local_ip()
        self.sub_apps: Dict[str, FastAPI] = {}
        self.config: Optional[AppConfig] = None
        self.registry_config: RegistryConfig = RegistryConfig()
        self.public_ip = get_public_ip()
        self.db_logger: RegistryDatabaseLogger = RegistryDatabaseLogger(logger=logger)
        self.exit_stack = None

        try:
            self.load_configuration()
        except (FileNotFoundError, Exception) as e:
            logger.critical(f"Failed to load configuration: {e}")
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
                
            # Add registered_via attribute for config servers
            for name, server in self.config.servers.items():
                setattr(server, 'registered_via', 'config')
                
            logger.info(f"Successfully loaded configuration from {self.config_path}")
        except (json.JSONDecodeError, Exception) as e:
            raise Exception(f"Configuration Invalid: {e}") from e

    async def initialize(self):
        """Initializes the MCPRegistry, including the database logger and HTTP client."""
        await self.db_logger.initialize()

        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            await self._sync_servers_to_db()
            await self._load_dynamic_servers_from_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")

        self.initialize_proxies()

    async def shutdown(self):
        """Shuts down the MCPRegistry, including closing the database logger."""
        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    async def _sync_servers_to_db(self):
        """Synchronizes config-declared servers to the database, marking them as registered_via='config'."""
        if not self.db_logger.is_active:
            logger.warning("Database logger is not active, skipping server sync to DB.")
            return

        for server_name, server_config in self.config.servers.items():
            if getattr(server_config, 'registered_via', 'dynamic') != 'config':
                continue
                
            try:
                await self.db_logger.log_server_registration(
                    server_name=server_name,
                    endpoint_url=server_config.endpoint or "",
                    port=server_config.port,
                    description=server_config.description,
                    active=True,  # Assuming config servers are meant to be active initially
                    registered_via="config",
                    source=server_config.source,
                    tags=server_config.tags,
                    current_version=server_config.current_version,
                    available_versions=server_config.available_versions,
                    deployment_mode=server_config.deployment_mode
                )
                logger.debug(f"Synced config server '{server_name}' to DB.")
            except Exception as e:
                logger.error(f"Failed to sync server '{server_name}' to DB from config: {e}")

    async def _load_dynamic_servers_from_db(self):
        """Restores dynamic servers from the DB that were active before the registry restarted."""
        logger.info("Restoring active dynamic MCP servers from database...")
        try:
            dynamic_servers = await self.db_logger.get_active_dynamic_servers()
        except Exception as e:
            logger.error(f"Failed to load dynamic servers from DB: {e}")
            return

        restored = 0
        skipped = 0
        for row in dynamic_servers:
            server_name = row.get("server_name")
            if not server_name:
                continue

            if server_name in self.config.servers:
                # Config server takes precedence — don't overwrite it with a DB snapshot.
                logger.debug(f"Skipping DB restore for '{server_name}' — already declared in config.")
                skipped += 1
                continue

            try:
                server_config = ServerConfig(
                    endpoint=row.get("endpoint_url", ""),
                    port=row.get("port"),
                    description=row.get("description", "A proxied MCP server"),
                    source=row.get("source"),
                    tags=row.get("tags", []),
                    current_version=row.get("current_version"),
                    available_versions=row.get("available_versions", []),
                    deployment_mode=row.get("deployment_mode", "docker")
                )
                setattr(server_config, 'registered_via', 'dynamic')
                self.config.servers[server_name] = server_config
                restored += 1
                logger.debug(f"Restored dynamic server '{server_name}' from DB.")
            except Exception as e:
                logger.error(f"Failed to restore dynamic server '{server_name}' from DB: {e}")

        logger.info(f"Dynamic server restore complete: {restored} restored, {skipped} skipped (config collision).")

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
                            server_config = ServerConfig(
                                endpoint=f"http://{host}:{port}",
                                description=server_info.get("server_config", {}).get("description", "Auto-discovered MCP server"),
                                source=server_info.get("source"),
                                tags=server_info.get("tags", []),
                                current_version=server_info.get("current_version"),
                                available_versions=server_info.get("available_versions", []),
                                deployment_mode=server_info.get("deployment_mode", "docker")
                            )
                            setattr(server_config, 'registered_via', 'dynamic')
                            self.config.servers[server_name] = server_config
                            logger.info(f"Discovered MCP server '{server_name}' at http://{host}:{port}")
                            
                            # Log discovery to DB
                            await self.db_logger.log_server_registration(
                                server_name=server_name,
                                endpoint_url=f"http://{host}:{port}",
                                port=port,
                                description=server_config.description,
                                active=True,
                                registered_via="dynamic",
                                source=server_config.source,
                                tags=server_config.tags,
                                current_version=server_config.current_version,
                                available_versions=server_config.available_versions,
                                deployment_mode=server_config.deployment_mode
                            )
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

    async def reload_config(self) -> JSONResponse:
        try:
            self.load_configuration()
            await self._sync_servers_to_db()
            self.initialize_proxies()
            return JSONResponse({"message": "Configuration reloaded", "servers": list(self.config.servers.keys())})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def register_server(self, app: FastAPI, server_registration: ServerRegistration) -> JSONResponse:
        """Registers a new MCP server dynamically via the /register endpoint."""
        server_name = server_registration.name
        if server_name in self.config.servers:
            logger.info(f"Server '{server_name}' is already registered. Updating its configuration.")

        # Update in-memory configuration
        server_config = ServerConfig(**server_registration.model_dump())
        self.config.servers[server_name] = server_config
        
        registered_via = server_registration.registered_via or "dynamic"

        # Initialize the proxy for the new server
        try:
            url = self._build_upstream_url(server_config)
            if not url:
                raise ValueError("No endpoint or port defined.")

            logger.info(f"  [Registering dynamically] {server_name} -> {url}")
            mcp = FastMCP.as_proxy(url, name=server_name)
            sub_app = mcp.http_app()
            
            # Mount and trigger lifespan manually for dynamic additions
            app.mount(f"/{server_name}", sub_app)
            self.sub_apps[server_name] = sub_app
            
            # If the application is already running, enter the lifespan context
            if self.exit_stack:
                await self.exit_stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
                
        except Exception as e:
            logger.error(f"Failed to initialize proxy for '{server_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize proxy: {str(e)}")

        # Log to DB
        await self.db_logger.log_server_registration(
            server_name=server_name,
            endpoint_url=server_config.endpoint or "",
            port=server_config.port,
            description=server_config.description,
            active=True,
            registered_via=registered_via,
            source=server_config.source,
            tags=server_config.tags,
            current_version=server_config.current_version,
            available_versions=server_config.available_versions,
            deployment_mode=server_config.deployment_mode
        )

        return JSONResponse({"message": f"Server '{server_name}' registered successfully."})

    async def deregister_server(self, server_deregistration: ServerDeregistration) -> JSONResponse:
        """Deregisters an MCP server."""
        server_name = server_deregistration.name
        
        if server_name not in self.config.servers:
            logger.warning(f"Attempted to deregister server '{server_name}', but it was not found.")
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        # We can't easily unmount from FastAPI dynamically, but we remove it from sub_apps 
        # so the middleware drops any HTTP traffic and it stops appearing in /info
        if server_name in self.sub_apps:
            del self.sub_apps[server_name]
            
        logger.info(f"Deactivating server '{server_name}'.")

        await self.db_logger.deregister_server(server_name=server_name)

        return JSONResponse({"message": f"Server '{server_name}' deactivated successfully."})
