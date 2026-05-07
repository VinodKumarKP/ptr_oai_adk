import json
import logging
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastmcp import FastMCP

from oai_mcp_registry.models import AppConfig, ServerConfig, RegistryConfig, ServerRegistration, ServerDeregistration
from oai_mcp_registry.utils.util import get_local_ip, get_public_ip
from oai_mcp_registry.services.db.database_logger import RegistryDatabaseLogger
from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.services.deployers.factory import DeployerFactory

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
        self.deployers: Dict[str, BaseDeployer] = {}
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

    def _build_seed_configs_from_servers(self) -> dict:
        """Groups config servers by deployment mode for factory initialization."""
        seeds: dict = {}
        for server_name, server_config in self.config.servers.items():
            mode = getattr(server_config, "deployment_mode", "docker") or "docker"
            if mode not in seeds:
                seeds[mode] = {}

            seeds[mode][server_name] = {
                "port": server_config.port,
                "source": server_config.source or "",
                "framework": getattr(server_config, "framework", ""),
                "tags": server_config.tags or [],
                "env": {},  # env vars come from the config file; expand here if needed
                "description": server_config.description or "",
                "current_version": server_config.current_version,
            }
        return seeds

    async def initialize(self):
        """Initializes the MCPRegistry, including the database logger and deployers."""
        await self.db_logger.initialize()

        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            await self._sync_servers_to_db()
            await self._load_dynamic_servers_from_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(os.path.join(current_dir, '..', 'resources', 'docker'))

        # Initialize deployers for available modes
        seed_configs = self._build_seed_configs_from_servers()

        if os.environ.get('MCP_REGISTRY_URL'):
            mcp_base_url = os.environ.get('MCP_REGISTRY_URL')
        elif os.environ.get('MCP_BASE_URL'):
            mcp_base_url = f"{os.environ.get('MCP_BASE_URL')}:{os.environ.get('MCP_BASE_URL_PORT', self.registry_config.port)}"
        else:
            mcp_base_url = f"http://localhost:{self.registry_config.port}"

        if not mcp_base_url.startswith('http'):
            mcp_base_url = f"http://{mcp_base_url}"

        for mode in ["docker", "python_package"]:
            try:
                self.deployers[mode] = DeployerFactory.get_deployer(
                    mode=mode,
                    seed_config=seed_configs.get(mode, {}),
                    compose_output_path=os.path.join(build_dir, "docker-compose.generated.yaml"),
                    base_compose_path=os.path.join(build_dir, "docker-compose.yaml"),
                    agent_base_url=f"{mcp_base_url}",
                    agent_local_registry_url=f"http://host.docker.internal:{self.registry_config.port}",
                )
                logger.info(f"Deployer '{mode}' initialized with {len(seed_configs.get(mode, {}))} seed servers.")
                await self.deployers[mode].initialize()
            except NotImplementedError as e:
                logger.debug(f"Deployer '{mode}' not initialized: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize deployer '{mode}': {e}")

        self.initialize_proxies()

    async def shutdown(self):
        """Shuts down the MCPRegistry, including closing deployers and the database logger."""
        for server, server_config in self.config.servers.items():
            if getattr(server_config, 'registered_via', 'dynamic') != 'dynamic':
                deployer = self._get_deployer(getattr(server_config, 'deployment_mode', 'docker'))
                if deployer:
                    deployer.remove_server(server)

        for mode, deployer in self.deployers.items():
            logger.info(f"Shutting down deployer '{mode}'...")
            await deployer.shutdown()

        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    def _get_deployer(self, mode: str) -> Optional[BaseDeployer]:
        """Gets the appropriate deployer, defaulting to docker."""
        return self.deployers.get(mode) or self.deployers.get("docker")

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
            dynamic_servers = await self.db_logger.get_all_servers()
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
            endpoint = f"http://localhost:{port}/info"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(endpoint, timeout=1.0)
                    if response.status_code == 200:
                        server_info = response.json()
                        server_name = server_info.get("server_name")
                        if server_name and server_name not in self.config.servers:
                            server_config = ServerConfig(
                                endpoint=f"http://{host}:{port}",
                                description=server_info.get("server_config", {}).get("description",
                                                                                     "Auto-discovered MCP server"),
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

    async def _get_merged_server_values(self, server_name: str, server_registration: ServerRegistration,
                                        registered_via: str) -> Dict[str, Any]:
        """Merges new registration values with existing database values for partial updates."""
        existing = await self.db_logger.get_server_details(server_name) if self.db_logger.is_active else None
        if existing is None:
            existing = {}

        merged = {
            'endpoint': server_registration.endpoint if server_registration.endpoint is not None else existing.get(
                'endpoint', ''),
            'port': server_registration.port if server_registration.port is not None else existing.get('port'),
            'source': server_registration.source if server_registration.source is not None else existing.get('source',
                                                                                                             ''),
            'active': True,
            'registered_via': registered_via if registered_via is not None else existing.get('registered_via',
                                                                                             'dynamic'),
            'tags': server_registration.tags if server_registration.tags is not None else existing.get('tags', []),
            'description': server_registration.description if server_registration.description is not None else existing.get(
                'description', ''),
            'current_version': server_registration.current_version if server_registration.current_version is not None else existing.get(
                'current_version'),
            'available_versions': server_registration.available_versions if len(
                server_registration.available_versions) > 0 and server_registration.available_versions is not None else existing.get(
                'available_versions', []),
            'deployment_mode': server_registration.deployment_mode if server_registration.deployment_mode is not None else existing.get(
                'deployment_mode', 'docker')
        }
        return merged

    async def register_server(self, app: FastAPI, server_registration: ServerRegistration,
                              stream_output: bool = False) -> Union[JSONResponse, StreamingResponse]:
        """Registers a new MCP server dynamically via the /register endpoint."""
        server_name = server_registration.name
        if server_name in self.config.servers:
            logger.info(f"Server '{server_name}' is already registered. Updating its configuration.")

        registered_via = server_registration.registered_via or "dynamic"
        assigned_port = server_registration.port

        if server_registration.deployment_mode == 'unknown':
            if server_name in self.config.servers:
                server_registration.deployment_mode = self.config.servers[server_name].deployment_mode
            else:
                server_registration.deployment_mode = "docker"

        deployer = self._get_deployer(server_registration.deployment_mode)

        if registered_via == "registry" and deployer:
            if not assigned_port:
                assigned_port = deployer.find_available_port()
                server_registration.port = assigned_port

            if stream_output:
                async def stream_generator():
                    try:
                        yield f"data: Starting server registration and deployment for '{server_name}'...\n\n"
                        async for line in deployer.stream_deploy_server(
                                server_name=server_name,
                                source_url=server_registration.source,
                                framework=getattr(server_registration, "framework", None),
                                env={},
                                description=getattr(server_registration, "description", ""),
                                tags=getattr(server_registration, "tags", []),
                                port=assigned_port,
                                current_version=getattr(server_registration, "current_version", None),
                                refresh_repo=False,
                        ):
                            yield f"data: {line.strip()}\n\n"

                        db_values = await self._get_merged_server_values(server_name, server_registration,
                                                                         registered_via)
                        if assigned_port:
                            db_values['port'] = assigned_port
                        server_config = ServerConfig(**db_values)
                        self.config.servers[server_name] = server_config

                        # Mount or update the proxy
                        url = self._build_upstream_url(server_config)
                        mcp = FastMCP.as_proxy(url, name=server_name)
                        sub_app = mcp.http_app()
                        if server_name not in self.sub_apps:
                            app.mount(f"/{server_name}", sub_app)
                        self.sub_apps[server_name] = sub_app
                        if self.exit_stack:
                            await self.exit_stack.enter_async_context(sub_app.router.lifespan_context(sub_app))

                        await self.db_logger.log_server_registration(
                            server_name=server_name,
                            endpoint_url=db_values['endpoint'],
                            port=db_values['port'],
                            source=db_values['source'],
                            active=db_values['active'],
                            registered_via=db_values['registered_via'],
                            tags=db_values['tags'],
                            description=db_values['description'],
                            current_version=db_values['current_version'],
                            available_versions=db_values['available_versions'],
                            deployment_mode=db_values['deployment_mode']
                        )
                        yield f"data: ✅ Server '{server_name}' registered successfully.\n\n"
                    except Exception as e:
                        yield f"data: ❌ Error during registration: {str(e)}\n\n"
                        raise

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                asyncio.create_task(
                    deployer.deploy_server(
                        server_name=server_name,
                        source_url=server_registration.source,
                        framework=getattr(server_registration, "framework", None),
                        env={},  # pass server-specific env if available
                        description=getattr(server_registration, "description", ""),
                        tags=getattr(server_registration, "tags", []),
                        port=assigned_port,
                        current_version=getattr(server_registration, "current_version", None),
                        refresh_repo=False,
                    )
                )

        # Non-streaming or non-registry dynamic mode
        db_values = await self._get_merged_server_values(server_name, server_registration, registered_via)
        if assigned_port:
            db_values['port'] = assigned_port

        server_config = ServerConfig(**db_values)
        self.config.servers[server_name] = server_config

        # Mount proxy if not already mounted; update endpoint if it is
        try:
            url = self._build_upstream_url(server_config)
            if server_name in self.sub_apps:
                if url:
                    # Server already mounted — re-create the proxy pointed at the
                    # new URL and replace the entry in sub_apps.  FastAPI does not
                    # support unmounting, so the old mount stays in the router but
                    # the middleware in app.py (which reads config.servers) will
                    # route traffic to the updated endpoint from this point on.
                    mcp = FastMCP.as_proxy(url, name=server_name)
                    sub_app = mcp.http_app()
                    self.sub_apps[server_name] = sub_app
                    if self.exit_stack:
                        await self.exit_stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
                    logger.info(f"Updated proxy for existing server '{server_name}' -> {url}")
            else:
                if url:
                    mcp = FastMCP.as_proxy(url, name=server_name)
                    sub_app = mcp.http_app()
                    app.mount(f"/{server_name}", sub_app)
                    self.sub_apps[server_name] = sub_app
                    if self.exit_stack:
                        await self.exit_stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
                    logger.info(f"Mounted new proxy for server '{server_name}' -> {url}")
        except Exception as e:
            logger.error(f"Failed to mount/update sub app proxy for {server_name}: {e}")

        await self.db_logger.log_server_registration(
            server_name=server_name,
            endpoint_url=db_values['endpoint'],
            port=db_values['port'],
            source=db_values['source'],
            active=db_values['active'],
            registered_via=db_values['registered_via'],
            tags=db_values['tags'],
            description=db_values['description'],
            current_version=db_values['current_version'],
            available_versions=db_values['available_versions'],
            deployment_mode=db_values['deployment_mode']
        )

        return JSONResponse({"message": f"Server '{server_name}' registered successfully."})

    async def deregister_server(self, server_deregistration: ServerDeregistration) -> JSONResponse:
        """Deregisters an MCP server."""
        server_name = server_deregistration.name

        if server_name not in self.config.servers:
            logger.warning(f"Attempted to deregister server '{server_name}', but it was not found.")
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        # Remove from running apps and stop deployer if running
        if server_name in self.sub_apps:
            del self.sub_apps[server_name]

        server_config = self.config.servers[server_name]
        deployer = self._get_deployer(getattr(server_config, 'deployment_mode', 'docker'))
        if deployer:
            deployer.remove_server(server_name)

        logger.info(f"Deactivating server '{server_name}'.")

        await self.db_logger.deregister_server(server_name=server_name)

        return JSONResponse({"message": f"Server '{server_name}' deactivated successfully."})

    async def execute_lifecycle_action(self, server_name: str, action: str, version: Optional[str] = None,
                                       stream_output: bool = False) -> Union[JSONResponse, StreamingResponse]:
        """Executes a lifecycle action on a server."""
        if server_name not in self.config.servers:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        server_config = self.config.servers[server_name]
        deployer = self._get_deployer(getattr(server_config, 'deployment_mode', 'docker'))

        logger.info(
            f"Executing lifecycle action '{action}' for server '{server_name}' using deployer '{getattr(server_config, 'deployment_mode', 'docker')}'")

        if deployer:
            try:
                if action in ["update", "upgrade", "downgrade"]:
                    if not version:
                        raise HTTPException(status_code=400,
                                            detail="Version is required for update/upgrade/downgrade action.")

                    image_already_exists = deployer.image_exists(server_name, version)

                    # Update version in memory
                    server_config.current_version = version
                    if server_config.available_versions is None:
                        server_config.available_versions = []
                    if version not in server_config.available_versions:
                        server_config.available_versions.append(version)

                    # Save to database
                    await self.db_logger.log_server_registration(
                        server_name=server_name,
                        endpoint_url=server_config.endpoint,
                        port=server_config.port,
                        source=server_config.source,
                        active=True,
                        registered_via=server_config.registered_via,
                        tags=server_config.tags,
                        description=server_config.description,
                        current_version=server_config.current_version,
                        available_versions=server_config.available_versions,
                        deployment_mode=server_config.deployment_mode
                    )

                    if stream_output:
                        _no_build = image_already_exists

                        async def stream_generator():
                            try:
                                yield f"data: Server '{server_name}' switching to version {version}...\n\n"
                                async for line in deployer.stream_deploy_server(
                                        server_name=server_name,
                                        source_url=server_config.source,
                                        framework=getattr(server_config, "framework", None),
                                        env={},
                                        description=server_config.description,
                                        tags=server_config.tags,
                                        port=server_config.port,
                                        current_version=server_config.current_version,
                                        refresh_repo=not _no_build,
                                        no_build=_no_build,
                                ):
                                    yield f"data: {line.strip()}\n\n"
                                yield f"data: Server '{server_name}' successfully switched to {version}!\n\n"
                            except Exception as e:
                                yield f"data: Error during version switch: {str(e)}\n\n"
                                raise

                        return StreamingResponse(stream_generator(), media_type="text/event-stream")
                    else:
                        await deployer.deploy_server(
                            server_name=server_name,
                            source_url=server_config.source,
                            framework=getattr(server_config, "framework", None),
                            env={},
                            description=server_config.description,
                            tags=server_config.tags,
                            port=server_config.port,
                            current_version=server_config.current_version,
                            refresh_repo=not image_already_exists,
                            no_build=image_already_exists,
                        )

                elif action == "stop":
                    deployer.stop_server(server_name)

                elif action == "start":
                    deployer.start_server(server_name)

                elif action == "restart":
                    deployer.remove_server(server_name)
                    logger.info(f"Server '{server_name}' stopped. Restarting...")
                    await asyncio.sleep(1)
                    deployer.start_server(server_name)

                elif action == "rebuild":
                    if stream_output:
                        async def stream_generator():
                            try:
                                deployer.remove_server(server_name)
                                yield f"data: Server '{server_name}' stopped. Rebuilding image...\n\n"
                                async for line in deployer.stream_deploy_server(
                                        server_name=server_name,
                                        source_url=server_config.source,
                                        framework=getattr(server_config, "framework", None),
                                        env={},
                                        description=server_config.description,
                                        tags=server_config.tags,
                                        port=server_config.port,
                                        current_version=server_config.current_version,
                                        refresh_repo=True,
                                ):
                                    yield f"data: {line.strip()}\n\n"
                                yield f"data: Server '{server_name}' rebuild completed successfully!\n\n"
                            except Exception as e:
                                yield f"data: Error during rebuild: {str(e)}\n\n"
                                raise

                        return StreamingResponse(stream_generator(), media_type="text/event-stream")
                    else:
                        deployer.remove_server(server_name)
                        await deployer.deploy_server(
                            server_name=server_name,
                            source_url=server_config.source,
                            framework=getattr(server_config, "framework", None),
                            env={},
                            description=server_config.description,
                            tags=server_config.tags,
                            port=server_config.port,
                            current_version=server_config.current_version,
                            refresh_repo=True,
                        )

                elif action == "redeploy":
                    if stream_output:
                        async def stream_generator():
                            try:
                                yield f"data: Starting async redeploy for server '{server_name}'...\n\n"
                                asyncio.create_task(
                                    deployer.deploy_server(
                                        server_name=server_name,
                                        source_url=server_config.source,
                                        framework=getattr(server_config, "framework", None),
                                        env={},
                                        description=server_config.description,
                                        tags=server_config.tags,
                                        port=server_config.port,
                                        current_version=server_config.current_version,
                                        refresh_repo=True,
                                    )
                                )
                                yield f"data: ✅ Redeploy task initiated for '{server_name}'\n\n"
                            except Exception as e:
                                yield f"data: Error initiating redeploy: {str(e)}\n\n"
                                raise

                        return StreamingResponse(stream_generator(), media_type="text/event-stream")
                    else:
                        asyncio.create_task(
                            deployer.deploy_server(
                                server_name=server_name,
                                source_url=server_config.source,
                                framework=getattr(server_config, "framework", None),
                                env={},
                                description=server_config.description,
                                tags=server_config.tags,
                                port=server_config.port,
                                current_version=server_config.current_version,
                                refresh_repo=True,
                            )
                        )
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action '{action}'")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to execute lifecycle action '{action}' for server '{server_name}': {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse({"message": f"Lifecycle action '{action}' executed for server '{server_name}'."})