import json
import logging
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, Union

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

from oai_agent_registry.models import Config, AgentConfig, RegistryConfig, AgentRegistration, AgentDeregistration
from oai_agent_registry.security.dependencies import _validate_token
from oai_agent_registry.services.db.database_logger import RegistryDatabaseLogger
from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.services.deployers.factory import DeployerFactory

logger = logging.getLogger(__name__)

class AgentRegistry:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('REGISTRY_CONFIG_PATH', '../config/registry_config.json')
        self.config: Optional[Config] = None
        self.agents: Dict[str, AgentConfig] = {}
        self.registry_config: RegistryConfig = RegistryConfig()
        self.start_time = time.time()
        self.client: Optional[httpx.AsyncClient] = None
        self.public_ip: Optional[str] = None
        self.private_ip: Optional[str] = None
        self.db_logger: RegistryDatabaseLogger = RegistryDatabaseLogger(logger=logger)
        self.deployers: Dict[str, BaseDeployer] = {}

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
                    agent_data_copy = agent_data.copy()
                    if 'enabled' not in agent_data_copy:
                        agent_data_copy['enabled'] = False
                    agent_data_copy['registered_via'] = 'config'
                    self.agents[agent_name] = AgentConfig(**agent_data_copy)

            logger.info(f"Loaded configuration with {len(self.agents)} agents.")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    async def initialize(self):
        """Initializes the AgentRegistry, including the database logger and HTTP client."""
        self.client = httpx.AsyncClient()
        await self.db_logger.initialize()

        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            await self._load_agents_from_db()
            await self._sync_agents_to_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(os.path.join(current_dir, '..', 'resources', 'docker'))

        # Initialize deployers for available modes
        seed_configs = self._build_seed_configs_from_agents()
        
        for mode in ["docker", "python_package"]:
            try:
                self.deployers[mode] = DeployerFactory.get_deployer(
                    mode=mode,
                    seed_config=seed_configs.get(mode, {}),
                    compose_output_path=os.path.join(build_dir, "docker-compose.generated.yaml"),
                    base_compose_path=os.path.join(build_dir, "docker-compose.yaml"),
                    agent_base_url=f"{os.environ.get('AGENT_BASE_URL', 'localhost')}:{os.environ.get('AGENT_BASE_URL_PORT', self.registry_config.port)}",
                    agent_local_registry_url=f"http://host.docker.internal:{self.registry_config.port}",
                )
                logger.info(f"Deployer '{mode}' initialized with {len(seed_configs.get(mode, {}))} seed agents.")
                await self.deployers[mode].initialize()
            except NotImplementedError as e:
                logger.debug(f"Deployer '{mode}' not initialized: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize deployer '{mode}': {e}")

        # Status check runs last so it validates both config and restored dynamic agents.
        await self._check_agent_statuses()

    async def shutdown(self):
        """Shuts down the AgentRegistry, including closing the database logger and HTTP client."""
        if self.client:
            await self.client.aclose()
            
        for agent, agent_config in self.agents.items():
            if agent_config.registered_via != 'dynamic':
                deployer = self._get_deployer(agent_config.deployment_mode)
                if deployer:
                    deployer.remove_agent(agent)
                    
        for mode, deployer in self.deployers.items():
            logger.info(f"Shutting down deployer '{mode}'...")
            await deployer.shutdown()
            
        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    def _get_deployer(self, mode: str) -> Optional[BaseDeployer]:
        """Gets the appropriate deployer, defaulting to docker."""
        return self.deployers.get(mode) or self.deployers.get("docker")

    async def _sync_agents_to_db(self):
        """Synchronizes config-declared agents to the database, marking them as registered_via='config'."""
        if not self.db_logger.is_active:
            logger.warning("Database logger is not active, skipping agent sync to DB.")
            return

        for agent_name, agent_config in self.agents.items():
            try:
                await self.db_logger.log_agent_registration(
                    agent_name=agent_name,
                    endpoint_url=agent_config.endpoint,
                    port=agent_config.port,
                    source=agent_config.source,
                    active=agent_config.enabled,
                    registered_via="config",
                    framework=agent_config.framework,
                    prompts=agent_config.prompts,
                    tags=agent_config.tags,
                    description=agent_config.description,
                    current_version=agent_config.current_version,
                    available_versions=agent_config.available_versions,
                    deployment_mode=agent_config.deployment_mode
                )
                logger.debug(f"Synced config agent '{agent_name}' to DB.")
            except Exception as e:
                logger.error(f"Failed to sync agent '{agent_name}' to DB from config: {e}")

    async def _load_agents_from_db(self):
        """Restores dynamic agents from the DB that were active before the registry restarted."""
        logger.info("Restoring active dynamic agents from database...")
        try:
            agents = await self.db_logger.get_all_agents()
        except Exception as e:
            logger.error(f"Failed to load dynamic agents from DB: {e}")
            return

        restored = 0
        skipped = 0
        for row in agents:
            agent_name = row.get("agent_name")
            if not agent_name:
                continue

            try:
                agent_config = AgentConfig(
                    name=agent_name,
                    endpoint=row.get("endpoint_url", ""),
                    port=row.get("port"),
                    source=row.get("source"),
                    enabled=False, # Start as disabled — _check_agent_statuses() will enable if the container is actually reachable.
                    framework=row.get("framework"),
                    prompts=row.get("prompts", []),
                    tags=row.get("tags", []),
                    current_version=row.get("current_version"),
                    available_versions=row.get("available_versions", []),
                    registered_via=row.get("registered_via", 'dynamic'),
                    deployment_mode=row.get("deployment_mode", 'docker')
                )
                self.agents[agent_name] = agent_config
                restored += 1
                logger.debug(f"Restored agent '{agent_name}' from DB (pending status check).")
            except Exception as e:
                logger.error(f"Failed to restore dynamic agent '{agent_name}' from DB: {e}")

        logger.info(f"Dynamic agent restore complete: {restored} restored, {skipped} skipped (config collision).")

    async def _check_agent_statuses(self):
        """
        Checks the /status endpoint of each agent and updates its enabled status.
        For dynamic agents restored from DB that fail the status check, we also mark
        them inactive in the DB so stale entries don't accumulate.
        """
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
                    logger.warning(f"Agent '{agent_name}' returned status {response.status_code}. Marking inactive.")
                    await self._mark_agent_inactive_in_db(agent_name)
            except httpx.RequestError as e:
                agent_config.enabled = False
                logger.warning(f"Could not reach agent '{agent_name}' at {agent_config.endpoint} ({e}). Marking inactive.")
                await self._mark_agent_inactive_in_db(agent_name)
            except Exception as e:
                agent_config.enabled = False
                logger.error(f"Error checking status for agent '{agent_name}': {e}. Marking inactive.")
                await self._mark_agent_inactive_in_db(agent_name)

    async def _mark_agent_inactive_in_db(self, agent_name: str) -> None:
        """
        Marks an agent inactive in the DB after a failed status check.
        Only relevant for DB-restored dynamic agents — config agents are expected
        to be temporarily unreachable without being permanently deregistered.
        Errors are logged but never raised so a single DB failure doesn't
        block the rest of the startup status sweep.
        """
        try:
            await self.db_logger.deregister_agent(agent_name=agent_name)
        except Exception as e:
            logger.error(f"Failed to mark agent '{agent_name}' inactive in DB after status check: {e}")

    async def _get_merged_agent_values(self, agent_name: str, agent_registration: AgentRegistration, registered_via: str) -> Dict[str, Any]:
        """
        Merges new registration values with existing database values.
        If a value in the new registration is None/null, the existing database value is preserved.
        This allows for partial updates without overwriting existing data.
        """
        # Try to fetch existing agent values from database
        existing = await self.db_logger.get_agent_details(agent_name) if self.db_logger.is_active else None

        if existing is None:
            existing = {}

        # Build merged values: use new value if not None, otherwise use existing
        merged = {
            'endpoint': agent_registration.endpoint if agent_registration.endpoint is not None else existing.get('endpoint_url', ''),
            'port': agent_registration.port if agent_registration.port is not None else existing.get('port'),
            'source': agent_registration.source if agent_registration.source is not None else existing.get('source', ''),
            'active': True,  # Registration always sets to active
            'registered_via': registered_via if registered_via is not None else existing.get('registered_via', 'dynamic'),
            'framework': agent_registration.framework if agent_registration.framework is not None else existing.get('framework'),
            'prompts': agent_registration.prompts if agent_registration.prompts is not None else existing.get('prompts', []),
            'tags': agent_registration.tags if agent_registration.tags is not None else existing.get('tags', []),
            'description': agent_registration.description if agent_registration.description is not None else existing.get('description', ''),
            'current_version': agent_registration.current_version if agent_registration.current_version is not None else existing.get('current_version'),
            'available_versions': agent_registration.available_versions if len(agent_registration.available_versions) > 0 and agent_registration.available_versions is not None else existing.get('available_versions', []),
            'deployment_mode': agent_registration.deployment_mode if agent_registration.deployment_mode is not None else existing.get('deployment_mode', 'docker')
        }

        logger.debug(f"Merged values for agent '{agent_name}': {merged}")
        return merged

    async def discover_agents(self):
        """Discover agents by scanning a range of ports."""
        start = self.registry_config.start_port
        end = self.registry_config.end_port
        host = self.registry_config.host or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        logger.info(f"Starting auto-discovery of agents in port range {start}-{end} on host {host}...")

        # Reuse the shared client rather than creating a new one per port.
        if not self.client:
            logger.error("HTTP client not initialized. Cannot run agent discovery.")
            return

        for port in range(start, end + 1):
            endpoint = f"http://{host}:{port}"
            try:
                response = await self.client.get(f"{endpoint}/info", timeout=1.0)
                if response.status_code == 200:
                    agent_info = response.json()
                    agent_name = agent_info.get("agent_name")
                    if agent_name and agent_name not in self.agents:
                        agent_config = AgentConfig(
                            name=agent_name,
                            endpoint=endpoint,
                            description=agent_info.get("description", "Auto-discovered agent"),
                            framework=agent_info.get("framework"),
                            source=agent_info.get("source"),
                            prompts=agent_info.get("prompts", []),
                            tags=agent_info.get("tags", []),
                            current_version=agent_info.get("current_version"),
                            available_versions=agent_info.get("available_versions", []),
                            deployment_mode=agent_info.get("deployment_mode", "docker"),
                        )
                        self.agents[agent_name] = agent_config
                        logger.info(f"Discovered agent '{agent_name}' at {endpoint}")
                        await self.db_logger.log_agent_registration(
                            agent_name=agent_name,
                            endpoint_url=endpoint,
                            port=port,
                            source=agent_info.get("source"),
                            active=True,
                            registered_via="dynamic",
                            framework=agent_info.get("framework"),
                            prompts=agent_info.get("prompts", []),
                            tags=agent_info.get("tags", []),
                            description=agent_info.get("description"),
                            current_version=agent_info.get("current_version"),
                            available_versions=agent_info.get("available_versions", []),
                            deployment_mode=agent_info.get("deployment_mode", "docker")
                        )
            except (httpx.RequestError, json.JSONDecodeError):
                pass

    async def get_info(self) -> JSONResponse:
        """Returns information about the registry and its agents."""
        enabled_agents = {name for name, agent in self.agents.items() if agent.enabled}
        endpoint = self.private_ip if os.environ.get('USE_PRIVATE_IP', 'false').lower() == 'true' else \
            os.environ.get('AGENT_BASE_URL', "localhost")
        info = {
            "registry": {
                "uptime_seconds": time.time() - self.start_time,
                "total_agents": len(self.agents),
                "enabled_agents": len(enabled_agents),
            },
            "agents": {
                name: {
                    "endpoint": f"{endpoint}:{self.registry_config.port}/{name}",
                    "enabled": agent.enabled,
                    "status": "active" if agent.enabled else 'inactive',
                    "description": getattr(agent, "description", None),
                    "framework": getattr(agent, "framework", None),
                    "registered_via": getattr(agent, "registered_via", "dynamic"),
                    "current_version": getattr(agent, "current_version", None),
                    "available_versions": getattr(agent, "available_versions", []),
                    "deployment_mode": getattr(agent, "deployment_mode", "docker"),
                    "available_actions": [
                        "start" if not agent.enabled else "stop",
                        "restart",
                        "rebuild",
                        "redeploy",
                        "update",
                        "upgrade",
                        "downgrade",
                    ]
                }
                for name, agent in self.agents.items() if agent.endpoint is not None
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
            await self._sync_agents_to_db()
            await self._check_agent_statuses()
            return JSONResponse({"message": "Configuration reloaded", "agents": list(self.agents.keys())})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def register_agent(self, agent_registration: AgentRegistration) -> JSONResponse:
        """Registers a new agent dynamically via the /register endpoint."""
        agent_name = agent_registration.name
        if agent_name in self.agents:
            logger.info(f"Agent '{agent_name}' is already registered. Updating its configuration.")

        registered_via = agent_registration.registered_via or "dynamic"

        assigned_port = agent_registration.port

        if agent_registration.deployment_mode == 'unknown':
            deployment_mode = self.agents[agent_name].deployment_mode
            agent_registration.deployment_mode = deployment_mode
        deployer = self._get_deployer(agent_registration.deployment_mode)

        if registered_via == "registry" and deployer:
            if not assigned_port:
                assigned_port = deployer.find_available_port()
                agent_registration.port = assigned_port

            asyncio.create_task(
                deployer.deploy_agent(
                    agent_name=agent_name,
                    source_url=agent_registration.source,
                    framework=agent_registration.framework,
                    env={},  # pass agent-specific env if available
                    description=getattr(agent_registration, "description", ""),
                    tags=getattr(agent_registration, "tags", []),
                    port=assigned_port,
                    current_version=getattr(agent_registration, "current_version", None),
                    refresh_repo=False,
                )
            )

        # Merge incoming values with existing database values for partial updates
        db_values = await self._get_merged_agent_values(agent_name, agent_registration, registered_via)

        # Ensure the assigned port makes it to the database
        if assigned_port:
            db_values['port'] = assigned_port

        agent_config = AgentConfig(**db_values)
        self.agents[agent_name] = agent_config

        logger.info(f"Registered agent '{agent_name}' with endpoint {agent_config.endpoint}")

        await self.db_logger.log_agent_registration(
            agent_name=agent_name,
            endpoint_url=db_values['endpoint'],
            port=db_values['port'],
            source=db_values['source'],
            active=db_values['active'],
            registered_via=db_values['registered_via'],
            framework=db_values['framework'],
            prompts=db_values['prompts'],
            tags=db_values['tags'],
            description=db_values['description'],
            current_version=db_values['current_version'],
            available_versions=db_values['available_versions'],
            deployment_mode=db_values['deployment_mode']
        )

        return JSONResponse({"message": f"Agent '{agent_name}' registered successfully."})

    async def deregister_agent(self, agent_deregistration: AgentDeregistration) -> JSONResponse:
        """Deregisters an agent by setting its active flag to False in DB and disabling it in memory."""
        agent_name = agent_deregistration.name

        if agent_name not in self.agents:
            logger.warning(f"Attempted to deregister agent '{agent_name}', but it was not found.")
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        self.agents[agent_name].enabled = False
        logger.info(f"Deactivating agent '{agent_name}'.")

        await self.db_logger.deregister_agent(agent_name=agent_name)

        return JSONResponse({"message": f"Agent '{agent_name}' deactivated successfully."})

    async def execute_lifecycle_action(self, agent_name: str, action: str, version: Optional[str] = None,
                                       stream_output: bool = False) -> Union[JSONResponse, StreamingResponse]:
        """
        Executes a lifecycle action on an agent.
        """
        if agent_name not in self.agents:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        agent_config = self.agents[agent_name]
        deployer = self._get_deployer(agent_config.deployment_mode)

        logger.info(f"Executing lifecycle action '{action}' for agent '{agent_name}' using deployer '{agent_config.deployment_mode}'")

        if deployer:
            try:
                if action in ["update", "upgrade", "downgrade"]:
                    if not version:
                        raise HTTPException(status_code=400,
                                            detail="Version is required for update/upgrade/downgrade action.")

                    image_already_exists = deployer.image_exists(agent_name, version)

                    if image_already_exists:
                        logger.info(
                            f"Artifact for '{agent_name}:{version}' found locally — skipping build for version switch."
                        )
                    else:
                        logger.info(
                            f"Artifact for '{agent_name}:{version}' not found locally — will build for version switch."
                        )

                    # Update version in memory
                    agent_config.current_version = version
                    if agent_config.available_versions is None:
                        agent_config.available_versions = []
                    if version not in agent_config.available_versions:
                        agent_config.available_versions.append(version)

                    # Save version update to database
                    await self.db_logger.log_agent_registration(
                        agent_name=agent_name,
                        endpoint_url=agent_config.endpoint,
                        port=agent_config.port,
                        source=agent_config.source,
                        active=agent_config.enabled,
                        registered_via=agent_config.registered_via,
                        framework=agent_config.framework,
                        prompts=agent_config.prompts,
                        tags=agent_config.tags,
                        description=agent_config.description,
                        current_version=agent_config.current_version,
                        available_versions=agent_config.available_versions,
                        deployment_mode=agent_config.deployment_mode
                    )

                    if stream_output:
                        # Capture image_already_exists in closure
                        _no_build = image_already_exists

                        async def stream_generator():
                            try:
                                if _no_build:
                                    yield f"data: Artifact for '{agent_name}:{version}' found locally — reusing without rebuild.\n\n"
                                else:
                                    yield f"data: Artifact for '{agent_name}:{version}' not found locally — building now.\n\n"
                                yield f"data: Agent '{agent_name}' switching to version {version}...\n\n"
                                async for line in deployer.stream_deploy_agent(
                                        agent_name=agent_name,
                                        source_url=agent_config.source,
                                        framework=agent_config.framework,
                                        env={},
                                        description=agent_config.description,
                                        tags=agent_config.tags,
                                        port=agent_config.port,
                                        current_version=agent_config.current_version,
                                        refresh_repo=not _no_build,
                                        no_build=_no_build,
                                ):
                                    yield f"data: {line.strip()}\n\n"
                                self.agents[agent_name].enabled = True
                                yield f"data: Agent '{agent_name}' successfully switched to {version}!\n\n"
                            except Exception as e:
                                yield f"data: Error during version switch: {str(e)}\n\n"
                                raise

                        return StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                        )
                    else:
                        logger.info(f"Agent '{agent_name}' switching to version {version}...")
                        await deployer.deploy_agent(
                            agent_name=agent_name,
                            source_url=agent_config.source,
                            framework=agent_config.framework,
                            env={},
                            description=agent_config.description,
                            tags=agent_config.tags,
                            port=agent_config.port,
                            current_version=agent_config.current_version,
                            refresh_repo=not image_already_exists,
                            no_build=image_already_exists,
                        )
                        self.agents[agent_name].enabled = True

                elif action == "stop":
                    deployer.stop_agent(agent_name)
                    self.agents[agent_name].enabled = False

                elif action == "start":
                    deployer.start_agent(agent_name)
                    self.agents[agent_name].enabled = True

                elif action == "restart":
                    deployer.remove_agent(agent_name)
                    self.agents[agent_name].enabled = False
                    logger.info(f"Agent '{agent_name}' stopped. Restarting...")
                    await asyncio.sleep(1)
                    deployer.start_agent(agent_name)
                    self.agents[agent_name].enabled = True

                elif action == "rebuild":
                    if stream_output:
                        async def stream_generator():
                            try:
                                deployer.remove_agent(agent_name)
                                self.agents[agent_name].enabled = False
                                yield f"data: Agent '{agent_name}' stopped. Rebuilding image...\n\n"

                                async for line in deployer.stream_deploy_agent(
                                        agent_name=agent_name,
                                        source_url=agent_config.source,
                                        framework=agent_config.framework,
                                        env={},
                                        description=agent_config.description,
                                        tags=agent_config.tags,
                                        port=agent_config.port,
                                        current_version=agent_config.current_version,
                                        refresh_repo=True,
                                ):
                                    yield f"data: {line.strip()}\n\n"

                                self.agents[agent_name].enabled = True
                                yield f"data: Agent '{agent_name}' rebuild completed successfully!\n\n"

                            except Exception as e:
                                yield f"data: Error during rebuild: {str(e)}\n\n"
                                raise

                        return StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                        )
                    else:
                        deployer.remove_agent(agent_name)
                        self.agents[agent_name].enabled = False
                        logger.info(f"Agent '{agent_name}' stopped. Rebuilding image...")
                        await deployer.deploy_agent(
                            agent_name=agent_name,
                            source_url=agent_config.source,
                            framework=agent_config.framework,
                            env={},
                            description=agent_config.description,
                            tags=agent_config.tags,
                            port=agent_config.port,
                            current_version=agent_config.current_version,
                            refresh_repo=True,
                        )
                        self.agents[agent_name].enabled = True

                elif action == "redeploy":
                    if stream_output:
                        async def stream_generator():
                            try:
                                yield f"data: Starting async redeploy for agent '{agent_name}'...\n\n"
                                
                                asyncio.create_task(
                                    deployer.deploy_agent(
                                        agent_name=agent_name,
                                        source_url=agent_config.source,
                                        framework=agent_config.framework,
                                        env={},
                                        description=agent_config.description,
                                        tags=agent_config.tags,
                                        port=agent_config.port,
                                        current_version=agent_config.current_version,
                                        refresh_repo=True,
                                    )
                                )
                                
                                yield f"data: ✅ Redeploy task initiated for '{agent_name}'\n\n"
                                yield f"data: Note: This is asynchronous - the agent will be updated in the background\n\n"
                                
                            except Exception as e:
                                yield f"data: Error initiating redeploy: {str(e)}\n\n"
                                raise

                        return StreamingResponse(
                            stream_generator(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                        )
                    else:
                        asyncio.create_task(
                            deployer.deploy_agent(
                                agent_name=agent_name,
                                source_url=agent_config.source,
                                framework=agent_config.framework,
                                env={},
                                description=agent_config.description,
                                tags=agent_config.tags,
                                port=agent_config.port,
                                current_version=agent_config.current_version,
                                refresh_repo=True,
                            )
                        )
                else:
                    raise HTTPException(status_code=400,
                                        detail=f"Unknown action '{action}'. Supported actions: start, stop, restart, rebuild, redeploy, update, upgrade, downgrade")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to execute lifecycle action '{action}' for agent '{agent_name}': {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse({
            "message": f"Lifecycle action '{action}' executed for agent '{agent_name}'.",
            "agent": agent_name,
            "action": action,
            "status": "completed" if action != "redeploy" else "initiated",
            "enabled": self.agents[agent_name].enabled
        })

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

        return await self._proxy_regular_request(agent_name, target_url, request.method, headers, body,
                                                 agent_config.timeout, path)

    async def _proxy_regular_request(self, agent_name: str, url: str, method: str, headers: dict, body: bytes,
                                     timeout: int, path: str) -> Response:
        try:
            response = await self.client.request(method, url, headers=headers, content=body, timeout=timeout)

            response_headers = self._clean_response_headers(response.headers)
            content = response.content

            if path and path.rstrip("/") in ["docs", "redoc"] and "text/html" in response.headers.get("content-type",
                                                                                                      ""):
                try:
                    text = response.text
                    text = text.replace('"/openapi.json"', f'"/{agent_name}/openapi.json"')
                    text = text.replace("'/openapi.json'", f"'/{agent_name}/openapi.json'")
                    content = text.encode("utf-8")
                    if "content-length" in response_headers:
                        del response_headers["content-length"]
                except Exception as e:
                    logger.warning(f"Failed to rewrite docs for {agent_name}: {e}")

            elif path and path.rstrip("/") == "openapi.json" and "application/json" in response.headers.get(
                    "content-type", ""):
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

            return Response(content=content, status_code=response.status_code, headers=response_headers,
                            media_type=response.headers.get("content-type"))
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Request to agent '{agent_name}' timed out.")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Proxy error: {e}")

    @staticmethod
    async def _proxy_streaming_request(url: str, method: str, headers: dict, body: bytes,
                                       timeout: int) -> StreamingResponse:
        async def stream_generator():
            try:
                async with httpx.AsyncClient(timeout=timeout) as stream_client:
                    async with stream_client.stream(method, url, headers=headers, content=body) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    @staticmethod
    def _clean_response_headers(headers: httpx.Headers) -> dict:
        excluded = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
        return {k: v for k, v in headers.items() if k.lower() not in excluded}

    def _build_seed_configs_from_agents(self) -> dict:
        """
        Groups config agents by deployment mode for factory initialization.
        """
        seeds: dict = {}
        for agent_name, agent_config in self.agents.items():
            mode = agent_config.deployment_mode or "docker"
            if mode not in seeds:
                seeds[mode] = {}
            
            seeds[mode][agent_name] = {
                "port": agent_config.port,
                "source": agent_config.source or "",
                "framework": agent_config.framework or "",
                "tags": [agent_config.framework.lower()] if agent_config.framework else [],
                "env": {},  # env vars come from the config file; expand here if needed
                "description": "",
                "current_version": agent_config.current_version,
            }
        return seeds