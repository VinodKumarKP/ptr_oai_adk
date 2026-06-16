"""
Agent Registry Service
Manages agent lifecycle: registration, deployment, proxying, and env-var configuration.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from oai_agent_registry.models import (
    AgentConfig,
    AgentDeregistration,
    AgentRegistration,
    Config,
    RegistryConfig,
)
from oai_agent_registry.security.dependencies import _validate_token
from oai_agent_registry.services.db.database_logger import RegistryDatabaseLogger
from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.services.deployers.factory import DeployerFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _mask_sensitive_env(env_vars: dict, sensitive_vars: list) -> dict:
    """Return env_vars with sensitive values replaced by '***'."""
    if not env_vars:
        return {}
    mask = set(sensitive_vars or [])
    return {k: "***" if k in mask else v for k, v in env_vars.items()}


def _sse(gen) -> StreamingResponse:
    """Wrap an async generator as a Server-Sent Events StreamingResponse."""
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv(
            "REGISTRY_CONFIG_PATH", "../config/registry_config.json"
        )
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

    # ── Config ────────────────────────────────────────────────────────────

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if not os.path.exists(self.config_path):
                logger.warning("Config file not found at %s, using defaults.", self.config_path)
                self.config = Config(agents={})
            else:
                with open(self.config_path, "r") as f:
                    config_data = json.load(f)
                self.config = Config(**config_data)

            self.registry_config = self.config.registry
            self.agents = {}
            for agent_name, agent_data in self.config.agents.items():
                if isinstance(agent_data, str):
                    self.agents[agent_name] = AgentConfig(
                        endpoint=agent_data, name=agent_name, enabled=False
                    )
                elif isinstance(agent_data, dict):
                    agent_data_copy = agent_data.copy()
                    agent_data_copy.setdefault("enabled", False)
                    agent_data_copy["registered_via"] = "config"
                    self.agents[agent_name] = AgentConfig(**agent_data_copy)

            logger.info("Loaded configuration with %d agents.", len(self.agents))
        except Exception as exc:
            logger.error("Error loading config: %s", exc)
            raise

    # ── Shared helpers ────────────────────────────────────────────────────

    def _get_deployer(self, mode: str) -> Optional[BaseDeployer]:
        """Return the deployer for the given mode, defaulting to docker."""
        return self.deployers.get(mode) or self.deployers.get("docker")

    def _agent_deploy_kwargs(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Build keyword arguments for stream_deploy_agent / deploy_agent calls."""
        base = {
            "agent_name": agent_name,
            "source_url": agent_config.source or "",
            "framework": agent_config.framework,
            "env": agent_config.env_vars or {},
            "description": agent_config.description or "",
            "tags": agent_config.tags or [],
            "port": agent_config.port,
            "current_version": agent_config.current_version,
        }
        base.update(overrides)
        return base

    async def _persist_agent_to_db(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        registered_via: Optional[str] = None,
    ) -> bool:
        """Persist agent config to the database. Returns True on success."""
        try:
            await self.db_logger.log_agent_registration(
                agent_name=agent_name,
                endpoint_url=agent_config.endpoint or "",
                port=agent_config.port,
                source=agent_config.source,
                active=agent_config.enabled,
                registered_via=registered_via or agent_config.registered_via,
                framework=agent_config.framework,
                prompts=agent_config.prompts or [],
                tags=agent_config.tags or [],
                description=agent_config.description or "",
                current_version=agent_config.current_version,
                available_versions=agent_config.available_versions or [],
                deployment_mode=agent_config.deployment_mode or "docker",
                env_vars=agent_config.env_vars,
                sensitive_vars=agent_config.sensitive_vars,
            )
            return True
        except Exception as exc:
            logger.warning("DB persist for agent '%s' failed: %s", agent_name, exc)
            return False

    def _lifecycle_json(
        self,
        agent_name: str,
        action: str,
        status: str = "completed",
    ) -> JSONResponse:
        """Build a standard lifecycle action response."""
        enabled = self.agents[agent_name].enabled if agent_name in self.agents else False
        return JSONResponse({
            "message": f"Lifecycle action '{action}' executed for agent '{agent_name}'.",
            "agent": agent_name,
            "action": action,
            "status": status,
            "enabled": enabled,
        })

    # ── Initialize ────────────────────────────────────────────────────────

    def _deployer_factory_kwargs(self, build_dir: str, seed_config: dict) -> dict:
        """Common keyword arguments for DeployerFactory.get_deployer()."""
        port = os.environ.get("AGENT_BASE_URL_PORT", self.registry_config.port)
        base_url = os.environ.get("AGENT_BASE_URL", "localhost")
        return {
            "seed_config": seed_config,
            "compose_output_path": os.path.join(build_dir, "docker-compose.generated.yaml"),
            "base_compose_path": os.path.join(build_dir, "docker-compose.yaml"),
            "agent_base_url": f"{base_url}:{port}",
            "agent_local_registry_url": f"http://host.docker.internal:{port}",
        }

    async def _auto_start_infra(self, build_dir: str) -> None:
        """Start infra services and wait for Postgres to be ready.

        Core infra (postgres, valkey) is started first and gated on healthchecks.
        Observability infra (jaeger, prometheus, grafana) is then brought up
        best-effort — failures there must not block the registry from reaching
        Postgres and loading agents.
        """
        from oai_agent_registry.services.infra_manager import InfraManager

        seed = self._build_seed_configs_from_agents()
        early_deployer = None
        try:
            early_deployer = DeployerFactory.get_deployer(
                mode="docker",
                **self._deployer_factory_kwargs(build_dir, seed.get("docker", {})),
            )
            early_deployer.start_infra_services()
        except Exception as exc:
            logger.error("auto_start_infra: docker compose startup failed: %s", exc)

        # Best-effort: bring up observability stack (Jaeger / Prometheus / Grafana).
        if early_deployer is not None:
            try:
                early_deployer.start_infra_services(["jaeger", "prometheus", "grafana"])
            except Exception as exc:
                logger.warning(
                    "auto_start_infra: observability services failed to start: %s", exc
                )

        try:
            await InfraManager.wait_for_postgres(
                timeout=self.registry_config.infra_startup_timeout,
            )
        except TimeoutError as exc:
            logger.error("auto_start_infra: Postgres did not become ready: %s", exc)
        except Exception as exc:
            logger.warning("auto_start_infra: Postgres readiness check failed: %s", exc)

    async def initialize(self):
        """Initialize the registry: HTTP client, infra, database, deployers, status checks."""
        self.client = httpx.AsyncClient()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(os.path.join(current_dir, "..", "resources", "docker"))

        # Optional: start infra services (postgres, valkey) before DB init.
        # Sequence: generate compose → docker compose up -d postgres valkey → TCP-poll postgres.
        if self.registry_config.auto_start_infra:
            await self._auto_start_infra(build_dir)

        await self.db_logger.initialize()

        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            await self._load_agents_from_db()
            await self._sync_agents_to_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")

        seed_configs = self._build_seed_configs_from_agents()

        for mode in ("docker", "python_package"):
            try:
                self.deployers[mode] = DeployerFactory.get_deployer(
                    mode=mode,
                    **self._deployer_factory_kwargs(build_dir, seed_configs.get(mode, {})),
                )
                logger.info(
                    "Deployer '%s' initialized with %d seed agents.",
                    mode, len(seed_configs.get(mode, {})),
                )
                await self.deployers[mode].initialize()
            except NotImplementedError as exc:
                logger.debug("Deployer '%s' not initialized: %s", mode, exc)
            except Exception as exc:
                logger.error("Failed to initialize deployer '%s': %s", mode, exc)

        # Status check runs last so it validates both config and restored dynamic agents.
        await self._check_agent_statuses()

    async def shutdown(self):
        """Shut down the registry: close HTTP client, deployers, and the database."""
        if self.client:
            await self.client.aclose()

        for agent, agent_config in self.agents.items():
            if agent_config.registered_via != "dynamic":
                deployer = self._get_deployer(agent_config.deployment_mode)
                if deployer:
                    deployer.remove_agent(agent)

        for mode, deployer in self.deployers.items():
            logger.info("Shutting down deployer '%s'...", mode)
            await deployer.shutdown()

        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    # ── DB sync / restore ─────────────────────────────────────────────────

    async def _sync_agents_to_db(self):
        """Synchronize config-declared agents to the database with registered_via='config'."""
        if not self.db_logger.is_active:
            logger.warning("Database logger inactive — skipping agent sync to DB.")
            return

        for agent_name, agent_config in self.agents.items():
            ok = await self._persist_agent_to_db(agent_name, agent_config, registered_via="config")
            if ok:
                logger.debug("Synced config agent '%s' to DB.", agent_name)
            else:
                logger.error("Failed to sync agent '%s' to DB from config.", agent_name)

    async def _load_agents_from_db(self):
        """Restore dynamic agents from the DB that were active before the registry restarted."""
        logger.info("Restoring active dynamic agents from database...")
        try:
            agents = await self.db_logger.get_all_agents()
        except Exception as exc:
            logger.error("Failed to load dynamic agents from DB: %s", exc)
            return

        restored = 0
        for row in agents:
            agent_name = row.get("agent_name")
            if not agent_name:
                continue
            try:
                self.agents[agent_name] = AgentConfig(
                    name=agent_name,
                    endpoint=row.get("endpoint_url", ""),
                    port=row.get("port"),
                    source=row.get("source"),
                    # Start disabled — _check_agent_statuses() enables if reachable.
                    enabled=False,
                    framework=row.get("framework"),
                    prompts=row.get("prompts", []),
                    tags=row.get("tags", []),
                    current_version=row.get("current_version"),
                    available_versions=row.get("available_versions", []),
                    registered_via=row.get("registered_via", "dynamic"),
                    deployment_mode=row.get("deployment_mode", "docker"),
                    env_vars=row.get("env_vars") or {},
                    sensitive_vars=row.get("sensitive_vars") or [],
                )
                restored += 1
                logger.debug("Restored agent '%s' from DB (pending status check).", agent_name)
            except Exception as exc:
                logger.error("Failed to restore agent '%s' from DB: %s", agent_name, exc)

        logger.info("Dynamic agent restore complete: %d restored.", restored)

    async def _check_agent_statuses(self):
        """
        Ping each agent's /status endpoint and set enabled accordingly.
        Agents that fail the check are marked inactive in the DB.
        """
        logger.info("Checking agent statuses...")
        if not self.client:
            logger.error("HTTP client not initialized — cannot check agent statuses.")
            return

        for agent_name, agent_config in self.agents.items():
            status_url = f"{agent_config.endpoint}/status"
            try:
                response = await self.client.get(status_url, timeout=agent_config.timeout)
                if response.status_code == 200:
                    agent_config.enabled = True
                    logger.info("Agent '%s' at %s is active.", agent_name, agent_config.endpoint)
                else:
                    agent_config.enabled = False
                    logger.warning(
                        "Agent '%s' returned status %d — marking inactive.",
                        agent_name, response.status_code,
                    )
                    await self._mark_agent_inactive_in_db(agent_name)
            except httpx.RequestError as exc:
                agent_config.enabled = False
                logger.warning(
                    "Could not reach agent '%s' at %s (%s) — marking inactive.",
                    agent_name, agent_config.endpoint, exc,
                )
                await self._mark_agent_inactive_in_db(agent_name)
            except Exception as exc:
                agent_config.enabled = False
                logger.error("Error checking status for '%s': %s — marking inactive.", agent_name, exc)
                await self._mark_agent_inactive_in_db(agent_name)

    async def _mark_agent_inactive_in_db(self, agent_name: str) -> None:
        """
        Mark an agent inactive in the DB after a failed status check.
        Errors are logged but never raised so a single DB failure doesn't
        block the startup status sweep.
        """
        try:
            await self.db_logger.deregister_agent(agent_name=agent_name)
        except Exception as exc:
            logger.error(
                "Failed to mark agent '%s' inactive in DB after status check: %s",
                agent_name, exc,
            )

    # ── Merge helper ──────────────────────────────────────────────────────

    async def _get_merged_agent_values(
        self,
        agent_name: str,
        agent_registration: AgentRegistration,
        registered_via: str,
    ) -> Dict[str, Any]:
        """
        Merge new registration values with existing DB values.
        A None value in the incoming registration keeps the existing DB value.
        """
        existing = (
            await self.db_logger.get_agent_details(agent_name)
            if self.db_logger.is_active
            else None
        ) or {}

        def _pick(new_val, db_key, default=None):
            return new_val if new_val is not None else existing.get(db_key, default)

        reg = agent_registration
        # Preserve existing available_versions when the new list is empty/None.
        avail = reg.available_versions
        merged_avail = avail if avail else existing.get("available_versions", [])

        merged = {
            "endpoint":           _pick(reg.endpoint,        "endpoint_url",    ""),
            "port":               _pick(reg.port,            "port"),
            "source":             _pick(reg.source,          "source",          ""),
            "active":             True,
            "registered_via":     registered_via or existing.get("registered_via", "dynamic"),
            "framework":          _pick(reg.framework,       "framework"),
            "prompts":            _pick(reg.prompts,         "prompts",         []),
            "tags":               _pick(reg.tags,            "tags",            []),
            "description":        _pick(reg.description,     "description",     ""),
            "current_version":    _pick(reg.current_version, "current_version"),
            "available_versions": merged_avail,
            "deployment_mode":    _pick(reg.deployment_mode, "deployment_mode", "docker"),
            "env_vars":           _pick(reg.env_vars,        "env_vars",        {}),
            "sensitive_vars":     _pick(reg.sensitive_vars,  "sensitive_vars",  []),
        }

        logger.debug("Merged values for agent '%s': %s", agent_name, merged)
        return merged

    # ── Discovery ─────────────────────────────────────────────────────────

    async def discover_agents(self):
        """Discover agents by scanning the configured port range."""
        start = self.registry_config.start_port
        end = self.registry_config.end_port
        host = self.registry_config.host or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        logger.info("Auto-discovery in port range %d-%d on %s...", start, end, host)

        if not self.client:
            logger.error("HTTP client not initialized — cannot run agent discovery.")
            return

        for port in range(start, end + 1):
            endpoint = f"http://{host.replace('http://', '')}:{port}"
            try:
                response = await self.client.get(f"{endpoint}/info", timeout=1.0)
                if response.status_code != 200:
                    continue
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
                    logger.info("Discovered agent '%s' at %s.", agent_name, endpoint)
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
                        deployment_mode=agent_info.get("deployment_mode", "docker"),
                    )
            except (httpx.RequestError, json.JSONDecodeError):
                pass

    # ── Info / health ─────────────────────────────────────────────────────

    async def get_info(self) -> JSONResponse:
        """Return registry summary and per-agent status."""
        enabled_agents = {name for name, a in self.agents.items() if a.enabled}
        endpoint = (
            self.private_ip
            if os.environ.get("USE_PRIVATE_IP", "false").lower() == "true"
            else f"{os.environ.get('AGENT_BASE_URL')}:{os.environ.get('AGENT_BASE_URL_PORT', self.registry_config.port)}"
        )
        info = {
            "registry": {
                "uptime_seconds": time.time() - self.start_time,
                "total_agents": len(self.agents),
                "enabled_agents": len(enabled_agents),
            },
            "agents": {
                name: {
                    "endpoint": f"{endpoint}/{name}",
                    "enabled": agent.enabled,
                    "status": "active" if agent.enabled else "inactive",
                    "description": agent.description,
                    "framework": agent.framework,
                    "registered_via": agent.registered_via,
                    "current_version": agent.current_version,
                    "available_versions": agent.available_versions or [],
                    "deployment_mode": agent.deployment_mode,
                    "env_vars": _mask_sensitive_env(
                        agent.env_vars or {},
                        agent.sensitive_vars or [],
                    ),
                    "sensitive_vars": agent.sensitive_vars or [],
                    "available_actions": [
                        "start" if not agent.enabled else "stop",
                        "restart", "rebuild", "redeploy",
                        "update", "upgrade", "downgrade", "delete",
                    ],
                }
                for name, agent in self.agents.items()
                if agent.endpoint is not None
            },
        }
        return JSONResponse(info)

    async def health_check(self) -> JSONResponse:
        """Ping each enabled agent and return aggregated health status."""
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
            except Exception as exc:
                health_status[agent_name] = {"status": "unreachable", "error": str(exc)}
                all_healthy = False

        return JSONResponse({
            "registry_status": "healthy",
            "all_agents_healthy": all_healthy,
            "agents": health_status,
            "timestamp": datetime.now().isoformat(),
        })

    async def reload_config(self) -> JSONResponse:
        """Reload configuration from file and re-sync agents."""
        try:
            self.load_config()
            await self._sync_agents_to_db()
            await self._check_agent_statuses()
            return JSONResponse({"message": "Configuration reloaded", "agents": list(self.agents.keys())})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Registration ──────────────────────────────────────────────────────

    async def register_agent(
        self,
        agent_registration: AgentRegistration,
        stream_output: bool = False,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Register (or re-register) an agent dynamically."""
        agent_name = agent_registration.name
        if agent_name in self.agents:
            logger.info("Agent '%s' already registered — updating configuration.", agent_name)

        registered_via = agent_registration.registered_via or "dynamic"
        assigned_port = agent_registration.port

        if agent_registration.deployment_mode == "unknown":
            agent_registration.deployment_mode = (
                self.agents[agent_name].deployment_mode
                if agent_name in self.agents
                else "docker"
            )

        deployer = self._get_deployer(agent_registration.deployment_mode)

        if registered_via == "registry" and deployer:
            if not assigned_port:
                assigned_port = deployer.find_available_port()
                agent_registration.port = assigned_port

            if stream_output:
                return _sse(
                    self._stream_register(
                        agent_name, agent_registration, assigned_port, registered_via, deployer
                    )
                )

            # Fire-and-forget deploy; persist immediately so the caller gets a fast response.
            asyncio.create_task(
                deployer.deploy_agent(
                    agent_name=agent_name,
                    source_url=agent_registration.source,
                    framework=agent_registration.framework,
                    env=agent_registration.env_vars or {},
                    description=agent_registration.description or "",
                    tags=agent_registration.tags or [],
                    port=assigned_port,
                    current_version=agent_registration.current_version,
                    refresh_repo=False,
                )
            )

        db_values = await self._get_merged_agent_values(agent_name, agent_registration, registered_via)
        if assigned_port:
            db_values["port"] = assigned_port

        agent_config = AgentConfig(**db_values)
        self.agents[agent_name] = agent_config
        logger.info("Registered agent '%s' at %s.", agent_name, agent_config.endpoint)
        await self._persist_agent_to_db(agent_name, agent_config)

        return JSONResponse({"message": f"Agent '{agent_name}' registered successfully."})

    async def _stream_register(
        self,
        agent_name: str,
        agent_registration: AgentRegistration,
        assigned_port: Optional[int],
        registered_via: str,
        deployer: BaseDeployer,
    ):
        """SSE generator: deploy agent and persist on success."""
        try:
            yield f"data: Starting registration and deployment for '{agent_name}'...\n\n"
            async for line in deployer.stream_deploy_agent(
                agent_name=agent_name,
                source_url=agent_registration.source,
                framework=agent_registration.framework,
                env=agent_registration.env_vars or {},
                description=agent_registration.description or "",
                tags=agent_registration.tags or [],
                port=assigned_port,
                current_version=agent_registration.current_version,
                refresh_repo=False,
            ):
                yield f"data: {line.strip()}\n\n"

            db_values = await self._get_merged_agent_values(
                agent_name, agent_registration, registered_via
            )
            if assigned_port:
                db_values["port"] = assigned_port

            agent_config = AgentConfig(**db_values)
            self.agents[agent_name] = agent_config
            logger.info("Registered agent '%s' at %s.", agent_name, agent_config.endpoint)
            await self._persist_agent_to_db(agent_name, agent_config)
            yield f"data: ✅ Agent '{agent_name}' registered successfully.\n\n"
        except Exception as exc:
            yield f"data: ❌ Error during registration: {exc}\n\n"
            raise

    async def deregister_agent(self, agent_deregistration: AgentDeregistration) -> JSONResponse:
        """Set an agent's active flag to False in DB and disable it in memory."""
        agent_name = agent_deregistration.name

        if agent_name not in self.agents:
            logger.warning("Attempted to deregister unknown agent '%s'.", agent_name)
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        self.agents[agent_name].enabled = False
        logger.info("Deactivating agent '%s'.", agent_name)
        await self.db_logger.deregister_agent(agent_name=agent_name)

        return JSONResponse({"message": f"Agent '{agent_name}' deactivated successfully."})

    # ── Lifecycle actions ─────────────────────────────────────────────────

    async def execute_lifecycle_action(
        self,
        agent_name: str,
        action: str,
        version: Optional[str] = None,
        stream_output: bool = False,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Dispatch a lifecycle action to the appropriate handler."""
        if agent_name not in self.agents:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        agent_config = self.agents[agent_name]
        deployer = self._get_deployer(agent_config.deployment_mode)
        logger.info("Lifecycle action '%s' for agent '%s'.", action, agent_name)

        try:
            if action in ("update", "upgrade", "downgrade"):
                return await self._action_version_switch(
                    agent_name, agent_config, deployer, version, stream_output
                )
            if action == "stop":
                return await self._action_stop(agent_name, agent_config, deployer)
            if action == "start":
                return await self._action_start(agent_name, agent_config, deployer)
            if action == "restart":
                return await self._action_restart(agent_name, agent_config, deployer, stream_output)
            if action == "delete":
                return await self._action_delete(agent_name, agent_config, deployer)
            if action == "rebuild":
                return await self._action_rebuild(agent_name, agent_config, deployer, stream_output)
            if action == "redeploy":
                return await self._action_redeploy(agent_name, agent_config, deployer, stream_output)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown action '{action}'. "
                    "Supported: start, stop, restart, rebuild, redeploy, update, upgrade, downgrade"
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Lifecycle action '%s' for '%s' failed: %s", action, agent_name, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # -- stop ---------------------------------------------------------------

    async def _action_stop(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
    ) -> JSONResponse:
        if deployer:
            deployer.stop_agent(agent_name)
        self.agents[agent_name].enabled = False
        await self.db_logger.log_agent_action(agent_name, "stop", None)
        return self._lifecycle_json(agent_name, "stop")

    # -- start --------------------------------------------------------------

    async def _action_start(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
    ) -> JSONResponse:
        if deployer:
            deployer.start_agent(agent_name)
        self.agents[agent_name].enabled = True
        await self.db_logger.log_agent_action(agent_name, "start", None)
        return self._lifecycle_json(agent_name, "start")

    # -- delete -------------------------------------------------------------

    async def _action_delete(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
    ) -> JSONResponse:
        if deployer:
            deployer.remove_agent(agent_name)
        self.agents.pop(agent_name, None)
        logger.info("Agent '%s' removed.", agent_name)
        await self.db_logger.delete_agent(agent_name)
        return JSONResponse({
            "message": f"Lifecycle action 'delete' executed for agent '{agent_name}'.",
            "agent": agent_name,
            "action": "delete",
            "status": "completed",
        })

    # -- restart ------------------------------------------------------------

    async def _action_restart(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
        stream_output: bool,
    ) -> Union[JSONResponse, StreamingResponse]:
        if not deployer:
            await self.db_logger.log_agent_action(agent_name, "restart", None)
            return self._lifecycle_json(agent_name, "restart")
        if stream_output:
            return _sse(self._stream_restart(agent_name, agent_config, deployer))
        # no_build=True: rewrites the compose file with current config then does
        # `docker compose up -d --no-deps` — avoids the _dynamic_services.pop bug.
        await deployer.deploy_agent(
            **self._agent_deploy_kwargs(agent_name, agent_config, no_build=True)
        )
        self.agents[agent_name].enabled = True
        await self.db_logger.log_agent_action(agent_name, "restart", None)
        return self._lifecycle_json(agent_name, "restart")

    async def _stream_restart(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: BaseDeployer,
    ):
        try:
            yield f"data: Restarting '{agent_name}'...\n\n"
            async for line in deployer.stream_deploy_agent(
                **self._agent_deploy_kwargs(agent_name, agent_config, no_build=True)
            ):
                yield f"data: {line.strip()}\n\n"
            self.agents[agent_name].enabled = True
            await self.db_logger.log_agent_action(agent_name, "restart", None)
            yield f"data: ✓ Agent '{agent_name}' restarted.\n\n"
        except Exception as exc:
            yield f"data: Error during restart: {exc}\n\n"
            raise

    # -- version switch (update / upgrade / downgrade) ----------------------

    async def _action_version_switch(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
        version: Optional[str],
        stream_output: bool,
    ) -> Union[JSONResponse, StreamingResponse]:
        if not version:
            raise HTTPException(
                status_code=400,
                detail="Version is required for update/upgrade/downgrade.",
            )

        # Determine whether to skip the image build.
        no_build = bool(deployer and deployer.image_exists(agent_name, version))
        if no_build:
            logger.info("Artifact '%s:%s' found locally — skipping build.", agent_name, version)
        else:
            logger.info("Artifact '%s:%s' not found — will build.", agent_name, version)

        # Update in-memory version before deploy so compose reflects the new tag.
        agent_config.current_version = version
        if agent_config.available_versions is None:
            agent_config.available_versions = []
        if version not in agent_config.available_versions:
            agent_config.available_versions.append(version)

        # Persist the version change immediately so the DB stays consistent.
        await self._persist_agent_to_db(agent_name, agent_config)

        if not deployer:
            await self.db_logger.log_agent_action(agent_name, "update", version)
            return self._lifecycle_json(agent_name, "update")

        if stream_output:
            return _sse(
                self._stream_version_switch(agent_name, agent_config, deployer, version, no_build)
            )

        await deployer.deploy_agent(
            **self._agent_deploy_kwargs(
                agent_name, agent_config,
                refresh_repo=not no_build,
                no_build=no_build,
            )
        )
        self.agents[agent_name].enabled = True
        await self.db_logger.log_agent_action(agent_name, "update", version)
        return self._lifecycle_json(agent_name, "update")

    async def _stream_version_switch(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: BaseDeployer,
        version: str,
        no_build: bool,
    ):
        try:
            if no_build:
                yield f"data: Artifact '{agent_name}:{version}' found locally — reusing without rebuild.\n\n"
            else:
                yield f"data: Artifact '{agent_name}:{version}' not found — building now.\n\n"
            yield f"data: Agent '{agent_name}' switching to version {version}...\n\n"
            async for line in deployer.stream_deploy_agent(
                **self._agent_deploy_kwargs(
                    agent_name, agent_config,
                    refresh_repo=not no_build,
                    no_build=no_build,
                )
            ):
                yield f"data: {line.strip()}\n\n"
            self.agents[agent_name].enabled = True
            await self.db_logger.log_agent_action(agent_name, "update", version)
            yield f"data: Agent '{agent_name}' successfully switched to {version}!\n\n"
        except Exception as exc:
            yield f"data: Error during version switch: {exc}\n\n"
            raise

    # -- rebuild ------------------------------------------------------------

    async def _action_rebuild(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
        stream_output: bool,
    ) -> Union[JSONResponse, StreamingResponse]:
        if not deployer:
            await self.db_logger.log_agent_action(agent_name, "rebuild", None)
            return self._lifecycle_json(agent_name, "rebuild")
        if stream_output:
            return _sse(self._stream_rebuild(agent_name, agent_config, deployer))
        deployer.remove_agent(agent_name)
        self.agents[agent_name].enabled = False
        logger.info("Agent '%s' stopped. Rebuilding image...", agent_name)
        await deployer.deploy_agent(
            **self._agent_deploy_kwargs(agent_name, agent_config, refresh_repo=True)
        )
        self.agents[agent_name].enabled = True
        await self.db_logger.log_agent_action(agent_name, "rebuild", None)
        return self._lifecycle_json(agent_name, "rebuild")

    async def _stream_rebuild(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: BaseDeployer,
    ):
        try:
            deployer.remove_agent(agent_name)
            self.agents[agent_name].enabled = False
            yield f"data: Agent '{agent_name}' stopped. Rebuilding image...\n\n"
            async for line in deployer.stream_deploy_agent(
                **self._agent_deploy_kwargs(agent_name, agent_config, refresh_repo=True)
            ):
                yield f"data: {line.strip()}\n\n"
            self.agents[agent_name].enabled = True
            await self.db_logger.log_agent_action(agent_name, "rebuild", None)
            yield f"data: ✓ Agent '{agent_name}' rebuild completed successfully!\n\n"
        except Exception as exc:
            yield f"data: Error during rebuild: {exc}\n\n"
            raise

    # -- redeploy -----------------------------------------------------------

    async def _action_redeploy(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
        stream_output: bool,
    ) -> Union[JSONResponse, StreamingResponse]:
        if not deployer:
            await self.db_logger.log_agent_action(agent_name, "redeploy", None)
            return self._lifecycle_json(agent_name, "redeploy", status="initiated")
        if stream_output:
            return _sse(self._stream_redeploy(agent_name, agent_config, deployer))
        asyncio.create_task(
            deployer.deploy_agent(
                **self._agent_deploy_kwargs(agent_name, agent_config, refresh_repo=True)
            )
        )
        await self.db_logger.log_agent_action(agent_name, "redeploy", None)
        return self._lifecycle_json(agent_name, "redeploy", status="initiated")

    async def _stream_redeploy(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: BaseDeployer,
    ):
        try:
            yield f"data: Starting async redeploy for agent '{agent_name}'...\n\n"
            asyncio.create_task(
                deployer.deploy_agent(
                    **self._agent_deploy_kwargs(agent_name, agent_config, refresh_repo=True)
                )
            )
            await self.db_logger.log_agent_action(agent_name, "redeploy", None)
            yield f"data: ✅ Redeploy task initiated for '{agent_name}'.\n\n"
            yield "data: This is asynchronous — the agent will be updated in the background.\n\n"
        except Exception as exc:
            yield f"data: Error initiating redeploy: {exc}\n\n"
            raise

    # ── Env-var update ────────────────────────────────────────────────────

    async def update_server_env_vars(
        self,
        agent_name: str,
        env_vars: Dict[str, str],
        sensitive_vars: Optional[list] = None,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Update environment variables for an agent then stream a container restart."""
        if agent_name not in self.agents:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

        agent_config = self.agents[agent_name]
        agent_config.env_vars = env_vars
        agent_config.sensitive_vars = sensitive_vars or []

        db_saved = await self._persist_agent_to_db(agent_name, agent_config)
        logger.info("Updated env_vars for agent '%s' (%d vars).", agent_name, len(env_vars))

        deployer = self._get_deployer(agent_config.deployment_mode)
        return _sse(
            self._stream_env_update(agent_name, agent_config, deployer, db_saved, len(env_vars))
        )

    async def _stream_env_update(
        self,
        agent_name: str,
        agent_config: AgentConfig,
        deployer: Optional[BaseDeployer],
        db_saved: bool,
        var_count: int,
    ):
        """SSE generator: report save result then redeploy with updated env vars."""
        saved_label = f"{var_count} variable(s)" + ("" if db_saved else " (DB save failed)")
        yield f"data: Saved {saved_label} for '{agent_name}'.\n\n"

        if not deployer:
            yield "data: No deployer configured — env vars updated in registry memory only.\n\n"
            yield "data: Restart the agent manually to apply changes.\n\n"
            return

        try:
            yield f"data: Redeploying '{agent_name}' with updated environment variables...\n\n"
            # no_build=True: re-registers the service in _dynamic_services so
            # docker-compose.generated.yaml is rewritten with the new env block,
            # then restarts the container without rebuilding the image.
            async for line in deployer.stream_deploy_agent(
                **self._agent_deploy_kwargs(agent_name, agent_config, no_build=True)
            ):
                yield f"data: {line.strip()}\n\n"
            self.agents[agent_name].enabled = True
            await self.db_logger.log_agent_action(agent_name, "env-update", agent_config.current_version)
            yield f"data: ✓ Agent '{agent_name}' redeployed with updated environment variables.\n\n"
        except Exception as exc:
            yield f"data: ✗ Error redeploying agent: {exc}\n\n"

    # ── Proxy ─────────────────────────────────────────────────────────────

    async def proxy_request(self, agent_name: str, path: str, request: Request) -> Response:
        """Proxy a request to the specified agent."""
        _validate_token(request, agent_name)

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

        # ── Env var injection ──────────────────────────────────────────────
        # 1. Normalise stored env var keys to UPPER_CASE and expand ${VAR} placeholders.
        env_to_inject: dict = {
            k.upper(): os.path.expandvars(str(v)) if isinstance(v, str) else str(v)
            for k, v in (agent_config.env_vars or {}).items()
        }
        # 2. Apply per-request overrides (X-Override-Env-{NAME}) — always win over stored values.
        keys_to_remove = [k for k in headers if k.lower().startswith("x-override-env-")]
        for key in keys_to_remove:
            var_name = key[len("x-override-env-"):].upper()
            env_to_inject[var_name] = headers.pop(key)
        # 3. Forward the merged set as plain {NAME} headers to the agent.
        for var_name, value in env_to_inject.items():
            headers[var_name] = str(value)
        # ──────────────────────────────────────────────────────────────────

        body = await request.body()

        accept_header = request.headers.get("accept", "").lower()
        is_streaming = (
            "stream" in path.lower()
            or "stream" in request.url.query.lower()
            or "sendmessagestream" in path.lower()
            or "event-stream" in accept_header
            or "x-ndjson" in accept_header
        )

        # Also detect streaming from JSON-RPC method name.
        if not is_streaming and body and "application/json" in request.headers.get("content-type", "").lower():
            try:
                body_json = json.loads(body)
                if isinstance(body_json, dict) and "stream" in body_json.get("method", "").lower():
                    is_streaming = True
                    logger.info("[STREAMING] Detected JSON-RPC streaming method: %s", body_json["method"])
            except (json.JSONDecodeError, Exception):
                pass

        if is_streaming:
            logger.info(
                "[STREAMING] Detected streaming request: %s %s (Accept: %s)",
                request.method, path, accept_header,
            )
            return await self._proxy_streaming_request(
                target_url, request.method, headers, body, agent_config.timeout
            )

        return await self._proxy_regular_request(
            agent_name, target_url, request.method, headers, body, agent_config.timeout, path
        )

    async def _proxy_regular_request(
        self,
        agent_name: str,
        url: str,
        method: str,
        headers: dict,
        body: bytes,
        timeout: int,
        path: str,
    ) -> Response:
        try:
            response = await self.client.request(
                method, url, headers=headers, content=body, timeout=timeout
            )
            response_headers = self._clean_response_headers(response.headers)
            content = response.content

            if path and path.rstrip("/") in ("docs", "redoc") and "text/html" in response.headers.get("content-type", ""):
                try:
                    text = response.text
                    text = text.replace('"/openapi.json"', f'"/{agent_name}/openapi.json"')
                    text = text.replace("'/openapi.json'", f"'/{agent_name}/openapi.json'")
                    content = text.encode("utf-8")
                    response_headers.pop("content-length", None)
                except Exception as exc:
                    logger.warning("Failed to rewrite docs for %s: %s", agent_name, exc)

            elif path and path.rstrip("/") == "openapi.json" and "application/json" in response.headers.get("content-type", ""):
                try:
                    data = response.json()
                    if "paths" in data:
                        data["paths"] = {
                            (f"/{agent_name}{p}" if not p.startswith(f"/{agent_name}") else p): methods
                            for p, methods in data["paths"].items()
                        }
                    content = json.dumps(data).encode("utf-8")
                    response_headers.pop("content-length", None)
                except Exception as exc:
                    logger.warning("Failed to rewrite openapi.json for %s: %s", agent_name, exc)

            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type"),
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Request to agent '{agent_name}' timed out.")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Proxy error: {exc}")

    @staticmethod
    async def _proxy_streaming_request(
        url: str,
        method: str,
        headers: dict,
        body: bytes,
        timeout: int,
    ) -> StreamingResponse:
        logger.info("[STREAM PROXY] Starting stream to %s", url)

        async def stream_generator():
            chunk_count = 0
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                ) as stream_client:
                    async with stream_client.stream(method, url, headers=headers, content=body) as response:
                        logger.info(
                            "[STREAM PROXY] Got response: %d, Content-Type: %s",
                            response.status_code,
                            response.headers.get("content-type"),
                        )
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                chunk_count += 1
                                logger.debug(
                                    "[STREAM PROXY] Yielding chunk %d (%d bytes)",
                                    chunk_count, len(chunk),
                                )
                                yield chunk
                        logger.info("[STREAM PROXY] Stream completed with %d chunks.", chunk_count)
            except Exception as exc:
                logger.error(
                    "[STREAM PROXY] Streaming error after %d chunks: %s", chunk_count, exc,
                    exc_info=True,
                )
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @staticmethod
    def _clean_response_headers(headers: httpx.Headers) -> dict:
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        return {k: v for k, v in headers.items() if k.lower() not in excluded}

    # ── Seed configs ──────────────────────────────────────────────────────

    def _build_seed_configs_from_agents(self) -> dict:
        """Group config agents by deployment mode for factory initialization."""
        seeds: dict = {}
        for agent_name, agent_config in self.agents.items():
            mode = agent_config.deployment_mode or "docker"
            seeds.setdefault(mode, {})[agent_name] = {
                "port": agent_config.port,
                "source": agent_config.source or "",
                "framework": agent_config.framework or "",
                "tags": [agent_config.framework.lower()] if agent_config.framework else [],
                "env": agent_config.env_vars or {},
                "description": "",
                "current_version": agent_config.current_version,
            }
        return seeds
