import json
import logging
import os
import asyncio
from typing import Any, Dict, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastmcp import FastMCP

from oai_mcp_registry.models import AppConfig, ServerConfig, RegistryConfig, ServerRegistration, ServerDeregistration
from oai_platform_core.networking import get_local_ip, get_public_ip
from oai_mcp_registry.services.db.database_logger import RegistryDatabaseLogger
from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.services.deployers.factory import DeployerFactory

logger = logging.getLogger("MCPRegistry")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(gen) -> StreamingResponse:
    """Wraps an async generator as a Server-Sent Events response."""
    return StreamingResponse(gen, media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MCPRegistry:

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv("MCP_CONFIG_PATH", "./config/proxy_config.json")
        self.host_ip = get_local_ip()
        self.public_ip = get_public_ip()
        self.sub_apps: Dict[str, FastAPI] = {}
        self.config: Optional[AppConfig] = None
        self.registry_config: RegistryConfig = RegistryConfig()
        self.db_logger: RegistryDatabaseLogger = RegistryDatabaseLogger(logger=logger)
        self.deployers: Dict[str, BaseDeployer] = {}
        # Injected by app.py lifespan so sub-app tasks stay in the same asyncio task.
        self.start_sub_app = None
        self.stop_sub_app = None

        try:
            self.load_configuration()
        except Exception as e:
            logger.critical("Failed to load configuration: %s", e)
            self.config = AppConfig(servers={}, registry=self.registry_config)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_configuration(self):
        if not os.path.exists(self.config_path):
            self.config = AppConfig(servers={}, registry=self.registry_config)
            return

        try:
            with open(self.config_path) as f:
                self.config = AppConfig(**json.load(f))
                self.registry_config = self.config.registry
            for server in self.config.servers.values():
                server.registered_via = "config"
            logger.info("Loaded configuration from %s", self.config_path)
        except Exception as e:
            raise Exception(f"Configuration invalid: {e}") from e

    def _build_seed_configs_from_servers(self) -> dict:
        """Groups config servers by deployment mode for deployer factory initialisation."""
        seeds: dict = {}
        for name, cfg in self.config.servers.items():
            mode = getattr(cfg, "deployment_mode", "docker") or "docker"
            seeds.setdefault(mode, {})[name] = {
                "port":            cfg.port,
                "source":          cfg.source or "",
                "framework":       getattr(cfg, "framework", ""),
                "tags":            cfg.tags or [],
                "env":             cfg.env_vars,
                "description":     cfg.description or "",
                "current_version": cfg.current_version,
            }
        return seeds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Starts infra (optional), DB, deployers, and proxy sub-apps."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir   = os.path.abspath(os.path.join(current_dir, "..", "resources", "docker"))

        compose_output = os.path.join(build_dir, "docker-compose.generated.yaml")
        base_compose   = os.path.join(build_dir, "docker-compose.yaml")
        base_url       = (f"{os.environ.get('MCP_BASE_URL', 'localhost')}"
                          f":{os.environ.get('MCP_BASE_URL_PORT', self.registry_config.port)}")
        local_reg_url  = f"http://host.docker.internal:{os.environ.get('MCP_BASE_URL_PORT', self.registry_config.port)}"

        # --- Optional: bring up Postgres + Valkey before DB init ---
        # Sequence: early deployer → start_infra_services() → TCP-poll Postgres → init DB.
        if self.registry_config.auto_start_infra:
            await self._auto_start_infra(build_dir, base_url, local_reg_url)

        await self.db_logger.initialize()
        if self.db_logger.is_active:
            logger.info("RegistryDatabaseLogger initialized successfully.")
            await self._sync_servers_to_db()
            await self._load_dynamic_servers_from_db()
        else:
            logger.warning("RegistryDatabaseLogger could not be initialized.")

        # Initialise deployers
        seed_configs = self._build_seed_configs_from_servers()
        for mode in ["docker", "python_package"]:
            try:
                self.deployers[mode] = DeployerFactory.get_deployer(
                    mode=mode,
                    seed_config=seed_configs.get(mode, {}),
                    compose_output_path=compose_output,
                    base_compose_path=base_compose,
                    agent_base_url=base_url,
                    agent_local_registry_url=local_reg_url,
                )
                logger.info("Deployer '%s' initialised with %d seed servers.", mode, len(seed_configs.get(mode, {})))
                await self.deployers[mode].initialize()
            except NotImplementedError as e:
                logger.debug("Deployer '%s' not available: %s", mode, e)
            except Exception as e:
                logger.error("Failed to initialise deployer '%s': %s", mode, e)

        self.initialize_proxies()

    async def _auto_start_infra(self, build_dir: str, base_url: str, local_reg_url: str):
        """Brings up Postgres + Valkey via Docker Compose, then waits for Postgres."""
        from oai_mcp_registry.services.infra_manager import InfraManager

        seed = self._build_seed_configs_from_servers()
        try:
            early = DeployerFactory.get_deployer(
                mode="docker",
                seed_config=seed.get("docker", {}),
                compose_output_path=os.path.join(build_dir, "docker-compose.generated.yaml"),
                base_compose_path=os.path.join(build_dir, "docker-compose.yaml"),
                agent_base_url=base_url,
                agent_local_registry_url=local_reg_url,
            )
            early.start_infra_services()
        except Exception as e:
            logger.error("auto_start_infra: docker compose startup failed: %s", e)

        # TCP safety-net — required when docker compose --wait is unavailable (< v2.4).
        try:
            await InfraManager.wait_for_postgres(timeout=self.registry_config.infra_startup_timeout)
        except TimeoutError as e:
            logger.error("auto_start_infra: Postgres did not become ready: %s", e)
        except Exception as e:
            logger.warning("auto_start_infra: Postgres readiness check failed: %s", e)

    async def shutdown(self):
        """Stops config-declared containers, shuts down deployers, closes DB."""
        # Dynamic containers are left running intentionally — they outlive the registry process.
        for name, cfg in self.config.servers.items():
            if getattr(cfg, "registered_via", "dynamic") != "dynamic":
                deployer = self._get_deployer(getattr(cfg, "deployment_mode", "docker"))
                if deployer:
                    deployer.remove_server(name)

        for mode, deployer in self.deployers.items():
            logger.info("Shutting down deployer '%s'...", mode)
            await deployer.shutdown()

        await self.db_logger.close()
        logger.info("RegistryDatabaseLogger closed.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_deployer(self, mode: str) -> Optional[BaseDeployer]:
        """Returns the deployer for *mode*, falling back to 'docker'."""
        return self.deployers.get(mode) or self.deployers.get("docker")

    def _get_server_current_version(self, server_name: str) -> Optional[str]:
        """Returns current_version for a server, or None if not found."""
        cfg = self.config.servers.get(server_name)
        return getattr(cfg, "current_version", None) if cfg else None

    def _server_deploy_kwargs(self, server_name: str, server_config: ServerConfig, **overrides) -> Dict[str, Any]:
        """Builds the common keyword arguments for every deployer deploy/stream call."""
        base = {
            "server_name":     server_name,
            "source_url":      server_config.source or "",
            "framework":       getattr(server_config, "framework", None),
            "env":             server_config.env_vars or {},
            "description":     server_config.description or "",
            "tags":            server_config.tags or [],
            "port":            server_config.port,
            "current_version": server_config.current_version,
        }
        base.update(overrides)
        return base

    async def _persist_server_to_db(self, server_name: str, server_config: ServerConfig) -> None:
        """Writes the current in-memory ServerConfig state to the database."""
        await self.db_logger.log_server_registration(
            server_name=server_name,
            endpoint_url=server_config.endpoint or "",
            port=server_config.port,
            description=server_config.description,
            active=server_config.enabled,
            registered_via=server_config.registered_via,
            source=server_config.source,
            tags=server_config.tags or [],
            current_version=server_config.current_version,
            available_versions=server_config.available_versions or [],
            deployment_mode=server_config.deployment_mode or "docker",
            env_vars=server_config.env_vars or None,
            sensitive_vars=server_config.sensitive_vars or None,
        )

    async def _mount_proxy(self, server_name: str, url: str) -> None:
        """Stops any existing sub-app for *server_name*, then mounts a fresh proxy."""
        mcp     = FastMCP.as_proxy(url, name=server_name)
        sub_app = mcp.http_app()
        await self.stop_sub_app(server_name)
        self.sub_apps[server_name] = sub_app
        await self.start_sub_app(server_name, sub_app)

    # ------------------------------------------------------------------
    # DB sync helpers (called during initialise)
    # ------------------------------------------------------------------

    async def _sync_servers_to_db(self):
        """Upserts config-declared servers into the DB.

        If a config server has no env_vars defined, previously-saved env_vars from
        the DB are preserved so that values set through the UI survive a restart.
        """
        if not self.db_logger.is_active:
            logger.warning("DB logger not active — skipping server sync.")
            return

        for server_name, cfg in self.config.servers.items():
            if getattr(cfg, "registered_via", "dynamic") != "config":
                continue

            config_env  = getattr(cfg, "env_vars",       None) or {}
            config_sens = getattr(cfg, "sensitive_vars",  None) or []

            if not config_env:
                # Config file has no env_vars — restore whatever the DB has so we
                # don't erase values the user set through the UI.
                try:
                    existing = await self.db_logger.get_server_details(server_name)
                    if existing:
                        db_env  = existing.get("env_vars")       or {}
                        db_sens = existing.get("sensitive_vars") or []
                        if db_env:
                            cfg.env_vars       = db_env
                            cfg.sensitive_vars = db_sens
                            config_env  = db_env
                            config_sens = db_sens
                            logger.debug("Restored %d env_vars from DB for '%s'.", len(db_env), server_name)
                except Exception as e:
                    logger.warning("Could not read existing env_vars for '%s' from DB: %s", server_name, e)

            try:
                await self.db_logger.log_server_registration(
                    server_name=server_name,
                    endpoint_url=cfg.endpoint or "",
                    port=cfg.port,
                    description=cfg.description,
                    active=True,
                    registered_via="config",
                    source=cfg.source,
                    tags=cfg.tags,
                    current_version=cfg.current_version,
                    available_versions=cfg.available_versions,
                    deployment_mode=cfg.deployment_mode,
                    env_vars=config_env or None,
                    sensitive_vars=config_sens or None,
                )
                logger.debug("Synced config server '%s' to DB.", server_name)
            except Exception as e:
                logger.error("Failed to sync '%s' to DB: %s", server_name, e)

    async def _load_dynamic_servers_from_db(self):
        """Restores dynamic servers that were registered before the last restart."""
        logger.info("Restoring dynamic MCP servers from database...")
        try:
            rows = await self.db_logger.get_all_servers()
        except Exception as e:
            logger.error("Failed to load dynamic servers from DB: %s", e)
            return

        restored = skipped = 0
        for row in rows:
            server_name = row.get("server_name")
            if not server_name:
                continue
            if server_name in self.config.servers:
                logger.debug("Skipping DB restore for '%s' — already in config.", server_name)
                skipped += 1
                continue
            try:
                cfg = ServerConfig(
                    endpoint=row.get("endpoint_url", ""),
                    port=row.get("port"),
                    description=row.get("description", "A proxied MCP server"),
                    source=row.get("source"),
                    tags=row.get("tags", []),
                    current_version=row.get("current_version"),
                    available_versions=row.get("available_versions", []),
                    deployment_mode=row.get("deployment_mode", "docker"),
                    env_vars=row.get("env_vars") or None,
                    sensitive_vars=row.get("sensitive_vars") or [],
                )
                cfg.registered_via = "dynamic"
                self.config.servers[server_name] = cfg
                restored += 1
                logger.debug("Restored dynamic server '%s' from DB.", server_name)
            except Exception as e:
                logger.error("Failed to restore '%s' from DB: %s", server_name, e)

        logger.info("Restore complete: %d restored, %d skipped (config collision).", restored, skipped)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def discover_servers(self, host: str = None):
        """Probes localhost ports for MCP /info endpoints and registers any found."""
        start = self.registry_config.start_port
        end   = self.registry_config.end_port
        # Always probe localhost; the stored endpoint uses the configured host.
        # (If host is a remote machine, discovery must be triggered from that host.)
        host = host or self.registry_config.host or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        logger.info("Auto-discovery: scanning ports %d–%d on host %s...", start, end, host)

        async def _probe(client: httpx.AsyncClient, port: int):
            try:
                resp = await client.get(f"http://localhost:{port}/info", timeout=1.0)
                if resp.status_code != 200:
                    return
                info        = resp.json()
                server_name = info.get("server_name")
                if not server_name or server_name in self.config.servers:
                    return
                cfg = ServerConfig(
                    endpoint=f"http://{host}:{port}",
                    description=info.get("server_config", {}).get("description", "Auto-discovered MCP server"),
                    source=info.get("source"),
                    tags=info.get("tags", []),
                    current_version=info.get("current_version"),
                    available_versions=info.get("available_versions", []),
                    deployment_mode=info.get("deployment_mode", "docker"),
                )
                cfg.registered_via = "dynamic"
                self.config.servers[server_name] = cfg
                logger.info("Discovered '%s' at http://%s:%d", server_name, host, port)
                await self.db_logger.log_server_registration(
                    server_name=server_name,
                    endpoint_url=f"http://{host}:{port}",
                    port=port,
                    description=cfg.description,
                    active=True,
                    registered_via="dynamic",
                    source=cfg.source,
                    tags=cfg.tags,
                    current_version=cfg.current_version,
                    available_versions=cfg.available_versions,
                    deployment_mode=cfg.deployment_mode,
                )
            except (httpx.RequestError, json.JSONDecodeError):
                pass

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[_probe(client, p) for p in range(start, end + 1)])

        self.initialize_proxies()

    # ------------------------------------------------------------------
    # Proxy management
    # ------------------------------------------------------------------

    def _build_upstream_url(self, server_info: ServerConfig) -> str:
        if server_info.endpoint:
            url = server_info.endpoint
        elif server_info.port:
            url = f"http://{self.host_ip}:{server_info.port}/mcp"
        else:
            return ""
        if not (url.endswith("/sse") or url.endswith("/mcp")):
            url = f"{url.rstrip('/')}/mcp"
        return url

    def initialize_proxies(self):
        if not self.config:
            return
        logger.info("Initialising %d upstream proxies...", len(self.config.servers))
        self.sub_apps = {}
        for name, info in self.config.servers.items():
            try:
                url = self._build_upstream_url(info)
                if not url:
                    logger.warning("Skipping '%s': no endpoint or port defined.", name)
                    continue
                logger.info("  [Proxy] %s -> %s", name, url)
                self.sub_apps[name] = FastMCP.as_proxy(url, name=name).http_app()
            except Exception as e:
                logger.error("  [Failed] Could not initialise proxy for '%s': %s", name, e)

    async def reload_config(self) -> JSONResponse:
        """Reloads the config file and re-initialises proxies."""
        try:
            self.load_configuration()
            await self._sync_servers_to_db()
            self.initialize_proxies()
            return JSONResponse({"message": "Configuration reloaded", "servers": list(self.config.servers.keys())})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _get_merged_server_values(
        self, server_name: str, reg: ServerRegistration, registered_via: str,
    ) -> Dict[str, Any]:
        """Merges a new registration request with existing DB values for partial updates."""
        existing = (await self.db_logger.get_server_details(server_name)) if self.db_logger.is_active else {}
        existing = existing or {}

        # Prefer new versions only when the caller actually supplies them.
        new_versions = reg.available_versions if reg.available_versions else None

        # env_vars: overlay new values on top of existing ones.
        merged_env = {**(existing.get("env_vars") or {}), **(reg.env_vars or {})} or None

        # sensitive_vars: union.
        merged_sensitive = list(set(existing.get("sensitive_vars") or []) | set(reg.sensitive_vars or []))

        def _pick(new_val, existing_key, default=None):
            return new_val if new_val is not None else existing.get(existing_key, default)

        return {
            "endpoint":          _pick(reg.endpoint,          "endpoint",          ""),
            "port":              _pick(reg.port,               "port"),
            "source":            _pick(reg.source,             "source",            ""),
            "active":            True,
            "registered_via":    _pick(registered_via,         "registered_via",   "dynamic"),
            "tags":              _pick(reg.tags,               "tags",             []),
            "description":       _pick(reg.description,        "description",       ""),
            "current_version":   _pick(reg.current_version,    "current_version"),
            "available_versions":_pick(new_versions,           "available_versions",[]),
            "deployment_mode":   _pick(reg.deployment_mode,    "deployment_mode",  "docker"),
            "env_vars":          merged_env,
            "sensitive_vars":    merged_sensitive,
        }

    async def register_server(
        self,
        server_registration: ServerRegistration,
        stream_output: bool = False,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Registers (or re-registers) an MCP server dynamically."""
        server_name    = server_registration.name
        registered_via = server_registration.registered_via or "dynamic"

        if server_name in self.config.servers:
            logger.info("Server '%s' already registered — updating.", server_name)

        # Resolve deployment_mode when caller passes 'unknown'.
        if server_registration.deployment_mode == "unknown":
            server_registration.deployment_mode = (
                self.config.servers[server_name].deployment_mode
                if server_name in self.config.servers else "docker"
            )

        deployer      = self._get_deployer(server_registration.deployment_mode)
        assigned_port = server_registration.port

        # --- Registry-managed deployment (build + run the container) ---
        if registered_via == "registry" and deployer:
            if not assigned_port:
                assigned_port = deployer.find_available_port()
                server_registration.port = assigned_port

            if stream_output:
                return _sse(self._stream_register(server_name, server_registration, registered_via, deployer, assigned_port))
            else:
                asyncio.create_task(deployer.deploy_server(
                    server_name=server_name,
                    source_url=server_registration.source,
                    framework=getattr(server_registration, "framework", None),
                    env=server_registration.env_vars or {},
                    description=getattr(server_registration, "description", ""),
                    tags=getattr(server_registration, "tags", []),
                    port=assigned_port,
                    current_version=getattr(server_registration, "current_version", None),
                    refresh_repo=False,
                ))

        # --- Register proxy + persist to DB (both streaming fallthrough and non-registry) ---
        db_values = await self._get_merged_server_values(server_name, server_registration, registered_via)
        if assigned_port:
            db_values["port"] = assigned_port

        server_config = ServerConfig(**db_values)
        self.config.servers[server_name] = server_config

        try:
            url = self._build_upstream_url(server_config)
            if url:
                is_update = server_name in self.sub_apps
                await self._mount_proxy(server_name, url)
                logger.info("%s proxy for '%s' -> %s", "Updated" if is_update else "Registered", server_name, url)
        except Exception as e:
            logger.error("Failed to mount proxy for '%s': %s", server_name, e)

        await self._persist_server_to_db(server_name, server_config)
        return JSONResponse({"message": f"Server '{server_name}' registered successfully."})

    async def _stream_register(
        self,
        server_name: str,
        reg: ServerRegistration,
        registered_via: str,
        deployer,
        assigned_port: int,
    ):
        """Async generator: deploys, mounts proxy, persists to DB, yields SSE lines."""
        try:
            yield f"data: Starting deployment for '{server_name}'...\n\n"
            async for line in deployer.stream_deploy_server(
                server_name=server_name,
                source_url=reg.source,
                framework=getattr(reg, "framework", None),
                env=reg.env_vars or {},
                description=getattr(reg, "description", ""),
                tags=getattr(reg, "tags", []),
                port=assigned_port,
                current_version=getattr(reg, "current_version", None),
                refresh_repo=False,
            ):
                yield f"data: {line.strip()}\n\n"

            db_values = await self._get_merged_server_values(server_name, reg, registered_via)
            if assigned_port:
                db_values["port"] = assigned_port
            server_config = ServerConfig(**db_values)
            self.config.servers[server_name] = server_config

            url = self._build_upstream_url(server_config)
            await self._mount_proxy(server_name, url)
            await self._persist_server_to_db(server_name, server_config)
            yield f"data: ✅ Server '{server_name}' registered successfully.\n\n"
        except Exception as e:
            yield f"data: ❌ Error during registration: {str(e)}\n\n"
            raise

    async def deregister_server(self, server_deregistration: ServerDeregistration) -> JSONResponse:
        """Deactivates a server (stops the container, removes the proxy, marks inactive in DB)."""
        server_name = server_deregistration.name
        if server_name not in self.config.servers:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        server_config = self.config.servers[server_name]
        deployer = self._get_deployer(getattr(server_config, "deployment_mode", "docker"))
        if deployer:
            deployer.stop_server(server_name)

        if self.stop_sub_app:
            await self.stop_sub_app(server_name)
        elif server_name in self.sub_apps:
            del self.sub_apps[server_name]

        server_config.enabled = False
        logger.info("Deactivated server '%s'.", server_name)
        await self.db_logger.deregister_server(server_name=server_name)
        return JSONResponse({"message": f"Server '{server_name}' deactivated successfully."})

    # ------------------------------------------------------------------
    # Env-var update
    # ------------------------------------------------------------------

    async def update_server_env_vars(
        self,
        server_name: str,
        env_vars: Dict[str, str],
        sensitive_vars: Optional[list] = None,
    ) -> StreamingResponse:
        """Saves updated env vars to memory + DB, then redeploys the container to apply them."""
        if server_name not in self.config.servers:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        server_config = self.config.servers[server_name]
        server_config.env_vars       = env_vars
        server_config.sensitive_vars = sensitive_vars or []

        db_saved = False
        try:
            await self._persist_server_to_db(server_name, server_config)
            db_saved = True
        except Exception as e:
            logger.warning("DB update for env_vars of '%s' failed: %s", server_name, e)

        logger.info("Updated env_vars for '%s' (%d vars).", server_name, len(env_vars))

        deployer = self._get_deployer(getattr(server_config, "deployment_mode", "docker"))
        return _sse(self._stream_env_update(server_name, server_config, deployer, db_saved))

    async def _stream_env_update(self, server_name: str, server_config: ServerConfig, deployer, db_saved: bool):
        """Async generator: streams redeployment progress after an env-var save."""
        n = len(server_config.env_vars or {})
        saved_note = "" if db_saved else " (DB save failed)"
        yield f"data: Saved {n} variable(s){saved_note} for '{server_name}'.\n\n"

        if not deployer:
            yield "data: No deployer configured — env vars updated in memory only.\n\n"
            yield "data: Restart the server manually to apply changes.\n\n"
            return

        try:
            yield f"data: Redeploying '{server_name}' with updated environment variables...\n\n"
            # no_build=True: rewrite compose file + restart container without rebuilding the image.
            async for line in deployer.stream_deploy_server(
                **self._server_deploy_kwargs(server_name, server_config, no_build=True)
            ):
                yield f"data: {line.strip()}\n\n"
            self.config.servers[server_name].enabled = True
            await self.db_logger.log_server_action(server_name, "env-update", server_config.current_version)
            yield f"data: ✓ Server '{server_name}' redeployed with updated environment variables.\n\n"
        except Exception as e:
            yield f"data: ✗ Error redeploying server: {str(e)}\n\n"

    # ------------------------------------------------------------------
    # Lifecycle actions
    # ------------------------------------------------------------------

    async def execute_lifecycle_action(
        self,
        server_name: str,
        action: str,
        version: Optional[str] = None,
        stream_output: bool = False,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Dispatches a lifecycle action (start/stop/restart/rebuild/redeploy/update) on a server."""
        if server_name not in self.config.servers:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

        server_config = self.config.servers[server_name]
        deployer      = self._get_deployer(getattr(server_config, "deployment_mode", "docker"))
        logger.info("Lifecycle action '%s' on '%s' (deployer: %s).",
                    action, server_name, getattr(server_config, "deployment_mode", "docker"))

        if not deployer:
            return self._lifecycle_json(server_name, action)

        try:
            if action in ("update", "upgrade", "downgrade"):
                return await self._action_version_switch(server_name, server_config, deployer, action, version, stream_output)

            if action == "stop":
                return await self._action_stop(server_name, server_config, deployer)

            if action == "start":
                return await self._action_start(server_name, server_config, deployer)

            if action == "restart":
                return await self._action_restart(server_name, server_config, deployer, stream_output)

            if action == "delete":
                return await self._action_delete(server_name, server_config, deployer)

            if action == "rebuild":
                return await self._action_rebuild(server_name, server_config, deployer, stream_output)

            if action == "redeploy":
                return await self._action_redeploy(server_name, server_config, deployer, stream_output)

            raise HTTPException(
                status_code=400,
                detail=f"Unknown action '{action}'. Supported: start, stop, restart, rebuild, redeploy, update, upgrade, downgrade, delete.",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Lifecycle action '%s' failed for '%s': %s", action, server_name, e)
            raise HTTPException(status_code=500, detail=str(e))

    # ── Individual action handlers ────────────────────────────────────────

    async def _action_stop(self, server_name, server_config, deployer) -> JSONResponse:
        deployer.stop_server(server_name)
        server_config.enabled = False
        await self.db_logger.log_server_action(server_name, "stop", server_config.current_version)
        return self._lifecycle_json(server_name, "stop")

    async def _action_start(self, server_name, server_config, deployer) -> JSONResponse:
        deployer.start_server(server_name)
        server_config.enabled = True
        await self.db_logger.log_server_action(server_name, "start", server_config.current_version)
        return self._lifecycle_json(server_name, "start")

    async def _action_delete(self, server_name, server_config, deployer) -> JSONResponse:
        deployer.remove_server(server_name)
        if self.stop_sub_app:
            await self.stop_sub_app(server_name)
        elif server_name in self.sub_apps:
            del self.sub_apps[server_name]
        self.config.servers.pop(server_name, None)
        await self.db_logger.delete_server(server_name)
        await self.db_logger.log_server_action(server_name, "delete", None)
        logger.info("Server '%s' deleted.", server_name)
        return JSONResponse({"message": f"Server '{server_name}' deleted.", "action": "delete", "status": "completed"})

    async def _action_version_switch(
        self, server_name, server_config, deployer, action, version, stream_output
    ) -> Union[JSONResponse, StreamingResponse]:
        if not version:
            raise HTTPException(status_code=400, detail="Version is required for update/upgrade/downgrade.")

        no_build = deployer.image_exists(server_name, version)

        # Update in memory + DB before deployment so the version is persisted even if
        # the process is interrupted.
        server_config.current_version = version
        server_config.available_versions = list(
            set(server_config.available_versions or []) | {version}
        )
        await self._persist_server_to_db(server_name, server_config)

        deploy_kwargs = self._server_deploy_kwargs(
            server_name, server_config, refresh_repo=not no_build, no_build=no_build
        )

        if stream_output:
            return _sse(self._stream_version_switch(server_name, server_config, deployer, action, version, deploy_kwargs))

        await deployer.deploy_server(**deploy_kwargs)
        server_config.enabled = True
        await self.db_logger.log_server_action(server_name, action, version)
        return self._lifecycle_json(server_name, action)

    async def _stream_version_switch(self, server_name, server_config, deployer, action, version, deploy_kwargs):
        try:
            yield f"data: Switching '{server_name}' to version {version}...\n\n"
            async for line in deployer.stream_deploy_server(**deploy_kwargs):
                yield f"data: {line.strip()}\n\n"
            self.config.servers[server_name].enabled = True
            await self.db_logger.log_server_action(server_name, action, version)
            yield f"data: ✓ '{server_name}' successfully switched to {version}!\n\n"
        except Exception as e:
            yield f"data: ✗ Error during version switch: {str(e)}\n\n"
            raise

    async def _action_restart(self, server_name, server_config, deployer, stream_output) -> Union[JSONResponse, StreamingResponse]:
        await self.db_logger.log_server_action(server_name, "restart", server_config.current_version)
        deploy_kwargs = self._server_deploy_kwargs(server_name, server_config, no_build=True)

        if stream_output:
            return _sse(self._stream_restart(server_name, server_config, deployer, deploy_kwargs))

        async for _ in deployer.stream_deploy_server(**deploy_kwargs):
            pass
        server_config.enabled = True
        return self._lifecycle_json(server_name, "restart")

    async def _stream_restart(self, server_name, server_config, deployer, deploy_kwargs):
        try:
            yield f"data: Restarting '{server_name}'...\n\n"
            async for line in deployer.stream_deploy_server(**deploy_kwargs):
                yield f"data: {line.strip()}\n\n"
            self.config.servers[server_name].enabled = True
            yield f"data: ✓ '{server_name}' restarted successfully!\n\n"
        except Exception as e:
            yield f"data: ✗ Error during restart: {str(e)}\n\n"
            raise

    async def _action_rebuild(self, server_name, server_config, deployer, stream_output) -> Union[JSONResponse, StreamingResponse]:
        deploy_kwargs = self._server_deploy_kwargs(server_name, server_config, refresh_repo=True)

        if stream_output:
            return _sse(self._stream_rebuild(server_name, server_config, deployer, deploy_kwargs))

        deployer.remove_server(server_name)
        server_config.enabled = False
        logger.info("Server '%s' stopped for rebuild.", server_name)
        await deployer.deploy_server(**deploy_kwargs)
        server_config.enabled = True
        await self.db_logger.log_server_action(server_name, "rebuild", server_config.current_version)
        return self._lifecycle_json(server_name, "rebuild")

    async def _stream_rebuild(self, server_name, server_config, deployer, deploy_kwargs):
        try:
            deployer.remove_server(server_name)
            yield f"data: '{server_name}' stopped. Rebuilding image...\n\n"
            async for line in deployer.stream_deploy_server(**deploy_kwargs):
                yield f"data: {line.strip()}\n\n"
            self.config.servers[server_name].enabled = True
            await self.db_logger.log_server_action(server_name, "rebuild", self._get_server_current_version(server_name))
            yield f"data: ✓ '{server_name}' rebuild completed successfully!\n\n"
        except Exception as e:
            yield f"data: ✗ Error during rebuild: {str(e)}\n\n"
            raise

    async def _action_redeploy(self, server_name, server_config, deployer, stream_output) -> Union[JSONResponse, StreamingResponse]:
        """Fires an async (fire-and-forget) redeploy task."""
        deploy_kwargs = self._server_deploy_kwargs(server_name, server_config, refresh_repo=True)

        if stream_output:
            return _sse(self._stream_redeploy(server_name, server_config, deployer, deploy_kwargs))

        asyncio.create_task(deployer.deploy_server(**deploy_kwargs))
        await self.db_logger.log_server_action(server_name, "redeploy", server_config.current_version)
        return self._lifecycle_json(server_name, "redeploy", status="initiated")

    async def _stream_redeploy(self, server_name, server_config, deployer, deploy_kwargs):
        try:
            asyncio.create_task(deployer.deploy_server(**deploy_kwargs))
            await self.db_logger.log_server_action(server_name, "redeploy", self._get_server_current_version(server_name))
            yield f"data: ✅ Redeploy task initiated for '{server_name}'.\n\n"
        except Exception as e:
            yield f"data: ✗ Error initiating redeploy: {str(e)}\n\n"
            raise

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _lifecycle_json(self, server_name: str, action: str, status: str = "completed") -> JSONResponse:
        enabled = self.config.servers[server_name].enabled if server_name in self.config.servers else False
        return JSONResponse({
            "message": f"Lifecycle action '{action}' executed for server '{server_name}'.",
            "server":  server_name,
            "action":  action,
            "status":  status,
            "enabled": enabled,
        })
