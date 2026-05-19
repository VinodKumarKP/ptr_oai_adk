"""
oai_platform_core.deployers.docker_compose_base — shared Docker Compose manager.

Both the agent-registry and mcp-registry packages manage a dynamically generated
``docker-compose.yaml``.  The two concrete implementations are >90 % identical;
they differ only in:

* Infrastructure services (networks, IPs, base images, container names)
* Per-service env-var builder (``get_common_agent_env`` / ``get_common_server_env``)
* Container naming prefix (``agent-`` / ``mcp-``)
* Network name (``agent-server-network`` / ``mcp-server-network``)
* Startup command suffix (MCP adds ``--transport streamable-http``)
* AWS-credentials volume path (``agentuser`` / ``mcpuser``)

This module provides :class:`BaseDockerComposeManager`, a concrete base class that
implements all shared logic.  Subclasses override the handful of abstract hooks:

``_build_infra_compose_dict()``
    Infrastructure block (valkey, postgres, base images).

``_build_service_dict()``
    Per-service Docker Compose block (image, build, ports, env, …).

``_get_service_env()``
    Returns the full env-var dict for a service (delegates to the package's
    ``get_common_*_env`` helper).

All public ``deploy_*`` / ``start_*`` / ``stop_*`` / ``remove_*`` methods are
implemented by the per-package subclasses (agent-registry keeps ``deploy_agent``,
mcp-registry keeps ``deploy_server``) but they all delegate to the private helpers
defined here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import time
from abc import abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ruamel.yaml import YAML

from oai_platform_core.deployers.base import BaseDeployer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level YAML instance (shared settings)
# ---------------------------------------------------------------------------
_yaml = YAML()
_yaml.default_flow_style = False
_yaml.indent(mapping=2, sequence=4, offset=2)
_yaml.width = 1000


# ---------------------------------------------------------------------------
# Standalone helpers (identical across both registry packages)
# ---------------------------------------------------------------------------

def _is_port_available(port: int) -> bool:
    """Return ``True`` if *port* can be bound on the local host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        return False


def _get_next_available_port(
    used_ports: Optional[List[int]] = None,
    start_port: int = 8000,
    skip_ports: Optional[List[int]] = None,
) -> int:
    """Return the next available port starting from *start_port*."""
    if skip_ports is None:
        skip_ports = [8080, 8081, 8082]

    used_ports = used_ports or []
    reserved = set(skip_ports + used_ports)

    port = start_port
    for _ in range(10_000):
        if port not in reserved and _is_port_available(port):
            return port
        port += 1

    raise RuntimeError(
        f"Could not find an available port starting from {start_port} "
        "(checked 10 000 ports)"
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDockerComposeManager(BaseDeployer):
    """Shared Docker Compose manager for OAI registry packages.

    Subclasses **must** set the following class-level attributes and
    implement the abstract hooks listed below.

    Class attributes
    ----------------
    _compose_name : str
        Docker Compose project name (``"agent-registry"`` or
        ``"mcp-registry"``).
    _network_name : str
        Docker network name (``"agent-server-network"`` or
        ``"mcp-server-network"``).
    _service_type_label : str
        Human-readable service type used in log messages
        (``"agent"`` or ``"server"``).
    """

    # Subclasses override these
    _compose_name: str = "registry"
    _network_name: str = "service-network"
    _service_type_label: str = "service"

    def __init__(
        self,
        seed_config: Dict[str, Any],
        compose_output_path: str = "docker-compose.generated.yaml",
        base_compose_path: str = "docker-compose.yaml",
        agent_base_url: str = (
            f"{os.environ.get('AGENT_BASE_URL', 'localhost')}"
            f":{os.environ.get('AGENT_BASE_URL_PORT', 8081)}"
        ),
        agent_local_registry_url: str = "http://host.docker.internal:8081",
    ) -> None:
        self._seed_config: Dict[str, Any] = dict(seed_config)
        self._dynamic_services: Dict[str, Any] = {}

        self._output_path = Path(compose_output_path)
        self._base_path = Path(base_compose_path)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url
        self.used_ports: List[int] = []

    # ------------------------------------------------------------------
    # BaseDeployer — lifecycle (identical in both packages)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Generate compose files and bring up all services."""
        self.write_compose_file()
        self._run_compose_up()

    async def shutdown(self) -> None:
        """Tear down all services."""
        await self._run_compose_down()

    def find_available_port(self) -> int:
        all_services = {**self._seed_config, **self._dynamic_services}
        for config in all_services.values():
            if isinstance(config, dict) and "port" in config:
                self.used_ports.append(config["port"])
        return _get_next_available_port(used_ports=self.used_ports)

    def image_exists(self, service_name: str, version: str) -> bool:
        """Return ``True`` if the Docker image for *service_name:version* exists locally."""
        image = f"{service_name.replace('_', '-')}:{version}"
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
        )
        exists = result.returncode == 0 and bool(result.stdout.strip())
        logger.debug("Image '%s' %s in local store.", image, "found" if exists else "not found")
        return exists

    # ------------------------------------------------------------------
    # Infrastructure services startup (identical in both packages)
    # ------------------------------------------------------------------

    def start_infra_services(self, services: Optional[List[str]] = None) -> None:
        """Generate compose files and start only infra services (postgres, valkey).

        Uses ``docker compose up --wait`` so callers block until healthchecks
        pass.  Falls back gracefully for older Docker Compose versions that
        do not support ``--wait``.

        Args:
            services: Service names to start; defaults to
                      ``["postgres", "valkey"]``.
        """
        self.write_compose_file()
        target = services or ["postgres", "valkey"]

        for use_wait in (True, False):
            cmd = [
                "docker", "compose",
                "-f", str(self._base_path),
                "up", "-d",
            ]
            if use_wait:
                cmd.append("--wait")
            cmd.extend(target)

            logger.info("Starting infra services %s: %s", target, " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Infra services started successfully.")
                return

            output = (result.stdout + result.stderr).lower()
            if use_wait and ("unknown flag" in output or "unknown shorthand" in output):
                logger.debug("--wait not supported; retrying without it")
                continue

            raise RuntimeError(
                f"docker compose up failed for infra services "
                f"(exit {result.returncode}):\n{result.stderr}"
            )

    # ------------------------------------------------------------------
    # Compose file generation
    # ------------------------------------------------------------------

    def write_compose_file(self, refresh_repo: bool = False) -> Path:
        """Write both the infra and services compose YAML files to disk."""
        infra_dict = self._build_infra_compose_dict()
        self._write_yaml(infra_dict, self._base_path)

        merged = {**self._seed_config, **self._dynamic_services}
        services_dict = self._build_services_compose_dict(merged, refresh_repo)
        self._write_yaml(services_dict, self._output_path)

        label = f"{self._service_type_label}s"
        logger.info(
            "Compose files written to %s and %s (%d seed + %d dynamic %s)",
            self._base_path,
            self._output_path,
            len(self._seed_config),
            len(self._dynamic_services),
            label,
        )
        return self._output_path

    def _build_services_compose_dict(
        self,
        service_config: Dict[str, Any],
        refresh_repo: bool = False,
    ) -> Dict[str, Any]:
        """Build the services (non-infra) portion of the compose dict."""
        services: Dict[str, Any] = {}

        for service_name, config in service_config.items():
            source = config.get("source")
            if not source:
                logger.warning("Skipping '%s': no source defined.", service_name)
                continue

            port = config.get("port")
            self.used_ports.append(port)
            if port is None or port in self.used_ports:
                port = self.find_available_port()
                self.used_ports.append(port)
                config["port"] = port

            services[service_name] = self._build_service_dict(
                service_name, config, self._base_url, self._local_registry_url, refresh_repo
            )

        return {
            "name": self._compose_name,
            "services": services,
            "networks": {self._network_name: None},
        }

    def _build_environment_list(
        self,
        service_name: str,
        port: int,
        env_overrides: Dict[str, Any],
        base_url: str,
        local_registry_url: str,
    ) -> List[str]:
        """Convert the service env-var dict to a Docker Compose ``environment`` list."""
        env_dict = self._get_service_env(
            service_name, port, env_overrides, base_url, local_registry_url
        )
        environment: List[str] = []
        for key, value in env_dict.items():
            value_str = str(value)
            if value_str.startswith("${") and value_str.endswith("}") and ":-" in value_str:
                environment.append(f"{key}={value_str}")
            elif "$" in value_str and ("{" in value_str or " " not in value_str):
                var_name = value_str.replace("${", "").replace("}", "").replace("$", "")
                environment.append(f"{key}=${{{var_name}:-$${var_name}}}")
            else:
                environment.append(f"{key}={value_str}")
        return environment

    # ------------------------------------------------------------------
    # Registration helper (identical logic, different dict key)
    # ------------------------------------------------------------------

    def _add_service_from_registration(
        self,
        service_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        current_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage a new service in ``_dynamic_services`` and return its config."""
        tags = tags or []
        if framework and framework.lower() not in tags:
            tags.append(framework.lower())

        if port is None:
            port = self.find_available_port()
            self.used_ports.append(port)
            logger.info(
                "Auto-assigned port %d to %s '%s'",
                port, self._service_type_label, service_name,
            )

        config: Dict[str, Any] = {
            "port": port,
            "source": source_url,
            "framework": framework or "",
            "tags": tags,
            "env": env or {},
            "description": description or "",
            "current_version": current_version,
        }
        self._dynamic_services[service_name] = config
        logger.debug(
            "Staged dynamic %s '%s' for next compose generation.",
            self._service_type_label, service_name,
        )
        return config

    # ------------------------------------------------------------------
    # Streaming deploy (identical async generator logic)
    # ------------------------------------------------------------------

    async def _stream_deploy_service(
        self,
        service_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        current_version: Optional[str] = None,
        refresh_repo: bool = False,
        no_build: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Async generator that stages, writes, and streams the compose build."""
        self._add_service_from_registration(
            service_name, source_url, framework, env,
            description, tags, port, current_version,
        )
        yield f"Staged {self._service_type_label} '{service_name}' for deployment\n"

        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        yield "Generated docker-compose files\n"

        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "up", "-d", "--no-deps",
            service_name,
        ]
        if not no_build:
            cmd.insert(-1, "--build")

        action = "Restarting with existing image" if no_build else "Starting docker compose build and deployment"
        yield f"{action}...\n"
        yield f"Command: {' '.join(cmd)}\n\n"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield line.decode("utf-8", errors="replace")
                await asyncio.sleep(0)

            rc = await process.wait()
            if rc != 0:
                yield f"\n❌ Deployment failed with exit code {rc}\n"
                raise RuntimeError(
                    f"docker compose up failed for '{service_name}' with exit code {rc}"
                )
            yield f"\n✅ {self._service_type_label.capitalize()} '{service_name}' deployed successfully!\n"

        except Exception as exc:
            yield f"\n❌ Error during deployment: {exc}\n"
            raise

    # ------------------------------------------------------------------
    # Private compose-command runners (identical in both packages)
    # ------------------------------------------------------------------

    def _run_compose_up_service(self, service_name: str, no_build: bool = False) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "up", "-d", "--no-deps",
            service_name,
        ]
        if not no_build:
            cmd.insert(-1, "--build")

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "docker compose up failed for '%s':\n%s", service_name, result.stderr
            )
            raise RuntimeError(
                f"docker compose up failed for '{service_name}': {result.stderr}"
            )
        logger.info(
            "%s '%s' deployed successfully.",
            self._service_type_label.capitalize(), service_name,
        )
        return result.stdout

    def _run_compose_up(self) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "up", "-d", "--no-deps", "--build",
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("docker compose up failed:\n%s", result.stderr)
            raise RuntimeError(f"docker compose up failed: {result.stderr}")
        return result.stdout

    async def _run_compose_down(self) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "down",
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("docker compose down failed:\n%s", result.stderr)
            raise RuntimeError(f"docker compose down failed: {result.stderr}")
        return result.stdout

    def _run_compose_down_service(self, service_name: str) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "down", service_name,
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "docker compose down failed for '%s':\n%s", service_name, result.stderr
            )
            raise RuntimeError(
                f"docker compose down failed for '{service_name}': {result.stderr}"
            )
        logger.info(
            "%s '%s' shutdown successfully.",
            self._service_type_label.capitalize(), service_name,
        )
        return result.stdout

    def _run_compose_stop_service(self, service_name: str) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "stop", service_name,
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "docker compose stop failed for '%s':\n%s", service_name, result.stderr
            )
            raise RuntimeError(
                f"docker compose stop failed for '{service_name}': {result.stderr}"
            )
        logger.info(
            "%s '%s' paused successfully.",
            self._service_type_label.capitalize(), service_name,
        )
        return result.stdout

    @staticmethod
    def _write_yaml(data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        _yaml.dump(data, buf)
        path.write_text(buf.getvalue(), encoding="utf-8")

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_infra_compose_dict(self) -> Dict[str, Any]:
        """Return the infrastructure compose block (valkey, postgres, base images)."""

    @abstractmethod
    def _build_service_dict(
        self,
        service_name: str,
        config: Dict[str, Any],
        base_url: str,
        local_registry_url: str,
        refresh_repo: bool = False,
    ) -> Dict[str, Any]:
        """Return the Docker Compose service block for one deployed service."""

    @abstractmethod
    def _get_service_env(
        self,
        service_name: str,
        port: int,
        env_overrides: Dict[str, Any],
        base_url: str,
        local_registry_url: str,
    ) -> Dict[str, str]:
        """Return the full env-var dict for a service (calls get_common_*_env)."""
