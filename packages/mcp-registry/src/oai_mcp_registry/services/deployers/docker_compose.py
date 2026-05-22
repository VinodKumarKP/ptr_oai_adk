"""
MCP-registry DockerComposeManager.

Extends :class:`oai_platform_core.deployers.docker_compose_base.BaseDockerComposeManager`
with MCP-server-specific infrastructure (single mcp base image, ``mcp-``
container prefix, ``mcp-server-network``) and satisfies the
:class:`oai_mcp_registry.services.deployers.base.BaseDeployer` abstract
contract (``deploy_server``, ``stream_deploy_server``, ``start_server``, …).
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.docker_compose_base import BaseDockerComposeManager

from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.utils.env_vars import get_common_server_env

_MCP_HOME_USER = "mcpuser"


class DockerComposeManager(BaseDockerComposeManager, BaseDeployer):  # type: ignore[misc]
    """Docker Compose manager for the MCP registry.

    Manages a dynamically generated docker-compose file that includes
    infrastructure services (valkey, postgres, mcp base image) and
    the deployed MCP server services.
    """

    _compose_name = "mcp-registry"
    _network_name = "mcp-server-network"
    _service_type_label = "server"

    # ------------------------------------------------------------------
    # Abstract hook — environment vars
    # ------------------------------------------------------------------

    def _get_service_env(
        self,
        service_name: str,
        port: int,
        env_overrides: Dict[str, Any],
        base_url: str,
        local_registry_url: str,
    ) -> Dict[str, str]:
        return get_common_server_env(
            server_name=service_name,
            port=port,
            base_url=base_url,
            local_registry_url=local_registry_url,
            env_overrides=env_overrides,
        )

    # ------------------------------------------------------------------
    # Abstract hook — per-service compose block
    # ------------------------------------------------------------------

    def _build_service_dict(
        self,
        service_name: str,
        config: Dict[str, Any],
        base_url: str,
        local_registry_url: str,
        refresh_repo: bool = False,
    ) -> Dict[str, Any]:
        port = config["port"]
        current_version = config.get("current_version")
        image_tag = current_version or "latest"
        image_name = f"{service_name.replace('_', '-')}:{image_tag}"

        base_image = "oai-adk-mcp-base-image:latest"
        dockerfile = "Dockerfile"

        build_args: Dict[str, str] = {
            "BASE_IMAGE": base_image,
            "GITHUB_TOKEN": "${GITHUB_TOKEN}",
            "GITHUB_URL": str(config.get("source", "")),
        }
        if refresh_repo:
            build_args["REFRESH_REPO"] = "true"
            build_args["CACHE_BUST"] = str(int(time.time()))

        volumes = ["./logs:/tmp"]
        aws_path = os.path.expanduser("~/.aws")
        if os.path.isdir(aws_path):
            volumes.append(f"{aws_path}:/home/{_MCP_HOME_USER}/.aws:ro")

        env_overrides = config.get("env") or config.get("environment") or {}

        return {
            "image": image_name,
            "build": {"context": ".", "dockerfile": dockerfile, "args": build_args},
            "container_name": f"mcp-{service_name.replace('_', '-')}",
            "ports": [f"{port}:{port}"],
            "volumes": volumes,
            "environment": self._build_environment_list(
                service_name, port, env_overrides, base_url, local_registry_url
            ),
            "command": f"--port {port} --transport streamable-http",
            "restart": "unless-stopped",
            "networks": [self._network_name],
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "healthcheck": {
                "test": ["CMD", "curl", "-f", f"http://localhost:{port}/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 15,
            },
            "depends_on": {
                "base":     {"condition": "service_completed_successfully"},
                "valkey":   {"condition": "service_healthy"},
                "postgres": {"condition": "service_healthy"},
            },
        }

    # ------------------------------------------------------------------
    # Abstract hook — infrastructure compose block
    # ------------------------------------------------------------------

    def _build_infra_compose_dict(self) -> Dict[str, Any]:
        services: Dict[str, Any] = {
            "base": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.base",
                    "secrets": ["github_token"],
                },
                "image": "oai-adk-mcp-base-image:latest",
            },
        }

        services["valkey"] = {
            "image": "valkey/valkey:latest",
            "container_name": "mcp-valkey",
            "environment": {"REDIS_PASSWORD": "admin"},
            "networks": {self._network_name: {"ipv4_address": "172.26.0.11"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "command": "valkey-server --appendonly yes",
            "ports": ["6380:6379"],
            "volumes": ["valkey-mcp-data:/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "redis-cli", "-a", "admin", "ping"],
                "interval": "30s", "timeout": "10s",
                "retries": 5, "start_period": "15s",
            },
        }

        services["postgres"] = {
            "image": "postgres:16",
            "container_name": "mcp_logs_db",
            "environment": {
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_DB": "mcp_logs",
                "PGDATA": "/var/lib/postgresql/data/pgdata",
            },
            "command": [
                "postgres",
                "-c", "max_connections=300",
                "-c", "shared_buffers=256MB",
                "-c", "effective_io_concurrency=200",
            ],
            "networks": {self._network_name: {"ipv4_address": "172.26.0.10"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "ports": ["5433:5432"],
            "volumes": ["mcp_postgres_data:/var/lib/postgresql/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres"],
                "interval": "30s", "timeout": "10s",
                "retries": 5, "start_period": "15s",
            },
        }

        return {
            "name": self._compose_name,
            "services": services,
            "networks": {
                self._network_name: {
                    "driver": "bridge",
                    "ipam": {"config": [{"subnet": "172.26.0.0/16"}]},
                }
            },
            "volumes": {"valkey-mcp-data": {}, "mcp_postgres_data": {}},
            "secrets": {"github_token": {"environment": "GITHUB_TOKEN"}},
        }

    # ------------------------------------------------------------------
    # BaseDeployer abstract methods — server-specific names
    # ------------------------------------------------------------------

    def add_server_from_registration(
        self,
        server_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        current_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._add_service_from_registration(
            server_name, source_url, framework, env, description, tags, port, current_version
        )

    async def deploy_server(
        self,
        server_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        current_version: Optional[str] = None,
        refresh_repo: bool = False,
        no_build: bool = False,
    ) -> str:
        self._add_service_from_registration(
            server_name, source_url, framework, env, description, tags, port, current_version
        )
        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        return self._run_compose_up_service(server_name, no_build=no_build)

    async def stream_deploy_server(
        self,
        server_name: str,
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
        async for line in self._stream_deploy_service(
            server_name, source_url, framework, env, description,
            tags, port, current_version, refresh_repo, no_build,
        ):
            yield line

    def start_server(self, server_name: str) -> str:
        return self._run_compose_up_service(server_name)

    def stop_server(self, server_name: str) -> str:
        return self._run_compose_stop_service(server_name)

    def remove_server(self, server_name: str) -> str:
        self._dynamic_services.pop(server_name, None)
        return self._run_compose_down_service(server_name)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_all_servers(self) -> Dict[str, Any]:
        return {**self._seed_config, **self._dynamic_services}
