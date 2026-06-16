"""
Agent-registry DockerComposeManager.

Extends :class:`oai_platform_core.deployers.docker_compose_base.BaseDockerComposeManager`
with agent-specific infrastructure (multiple framework base images, ``agent-``
container prefix, ``agent-server-network``) and satisfies the
:class:`oai_agent_registry.services.deployers.base.BaseDeployer` abstract
contract (``deploy_agent``, ``stream_deploy_agent``, ``start_agent``, …).
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.docker_compose_base import BaseDockerComposeManager

from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.utils.env_vars import get_common_agent_env

_AWS_HOME_USER = "agentuser"


class DockerComposeManager(BaseDockerComposeManager, BaseDeployer):  # type: ignore[misc]
    """Docker Compose manager for the agent registry.

    Manages a dynamically generated docker-compose file that includes
    infrastructure services (valkey, postgres, framework base images) and
    the deployed agent services.
    """

    _compose_name = "agent-registry"
    _network_name = "agent-server-network"
    _service_type_label = "agent"

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
        return get_common_agent_env(
            agent_name=service_name,
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
        tags = config.get("tags", [])
        framework = config.get("framework", "")
        current_version = config.get("current_version")
        image_tag = current_version or "latest"
        image_name = f"{service_name.replace('_', '-')}:{image_tag}"

        # Framework-specific base images
        _base_images = {
            "langgraph": "oai-adk-langgraph-base-image:latest",
            "crewai": "oai-adk-crewai-base-image:latest",
            "strands": "oai-adk-strands-base-image:latest",
            "openai": "oai-adk-openai-base-image:latest",
        }
        base_image = _base_images.get(framework, "oai-adk-base-image:latest")
        dockerfile = "Dockerfile_debian" if "crewai" in tags else "Dockerfile"

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
            volumes.append(f"{aws_path}:/home/{_AWS_HOME_USER}/.aws:ro")

        env_overrides = config.get("env") or config.get("environment") or {}

        return {
            "image": image_name,
            "build": {"context": ".", "dockerfile": dockerfile, "args": build_args},
            "container_name": f"agent-{service_name.replace('_', '-')}",
            "ports": [f"{port}:{port}"],
            # Labels consumed by Prometheus Docker service discovery so newly
            # deployed agents are scraped automatically (see resources/docker/
            # prometheus.yml -> job 'oai-agents').
            "labels": {
                "oai.scrape": "true",
                "oai.metrics.port": str(port),
                "oai.metrics.path": "/metrics",
            },
            "volumes": volumes,
            "environment": self._build_environment_list(
                service_name, port, env_overrides, base_url, local_registry_url
            ),
            "command": f"--port {port}",
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
                "image": "oai-adk-base-image:latest",
            },
        }

        _frameworks = [
            ("base-langgraph", "oai-langgraph-core",   "packages/langgraph-core",  "oai-adk-langgraph-base-image:latest"),
            ("base-crewai",    "oai-crewai-core",       "packages/crewai-core",     "oai-adk-crewai-base-image:latest"),
            ("base-strands",   "oai-aws-strands-core",  "packages/aws-strands-core","oai-adk-strands-base-image:latest"),
            ("base-openai",    "oai-openai-core",       "packages/openai-core",     "oai-adk-openai-base-image:latest"),
        ]
        for svc_name, pkg, subdir, image in _frameworks:
            services[svc_name] = {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.framework_base",
                    "args": {
                        "FRAMEWORK_PACKAGE": pkg,
                        "FRAMEWORK_SUBDIR": subdir,
                        "ADK_VERSION": "main",
                    },
                },
                "image": image,
                "depends_on": {"base": {"condition": "service_completed_successfully"}},
            }

        services["valkey"] = {
            "image": "valkey/valkey:latest",
            "container_name": "agent-valkey",
            "environment": {"REDIS_PASSWORD": "admin"},
            "networks": {self._network_name: {"ipv4_address": "172.25.0.11"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "command": "valkey-server --appendonly yes",
            "ports": ["6379:6379"],
            "volumes": ["valkey-agent-data:/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "redis-cli", "-a", "admin", "ping"],
                "interval": "30s", "timeout": "10s",
                "retries": 5, "start_period": "15s",
            },
        }

        services["postgres"] = {
            "image": "postgres:16",
            "container_name": "agent_logs_db",
            "environment": {
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_DB": "agent_logs",
                "PGDATA": "/var/lib/postgresql/data/pgdata",
            },
            "command": [
                "postgres",
                "-c", "max_connections=300",
                "-c", "shared_buffers=256MB",
                "-c", "effective_io_concurrency=200",
            ],
            "networks": {self._network_name: {"ipv4_address": "172.25.0.10"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "ports": ["5432:5432"],
            "volumes": ["postgres_data:/var/lib/postgresql/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres"],
                "interval": "30s", "timeout": "10s",
                "retries": 5, "start_period": "15s",
            },
        }

        # ── Observability infra (distributed tracing + metrics dashboards) ─────
        # Mirrors packages/agent-server/examples/docker-compose.yml. Agents and
        # the registry export OTLP traces to Jaeger (:4317) and expose Prometheus
        # metrics; Grafana is pre-wired to both via provisioned data sources.
        services["jaeger"] = {
            # Jaeger v2 (OpenTelemetry-Collector based). v1 (all-in-one) is EOL as
            # of 2025-12-31. v2 is OTLP-native and ships an all-in-one default
            # config (OTLP receivers + in-memory storage + UI), so no extra flags
            # are needed. Legacy agent (6831/udp) and collector (14250) ports are
            # gone — agents send OTLP to :4317.
            "image": "jaegertracing/jaeger:latest",
            "container_name": "agent-jaeger",
            "networks": {self._network_name: {"ipv4_address": "172.25.0.12"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "ports": [
                "16686:16686",     # Jaeger UI
                "4317:4317",       # OTLP gRPC
                "4318:4318",       # OTLP HTTP
            ],
            "restart": "unless-stopped",
        }

        services["prometheus"] = {
            "image": "prom/prometheus:latest",
            "container_name": "agent-prometheus",
            # Run as root so the container can read the host Docker socket used by
            # docker_sd_configs (the default 'nobody' user gets permission denied).
            "user": "root",
            "command": [
                "--config.file=/etc/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
            ],
            "networks": {self._network_name: {"ipv4_address": "172.25.0.13"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "ports": ["9090:9090"],
            "volumes": [
                "./prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                "prometheus_data:/prometheus",
                # Read-only Docker socket for service discovery of agent containers.
                "/var/run/docker.sock:/var/run/docker.sock:ro",
            ],
            "restart": "unless-stopped",
        }

        services["grafana"] = {
            "image": "grafana/grafana:latest",
            "container_name": "agent-grafana",
            "environment": {
                "GF_SECURITY_ADMIN_USER": "admin",
                "GF_SECURITY_ADMIN_PASSWORD": "admin",
                "GF_AUTH_ANONYMOUS_ENABLED": "true",
            },
            "networks": {self._network_name: {"ipv4_address": "172.25.0.14"}},
            "extra_hosts": ["host.docker.internal:host-gateway"],
            "ports": ["3000:3000"],
            "volumes": [
                "grafana_data:/var/lib/grafana",
                "./grafana/provisioning:/etc/grafana/provisioning:ro",
            ],
            "depends_on": ["prometheus", "jaeger"],
            "restart": "unless-stopped",
        }

        return {
            "name": self._compose_name,
            "services": services,
            "networks": {
                self._network_name: {
                    "driver": "bridge",
                    "ipam": {"config": [{"subnet": "172.25.0.0/16"}]},
                }
            },
            "volumes": {
                "valkey-agent-data": {},
                "postgres_data": {},
                "prometheus_data": {},
                "grafana_data": {},
            },
            "secrets": {"github_token": {"environment": "GITHUB_TOKEN"}},
        }

    # ------------------------------------------------------------------
    # BaseDeployer abstract methods — agent-specific names
    # ------------------------------------------------------------------

    def add_agent_from_registration(
        self,
        agent_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        current_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._add_service_from_registration(
            agent_name, source_url, framework, env, description, tags, port, current_version
        )

    async def deploy_agent(
        self,
        agent_name: str,
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
            agent_name, source_url, framework, env, description, tags, port, current_version
        )
        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        return self._run_compose_up_service(agent_name, no_build=no_build)

    async def stream_deploy_agent(
        self,
        agent_name: str,
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
            agent_name, source_url, framework, env, description,
            tags, port, current_version, refresh_repo, no_build,
        ):
            yield line

    def start_agent(self, agent_name: str) -> str:
        return self._run_compose_up_service(agent_name)

    def stop_agent(self, agent_name: str) -> str:
        return self._run_compose_stop_service(agent_name)

    def remove_agent(self, agent_name: str) -> str:
        self._dynamic_services.pop(agent_name, None)
        return self._run_compose_down_service(agent_name)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_all_agents(self) -> Dict[str, Any]:
        return {**self._seed_config, **self._dynamic_services}
