"""
DockerComposeManager — wraps docker_compose_generator.py logic into a class
that can be driven by the AgentRegistry (/register endpoint) at runtime.

Responsibilities:
  1. Accept a seed config dict (same shape as the YAML/JSON files your
     existing generator loads from disk) so the infra services (valkey,
     postgres, base image, agent-proxy) are always present.
  2. Merge dynamically registered agents on top of that seed.
  3. Re-generate the compose file and run `docker compose up` for only
     the new service — without touching already-running containers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from ruamel.yaml import YAML

from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.utils.env_vars import get_common_agent_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level YAML instance (same settings as the original generator)
# ---------------------------------------------------------------------------
_yaml = YAML()
_yaml.default_flow_style = False
_yaml.indent(mapping=2, sequence=4, offset=2)
_yaml.width = 1000


# ---------------------------------------------------------------------------
# Helpers (ported from docker_compose_generator.py, kept as private statics)
# ---------------------------------------------------------------------------

def _aws_volume() -> Optional[str]:
    aws_path = os.path.expanduser("~/.aws")
    if os.path.isdir(aws_path):
        return f"{aws_path}:/home/agentuser/.aws:ro"
    return None


def _is_port_available(port: int) -> bool:
    """
    Checks if a port is available by attempting to bind to it.
    Returns True if the port is available, False otherwise.
    """
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
    if skip_ports is None:
        skip_ports = [8080, 8081, 8082]

    used_ports = used_ports or []
    reserved_ports = set(skip_ports + used_ports)

    port = start_port
    max_attempts = 10000

    for _ in range(max_attempts):
        if port not in reserved_ports and _is_port_available(port):
            return port
        port += 1

    raise RuntimeError(
        f"Could not find an available port starting from {start_port} "
        f"(checked {max_attempts} ports)"
    )


def _build_environment(service_name: str, port: int, env_overrides: Dict[str, Any],
                       base_url: str, local_registry_url: str) -> List[str]:
    """
    Builds the environment list for an agent service.
    Mirrors the logic in generate_service_config() from the original module.
    """
    env_dict = get_common_agent_env(
        agent_name=service_name,
        port=port,
        base_url=base_url,
        local_registry_url=local_registry_url,
        env_overrides=env_overrides
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


def _build_service(service_name: str, config: Dict[str, Any],
                   base_url: str, local_registry_url: str,
                   refresh_repo: bool = False) -> Dict[str, Any]:
    port = config["port"]

    tags = config.get("tags", [])
    framework = config.get("framework", "")
    current_version = config.get("current_version")
    image_tag = current_version if current_version else "latest"
    image_name = f"{service_name.replace('_', '-')}:{image_tag}"

    if framework == 'langgraph':
        base_image = 'oai-adk-langgraph-base-image:latest'
    elif framework == 'crewai':
        base_image = 'oai-adk-crewai-base-image:latest'
    elif framework == 'strands':
        base_image = 'oai-adk-strands-base-image:latest'
    elif framework == 'openai':
        base_image = 'oai-adk-openai-base-image:latest'
    else:
        base_image = 'oai-adk-base-image:latest'

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
    aws_vol = _aws_volume()
    if aws_vol:
        volumes.append(aws_vol)

    env_overrides = config.get("env", {}) or config.get("environment", {}) or {}

    return {
        "image": image_name,
        "build": {
            "context": ".",
            "dockerfile": dockerfile,
            "args": build_args,
        },
        "container_name": f"agent-{service_name.replace('_', '-')}",
        "ports": [f"{port}:{port}"],
        "volumes": volumes,
        "environment": _build_environment(
            service_name, port, env_overrides, base_url, local_registry_url
        ),
        "command": f"--port {port}",
        "restart": "unless-stopped",
        "networks": ["agent-server-network"],
        "extra_hosts": [
            "host.docker.internal:host-gateway"
        ],
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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DockerComposeManager(BaseDeployer):
    """
    Manages a dynamically generated docker-compose file for agent services.
    """

    def __init__(
        self,
        seed_config: Dict[str, Any],
        compose_output_path: str = "docker-compose.generated.yaml",
        base_compose_path: str = "docker-compose.yaml",
        agent_base_url: str = f"{os.environ.get('AGENT_BASE_URL', 'localhost')}:{os.environ.get('AGENT_BASE_URL_PORT', 8081)}",
        agent_local_registry_url: str = "http://host.docker.internal:8081",
    ):
        self._seed_config: Dict[str, Any] = dict(seed_config)
        self._dynamic_agents: Dict[str, Any] = {}

        self._output_path = Path(compose_output_path)
        self._base_path = Path(base_compose_path)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url
        self.used_ports = []

    async def initialize(self) -> None:
        """Implements BaseDeployer.initialize — generates compose files and starts agents.

        Infra services (postgres, valkey) are expected to already be running by the
        time this is called.  If auto_start_infra is enabled, call
        start_infra_services() before initialising the database instead.
        """
        self.write_compose_file()
        self._run_compose_up()

    def start_infra_services(self, services: Optional[List[str]] = None) -> None:
        """Generate compose files and start only infra services (postgres, valkey).

        This must be called *before* the database logger is initialised so that
        Postgres is accepting connections when asyncpg tries to connect.

        Uses ``docker compose up --wait`` which blocks until the healthchecks
        defined in the compose file pass (pg_isready / redis ping).  Falls back
        to starting without --wait for older Docker Compose versions.

        Args:
            services: service names to start; defaults to ["postgres", "valkey"].
        """
        self.write_compose_file()   # ensure docker-compose.yaml is up to date

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

    async def shutdown(self) -> None:
        """Implements BaseDeployer.shutdown"""
        await self._run_compose_down()

    def find_available_port(self) -> int:
        """Implements BaseDeployer.find_available_port"""
        all_agents = {**self._seed_config, **self._dynamic_agents}
        for agent_config in all_agents.values():
            if isinstance(agent_config, dict) and "port" in agent_config:
                self.used_ports.append(agent_config["port"])

        return _get_next_available_port(used_ports=self.used_ports)

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
        tags = tags or []
        if framework and framework.lower() not in tags:
            tags.append(framework.lower())

        if port is None:
            port = self.find_available_port()
            self.used_ports.append(port)
            logger.info(f"Auto-assigned port {port} to agent '{agent_name}'")

        config = {
            "port": port,
            "source": source_url,
            "framework": framework or "",
            "tags": tags,
            "env": env or {},
            "description": description or "",
            "current_version": current_version,
        }
        self._dynamic_agents[agent_name] = config
        logger.debug(f"Staged dynamic agent '{agent_name}' for next compose generation.")
        return config

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
        self.add_agent_from_registration(
            agent_name=agent_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
            current_version=current_version,
        )
        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        return self._run_compose_up_agent(agent_name, no_build=no_build)

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
        self.add_agent_from_registration(
            agent_name=agent_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
            current_version=current_version,
        )
        yield f"Staged agent '{agent_name}' for deployment\n"

        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        yield f"Generated docker-compose files\n"

        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",
            service_name,
        ]
        if not no_build:
            cmd.insert(-1, "--build")

        action_label = "Restarting with existing image" if no_build else "Starting docker compose build and deployment"
        yield f"{action_label}...\n"
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
                decoded_line = line.decode('utf-8', errors='replace')
                yield decoded_line
                await asyncio.sleep(0)

            return_code = await process.wait()

            if return_code != 0:
                yield f"\n❌ Deployment failed with exit code {return_code}\n"
                raise RuntimeError(f"docker compose up failed for '{agent_name}' with exit code {return_code}")
            else:
                yield f"\n✅ Agent '{agent_name}' deployed successfully!\n"

        except Exception as e:
            yield f"\n❌ Error during deployment: {str(e)}\n"
            raise

    def start_agent(self, agent_name: str) -> str:
        """Implements BaseDeployer.start_agent"""
        return self._run_compose_up_agent(agent_name)

    def stop_agent(self, agent_name: str) -> str:
        """Implements BaseDeployer.stop_agent"""
        return self._run_compose_stop_agent(agent_name)

    def remove_agent(self, agent_name: str) -> str:
        """Implements BaseDeployer.remove_agent"""
        if agent_name in self._dynamic_agents:
            del self._dynamic_agents[agent_name]
        return self._run_compose_down_agent(agent_name)

    def image_exists(self, agent_name: str, version: str) -> bool:
        """Check if the deployment artifact for the given version already exists locally."""
        service_image = f"{agent_name.replace('_', '-')}:{version}"
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", service_image],
            capture_output=True,
            text=True,
        )
        exists = result.returncode == 0 and bool(result.stdout.strip())
        logger.debug(f"Image '{service_image}' {'found' if exists else 'not found'} in local store.")
        return exists

    def write_compose_file(self, refresh_repo: bool = False) -> Path:
        """
        Writes two compose YAMLs:
         1. The base compose (postgres, valkey, base image builders)
         2. The generated agents compose
        """
        # 1. Write the infrastructure / base components to the base path
        infra_dict = self._build_infra_compose_dict()
        self._write_yaml(infra_dict, self._base_path)

        # 2. Write the agents to the output path
        merged_config = {**self._seed_config, **self._dynamic_agents}
        agents_dict = self._build_agents_compose_dict(merged_config, refresh_repo)
        self._write_yaml(agents_dict, self._output_path)

        logger.info(
            f"Compose files written to {self._base_path} and {self._output_path} "
            f"({len(self._seed_config)} seed + {len(self._dynamic_agents)} dynamic agents)"
        )
        return self._output_path

    def get_all_agents(self) -> Dict[str, Any]:
        return {**self._seed_config, **self._dynamic_agents}

    def _build_infra_compose_dict(self) -> Dict[str, Any]:
        """Builds the infrastructure components (valkey, postgres, base images)."""
        services: Dict[str, Any] = {}

        services["base"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.base",
                "secrets": ["github_token"],
            },
            "image": "oai-adk-base-image:latest",
        }

        services["base-langgraph"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.framework_base",
                "args": {
                    "FRAMEWORK_PACKAGE": "oai-langgraph-core",
                    "FRAMEWORK_SUBDIR": "packages/langgraph-core",
                    "ADK_VERSION": "main"
                },
            },
            "image": "oai-adk-langgraph-base-image:latest",
            "depends_on": {
                "base": {"condition": "service_completed_successfully"},
            }
        }

        services["base-crewai"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.framework_base",
                "args": {
                    "FRAMEWORK_PACKAGE": "oai-crewai-core",
                    "FRAMEWORK_SUBDIR": "packages/crewai-core",
                    "ADK_VERSION": "main"
                },
            },
            "image": "oai-adk-crewai-base-image:latest",
            "depends_on": {
                "base": {"condition": "service_completed_successfully"},
            }
        }

        services["base-strands"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.framework_base",
                "args": {
                    "FRAMEWORK_PACKAGE": "oai-aws-strands-core",
                    "FRAMEWORK_SUBDIR": "packages/aws-strands-core",
                    "ADK_VERSION": "main"
                },
            },
            "image": "oai-adk-strands-base-image:latest",
            "depends_on": {
                "base": {"condition": "service_completed_successfully"},
            }
        }

        services["base-openai"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.framework_base",
                "args": {
                    "FRAMEWORK_PACKAGE": "oai-openai-core",
                    "FRAMEWORK_SUBDIR": "packages/openai-core",
                    "ADK_VERSION": "main"
                },
            },
            "image": "oai-adk-openai-base-image:latest",
            "depends_on": {
                "base": {"condition": "service_completed_successfully"},
            }
        }

        services["valkey"] = {
            "image": "valkey/valkey:latest",
            "container_name": "agent-valkey",
            "environment": {"REDIS_PASSWORD": "admin"},
            "networks": {
                "agent-server-network": {
                    "ipv4_address": '172.25.0.11'
                }
            },
            "extra_hosts": [
                "host.docker.internal:host-gateway"
            ],
            "command": "valkey-server --appendonly yes",
            "ports": ["6379:6379"],
            "volumes": ["valkey-agent-data:/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD", "redis-cli", "-a", "admin", "ping"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 5,
                "start_period": "15s",
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
            "networks": {
                "agent-server-network": {
                    "ipv4_address": '172.25.0.10'
                }
            },
            "extra_hosts": [
                "host.docker.internal:host-gateway"
            ],
            "ports": ["5432:5432"],
            "volumes": ["postgres_data:/var/lib/postgresql/data"],
            "restart": "unless-stopped",
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 5,
                "start_period": "15s",
            },
        }

        return {
            "name": "agent-registry",
            "services": services,
            "networks": {
                "agent-server-network": {
                    "driver": "bridge",
                    "ipam": {"config": [{"subnet": "172.25.0.0/16"}]},
                }
            },
            "volumes": {
                "valkey-agent-data": {},
                "postgres_data": {},
            },
            "secrets": {
                "github_token": {
                    "environment": "GITHUB_TOKEN"
                }
            }
        }

    def _build_agents_compose_dict(
        self, agent_config: Dict[str, Any], refresh_repo: bool = False
    ) -> Dict[str, Any]:
        """Builds the agent-specific compose dictionary."""
        services: Dict[str, Any] = {}

        for service_name, config in agent_config.items():
            source = config.get("source")
            if source is None or source == "":
                logger.warning(f"Skipping '{source}': no source defined.")
                continue
            port = config.get('port')
            self.used_ports.append(port)

            if port is None or port in self.used_ports:
                port = self.find_available_port()
                self.used_ports.append(port)
                config['port'] = port

            services[service_name] = _build_service(
                service_name, config, self._base_url, self._local_registry_url, refresh_repo
            )

        return {
            "name": "agent-registry",
            "services": services,
            "networks": {
                "agent-server-network": None
            }
        }

    def _run_compose_up_agent(self, agent_name: str, no_build: bool = False) -> str:
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",
            service_name,
        ]
        if not no_build:
            cmd.insert(-1, "--build")
            
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose up failed for '{agent_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose up failed for '{agent_name}': {result.stderr}"
            )

        logger.info(f"Agent '{agent_name}' deployed successfully.")
        return result.stdout

    def _run_compose_up(self) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",
            "--build",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose up failed :\n{result.stderr}")
            raise RuntimeError(
                f"docker compose up failed : {result.stderr}"
            )
        return result.stdout

    async def _run_compose_down(self):
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "down",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose down failed :\n{result.stderr}")
            raise RuntimeError(
                f"docker compose down failed : {result.stderr}"
            )
        return result.stdout

    def _run_compose_down_agent(self, agent_name: str) -> str:
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "down",
            service_name,
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose down failed for '{agent_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose down failed for '{agent_name}': {result.stderr}"
            )

        logger.info(f"Agent '{agent_name}' shutdown successfully.")
        return result.stdout

    def _run_compose_stop_agent(self, agent_name: str) -> str:
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "stop",
            service_name,
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose stop failed for '{agent_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose stop failed for '{agent_name}': {result.stderr}"
            )

        logger.info(f"Agent '{agent_name}' paused successfully.")
        return result.stdout

    @staticmethod
    def _write_yaml(data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        _yaml.dump(data, buf)
        path.write_text(buf.getvalue(), encoding="utf-8")
