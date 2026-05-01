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
from typing import Any, Dict, List, Optional

from ruamel.yaml import YAML

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
    """
    Finds the next available port starting from start_port.
    Skips ports specified in skip_ports and any ports that are already used.

    Args:
        used_ports: List of port numbers already in use by known agents.
        start_port: Port number to start searching from (default: 8000).
        skip_ports: List of port numbers to skip (default: [8080, 8081, 8082]).

    Returns:
        The first available port number.
    """
    if skip_ports is None:
        skip_ports = [8080, 8081, 8082]

    used_ports = used_ports or []
    reserved_ports = set(skip_ports + used_ports)

    port = start_port
    max_attempts = 10000  # Prevent infinite loops

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
    environment: List[str] = [f"AGENT_NAME={service_name}"]

    # Caller-supplied env vars (from the agent's seed/registration config)
    for key, value in env_overrides.items():
        value_str = str(value)
        if value_str.startswith("${") and value_str.endswith("}") and ":-" in value_str:
            environment.append(f"{key}={value_str}")
        elif "$" in value_str and ("{" in value_str or " " not in value_str):
            var_name = value_str.replace("${", "").replace("}", "").replace("$", "")
            environment.append(f"{key}=${{{var_name}:-$${var_name}}}")
        else:
            environment.append(f"{key}={value_str}")

    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    environment += [
        f"AWS_REGION={aws_region}",
        "REDIS_HOST=agent-valkey",
        "REDIS_PORT=6379",
        "AGENT_AUTH_ENABLED=true",
        "LOGGING_DB_HOST=agent_logs_db",
        "LOGGING_DB_PORT=5432",
        "LOGGING_DB_USER=postgres",
        "LOGGING_DB_PASSWORD=postgres",
        "LOGGING_DB_NAME=agent_logs",
        "DB_LOGGING_ENABLED=true",
        "DB_POOL_MAX_SIZE=2",
        "DB_POOL_TIMEOUT=60",
        "DB_POOL_MIN_SIZE=1",
        f"AGENT_BASE_URL={base_url}",
        f"AGENT_LOCAL_REGISTRY_URL={local_registry_url}",
    ]
    return environment


def _build_service(service_name: str, config: Dict[str, Any],
                   base_url: str, local_registry_url: str,
                   refresh_repo: bool = False) -> Dict[str, Any]:
    """
    Produces a single Compose service dict for an agent.
    `config` is the per-agent dict from the seed / DB record:
        {
            "port":        8010,
            "source":      "https://github.com/org/repo",
            "framework":   "crewai",          # optional
            "tags":        ["crewai"],         # optional
            "env":         {"MY_VAR": "val"}, # optional
            "description": "...",             # optional
        }
    """
    port = config["port"]
    tags = config.get("tags", [])
    framework = config.get("framework", "")

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

class DockerComposeManager:
    """
    Manages a dynamically generated docker-compose file for agent services.

    Usage in AgentRegistry.register_agent():

        # At startup — pass the seed config that your existing generator
        # already loads from YAML/JSON files on disk.
        self.compose_manager = DockerComposeManager(
            seed_config=server_config,          # Dict[str, agent-config-dict]
            compose_output_path="docker-compose.generated.yaml",
            base_compose_path="docker-compose.yaml",
            agent_base_url="http://192.168.1.132:8081",
            agent_local_registry_url="http://host.docker.internal:8081",
        )

        # On /register — after saving to DB:
        await self.compose_manager.deploy_agent(agent_name, agent_config_dict)
    """

    def __init__(
        self,
        seed_config: Dict[str, Any],
        compose_output_path: str = "docker-compose.generated.yaml",
        base_compose_path: str = "docker-compose.yaml",
        agent_base_url: str = "http://192.168.1.132:8081",
        agent_local_registry_url: str = "http://host.docker.internal:8081",
    ):
        """
        Args:
            seed_config:
                The static agent config already loaded from disk
                (same dict your existing `load_server_config()` returns).
                Agents here always appear in the generated compose.

            compose_output_path:
                Where to write the generated compose file.  Relative paths
                are resolved from the current working directory.

            base_compose_path:
                Your infra compose file (valkey, postgres, base image, etc.).
                Passed as the first `-f` arg to `docker compose` so its
                networks and volumes are always visible.

            agent_base_url / agent_local_registry_url:
                Injected as env vars into every agent container.
        """
        self._seed_config: Dict[str, Any] = dict(seed_config)
        # Dynamic agents added via /register are kept separately so we can
        # tell them apart from seed agents if needed.
        self._dynamic_agents: Dict[str, Any] = {}

        self._output_path = Path(compose_output_path)
        self._base_path = Path(base_compose_path)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url

    # ------------------------------------------------------------------
    # Public API
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
    ) -> Dict[str, Any]:
        """
        Converts an AgentRegistration record into the seed-config shape and
        stores it so the next compose regeneration includes it.

        If port is not provided, automatically assigns the next available port
        starting from 8000 and skipping 8080, 8081, 8082.

        Returns the config dict that was stored (useful for logging / testing).
        """
        tags = tags or []
        if framework and framework.lower() not in tags:
            tags.append(framework.lower())

        # Auto-assign port if not provided
        if port is None:
            port = self._find_available_port()
            logger.info(f"Auto-assigned port {port} to agent '{agent_name}'")

        config = {
            "port": port,
            "source": source_url,
            "framework": framework or "",
            "tags": tags,
            "env": env or {},
            "description": description or "",
        }
        self._dynamic_agents[agent_name] = config
        logger.debug(f"Staged dynamic agent '{agent_name}' for next compose generation.")
        return config

    def remove_agent(self, agent_name: str) -> bool:
        """
        Removes a dynamic agent from future compose generations.
        Returns True if the agent was found and removed, False otherwise.
        Note: seed-config agents cannot be removed this way.
        """
        if agent_name in self._dynamic_agents:
            del self._dynamic_agents[agent_name]
            logger.info(f"Removed dynamic agent '{agent_name}' from compose manager.")
            return True
        logger.warning(f"Agent '{agent_name}' not found in dynamic agents (may be a seed agent).")
        return False

    async def deploy_agent(
        self,
        agent_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        refresh_repo: bool = False,
    ) -> str:
        """
        Full pipeline for a newly registered agent:
          1. Stage the agent config (auto-assigning port if not provided).
          2. Regenerate the compose file (all seed + all dynamic agents).
          3. Run `docker compose up -d --no-deps --build <service>`.

        If port is not provided, automatically assigns the next available port
        starting from 8000 and skipping 8080, 8081, 8082.

        Returns the stdout from docker compose.
        Raises RuntimeError if docker compose exits with a non-zero code.
        """
        self.add_agent_from_registration(
            agent_name=agent_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
        )
        self.write_compose_file(refresh_repo=refresh_repo)
        return self._run_compose_up_agent(agent_name)

    def write_compose_file(self, refresh_repo: bool = False) -> Path:
        """
        Regenerates the compose YAML from the current seed + dynamic agents.
        Returns the path of the written file.
        """
        merged_config = {**self._seed_config, **self._dynamic_agents}
        compose_dict = self._build_compose_dict(merged_config, refresh_repo)
        self._write_yaml(compose_dict, self._output_path)
        logger.info(
            f"Compose file written to {self._output_path} "
            f"({len(self._seed_config)} seed + {len(self._dynamic_agents)} dynamic agents)"
        )
        return self._output_path

    def get_all_agents(self) -> Dict[str, Any]:
        """Returns a merged view of seed + dynamic agent configs."""
        return {**self._seed_config, **self._dynamic_agents}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_available_port(self) -> int:
        """
        Finds the next available port considering all agents in both
        seed and dynamic configurations.
        Starts from 8000 and skips 8080, 8081, 8082.
        """
        used_ports = []
        all_agents = {**self._seed_config, **self._dynamic_agents}
        for agent_config in all_agents.values():
            if isinstance(agent_config, dict) and "port" in agent_config:
                used_ports.append(agent_config["port"])

        return _get_next_available_port(used_ports=used_ports)

    def _build_compose_dict(
        self, agent_config: Dict[str, Any], refresh_repo: bool = False
    ) -> Dict[str, Any]:
        """Builds the full compose dict (mirrors generate_docker_compose())."""
        services: Dict[str, Any] = {}

        # Base image builder — always present
        services["base"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.base",
                "args": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
            },
            "image": "oai-adk-base-image:latest",
        }

        # Base image builder — always present
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
        }

        # Agent services
        for service_name, config in agent_config.items():
            source = config.get("source")
            if source is None or source == "":
                logger.warning(f"Skipping '{source}': no source defined.")
                continue
            services[service_name] = _build_service(
                service_name, config, self._base_url, self._local_registry_url, refresh_repo
            )

        # Infra: valkey
        services["valkey"] = {
            "image": "valkey/valkey:latest",
            "container_name": "agent-valkey",
            "environment": {"REDIS_PASSWORD": "admin"},
            "networks": ["agent-server-network"],
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

        # Infra: postgres
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
            "networks": ["agent-server-network"],
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

        # agent-proxy depends on every agent service being healthy
        # proxy_depends: Dict[str, Any] = {
        #     "valkey": {"condition": "service_healthy"}
        # }
        # for service_name in agent_config:
        #     proxy_depends[service_name] = {"condition": "service_healthy"}
        #
        # services["agent-proxy"] = {
        #     "build": {
        #         "context": ".",
        #         "dockerfile": "Dockerfile.proxy",
        #         "args": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
        #     },
        #     "container_name": "agent-proxy",
        #     "network_mode": "host",
        #     "restart": "unless-stopped",
        #     "ports": ["8081:8081"],
        #     "command": (
        #         "oai-agent-registry --port 8081 --start-port 8001 "
        #         "--end-port 8200 --enable-auto-discovery"
        #     ),
        #     "volumes": ["./logs:/tmp"],
        #     "depends_on": proxy_depends,
        #     "environment": {"AGENT_BASE_URL": self._base_url.rsplit(":", 1)[0]},
        # }

        return {
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
        }

    def _run_compose_up_agent(self, agent_name: str) -> str:
        """
        Runs `docker compose up -d --no-deps --build <service>` for the
        single named agent service without disturbing other containers.
        """
        # service_name = agent_name.replace("_", "-")
        service_name = agent_name
        cmd = [
            "docker", "compose",
            # "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",    # don't restart valkey/postgres/etc.
            "--build",      # build the image if not cached
            service_name,
        ]
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
        """
        Runs `docker compose up -d --no-deps --build`
        """
        cmd = [
            "docker", "compose",
            # "-f", str(self._base_path),    # infra (networks, volumes)
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",    # don't restart valkey/postgres/etc.
            "--build",      # build the image if not cached
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
        """
        Runs `docker compose down`
        """
        cmd = [
            "docker", "compose",
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
        """
        Stops and removes the agent's container using docker compose down.
        This ensures a clean shutdown, not just a pause.
        """
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._output_path),
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
        """
        Pauses the agent's container using docker compose stop (doesn't remove).
        Use this if you want to keep the container artifact but pause execution.
        """
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._output_path),
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

    async def stream_deploy_agent(
        self,
        agent_name: str,
        source_url: str,
        framework: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        port: Optional[int] = None,
        refresh_repo: bool = False,
    ):
        """
        Streams the output of the full deployment pipeline for a newly registered agent.
        Yields output lines in real-time for UI consumption.

        Pipeline:
          1. Stage the agent config (auto-assigning port if not provided).
          2. Regenerate the compose file (all seed + all dynamic agents).
          3. Run `docker compose up -d --no-deps --build <service>` with streaming output.

        If port is not provided, automatically assigns the next available port
        starting from 8000 and skipping 8080, 8081, 8082.

        Yields:
            str: Lines of output from the docker compose command
        """
        # Stage the agent config
        self.add_agent_from_registration(
            agent_name=agent_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
        )
        yield f"Staged agent '{agent_name}' for deployment\n"

        # Regenerate compose file
        self.write_compose_file(refresh_repo=refresh_repo)
        yield f"Generated docker-compose file\n"

        # Stream the docker compose up command
        service_name = agent_name
        cmd = [
            "docker", "compose",
            "-f", str(self._output_path),  # generated agents
            "up", "-d",
            "--no-deps",    # don't restart valkey/postgres/etc.
            "--build",      # build the image if not cached
            service_name,
        ]

        yield f"Starting docker compose build and deployment...\n"
        yield f"Command: {' '.join(cmd)}\n\n"

        try:
            # Use asyncio subprocess for streaming with unbuffered output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Combine stdout and stderr
                env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Unbuffered output
            )

            # Read output line by line with smaller buffer
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                # Decode and yield immediately
                decoded_line = line.decode('utf-8', errors='replace')
                yield decoded_line
                # Force immediate yield (though asyncio should handle this)
                await asyncio.sleep(0)

            # Wait for process to complete
            return_code = await process.wait()

            if return_code != 0:
                yield f"\n❌ Deployment failed with exit code {return_code}\n"
                raise RuntimeError(f"docker compose up failed for '{agent_name}' with exit code {return_code}")
            else:
                yield f"\n✅ Agent '{agent_name}' deployed successfully!\n"

        except Exception as e:
            yield f"\n❌ Error during deployment: {str(e)}\n"
            raise

    @staticmethod
    def _write_yaml(data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        _yaml.dump(data, buf)
        path.write_text(buf.getvalue(), encoding="utf-8")
