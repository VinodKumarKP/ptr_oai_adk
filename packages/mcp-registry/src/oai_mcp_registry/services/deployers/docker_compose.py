"""
DockerComposeManager for MCP
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

from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.utils.env_vars import get_common_server_env

logger = logging.getLogger(__name__)

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.indent(mapping=2, sequence=4, offset=2)
_yaml.width = 1000

def _aws_volume() -> Optional[str]:
    aws_path = os.path.expanduser("~/.aws")
    if os.path.isdir(aws_path):
        return f"{aws_path}:/home/mcpuser/.aws:ro"
    return None

def _is_port_available(port: int) -> bool:
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
    env_dict = get_common_server_env(
        server_name=service_name,
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
    current_version = config.get("current_version")
    image_tag = current_version if current_version else "latest"
    image_name = f"{service_name.replace('_', '-')}:{image_tag}"

    # For MCP servers, we primarily use the mcp base image
    base_image = 'oai-adk-mcp-base-image:latest'

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
        "container_name": f"mcp-{service_name.replace('_', '-')}",
        "ports": [f"{port}:{port}"],
        "volumes": volumes,
        "environment": _build_environment(
            service_name, port, env_overrides, base_url, local_registry_url
        ),
        "command": f"--port {port} --transport streamable-http",
        "restart": "unless-stopped",
        "networks": ["mcp-server-network"],
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

class DockerComposeManager(BaseDeployer):
    """
    Manages a dynamically generated docker-compose file for MCP services.
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
        self._dynamic_servers: Dict[str, Any] = {}

        self._output_path = Path(compose_output_path)
        self._base_path = Path(base_compose_path)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url
        self.used_ports = []

    async def initialize(self) -> None:
        self.write_compose_file()
        self._run_compose_up()

    async def shutdown(self) -> None:
        await self._run_compose_down()

    def find_available_port(self) -> int:
        all_servers = {**self._seed_config, **self._dynamic_servers}
        for server_config in all_servers.values():
            if isinstance(server_config, dict) and "port" in server_config:
                self.used_ports.append(server_config["port"])

        return _get_next_available_port(used_ports=self.used_ports)

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
        tags = tags or []
        if framework and framework.lower() not in tags:
            tags.append(framework.lower())

        if port is None:
            port = self.find_available_port()
            self.used_ports.append(port)
            logger.info(f"Auto-assigned port {port} to server '{server_name}'")

        config = {
            "port": port,
            "source": source_url,
            "framework": framework or "",
            "tags": tags,
            "env": env or {},
            "description": description or "",
            "current_version": current_version,
        }
        self._dynamic_servers[server_name] = config
        logger.debug(f"Staged dynamic server '{server_name}' for next compose generation.")
        return config

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
        self.add_server_from_registration(
            server_name=server_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
            current_version=current_version,
        )
        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        return self._run_compose_up_server(server_name, no_build=no_build)

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
        self.add_server_from_registration(
            server_name=server_name,
            source_url=source_url,
            framework=framework,
            env=env,
            description=description,
            tags=tags,
            port=port,
            current_version=current_version,
        )
        yield f"Staged server '{server_name}' for deployment\n"

        self.write_compose_file(refresh_repo=False if no_build else refresh_repo)
        yield f"Generated docker-compose files\n"

        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "up", "-d",
            "--no-deps",
            server_name,
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
                raise RuntimeError(f"docker compose up failed for '{server_name}' with exit code {return_code}")
            else:
                yield f"\n✅ Server '{server_name}' deployed successfully!\n"

        except Exception as e:
            yield f"\n❌ Error during deployment: {str(e)}\n"
            raise

    def start_server(self, server_name: str) -> str:
        return self._run_compose_up_server(server_name)

    def stop_server(self, server_name: str) -> str:
        return self._run_compose_stop_server(server_name)

    def remove_server(self, server_name: str) -> str:
        if server_name in self._dynamic_servers:
            del self._dynamic_servers[server_name]
        return self._run_compose_down_server(server_name)

    def image_exists(self, server_name: str, version: str) -> bool:
        service_image = f"{server_name.replace('_', '-')}:{version}"
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", service_image],
            capture_output=True,
            text=True,
        )
        exists = result.returncode == 0 and bool(result.stdout.strip())
        logger.debug(f"Image '{service_image}' {'found' if exists else 'not found'} in local store.")
        return exists

    def write_compose_file(self, refresh_repo: bool = False) -> Path:
        infra_dict = self._build_infra_compose_dict()
        self._write_yaml(infra_dict, self._base_path)

        merged_config = {**self._seed_config, **self._dynamic_servers}
        servers_dict = self._build_servers_compose_dict(merged_config, refresh_repo)
        self._write_yaml(servers_dict, self._output_path)

        logger.info(
            f"Compose files written to {self._base_path} and {self._output_path} "
            f"({len(self._seed_config)} seed + {len(self._dynamic_servers)} dynamic servers)"
        )
        return self._output_path

    def get_all_servers(self) -> Dict[str, Any]:
        return {**self._seed_config, **self._dynamic_servers}

    def _build_infra_compose_dict(self) -> Dict[str, Any]:
        services: Dict[str, Any] = {}

        services["base"] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile.base",
                "secrets": ["github_token"],
            },
            "image": "oai-adk-mcp-base-image:latest",
        }

        services["valkey"] = {
            "image": "valkey/valkey:latest",
            "container_name": "mcp-valkey",
            "environment": {"REDIS_PASSWORD": "admin"},
            "networks": {
                "mcp-server-network": {
                    "ipv4_address": '172.26.0.11'
                }
            },
            "command": "valkey-server --appendonly yes",
            "ports": ["6379:6379"],
            "volumes": ["valkey-mcp-data:/data"],
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
            "networks": {
                "mcp-server-network": {
                    "ipv4_address": '172.26.0.10'
                }
            },
            "ports": ["5432:5432"],
            "volumes": ["mcp_postgres_data:/var/lib/postgresql/data"],
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
            "services": services,
            "networks": {
                "mcp-server-network": {
                    "driver": "bridge",
                    "ipam": {"config": [{"subnet": "172.26.0.0/16"}]},
                }
            },
            "volumes": {
                "valkey-mcp-data": {},
                "mcp_postgres_data": {},
            },
            "secrets": {
                "github_token": {
                    "environment": "GITHUB_TOKEN"
                }
            }
        }

    def _build_servers_compose_dict(
        self, server_config: Dict[str, Any], refresh_repo: bool = False
    ) -> Dict[str, Any]:
        services: Dict[str, Any] = {}

        for service_name, config in server_config.items():
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
            "services": services,
            "networks": {
                "mcp-server-network": None
            }
        }

    def _run_compose_up_server(self, server_name: str, no_build: bool = False) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "up", "-d",
            "--no-deps",
            server_name,
        ]
        if not no_build:
            cmd.insert(-1, "--build")
            
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose up failed for '{server_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose up failed for '{server_name}': {result.stderr}"
            )

        logger.info(f"Server '{server_name}' deployed successfully.")
        return result.stdout

    def _run_compose_up(self) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
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
            "-f", str(self._base_path),
            "-f", str(self._output_path),
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

    def _run_compose_down_server(self, server_name: str) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "down",
            server_name,
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose down failed for '{server_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose down failed for '{server_name}': {result.stderr}"
            )

        logger.info(f"Server '{server_name}' shutdown successfully.")
        return result.stdout

    def _run_compose_stop_server(self, server_name: str) -> str:
        cmd = [
            "docker", "compose",
            "-f", str(self._base_path),
            "-f", str(self._output_path),
            "stop",
            server_name,
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"docker compose stop failed for '{server_name}':\n{result.stderr}")
            raise RuntimeError(
                f"docker compose stop failed for '{server_name}': {result.stderr}"
            )

        logger.info(f"Server '{server_name}' paused successfully.")
        return result.stdout

    @staticmethod
    def _write_yaml(data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        _yaml.dump(data, buf)
        path.write_text(buf.getvalue(), encoding="utf-8")
