"""
oai_platform_core.deployers.python_package_base — shared Python-package deployer.

Both the agent-registry and mcp-registry ship a ``PythonPackageDeployer``
that clones a Git repository, creates a virtualenv, installs dependencies
with uv, and runs the service as a subprocess.

The two implementations are >95 % identical; they differ only in:

* The env-var builder (``get_common_agent_env`` / ``get_common_server_env``)
* The ``server.py`` path convention inside the repository
* The subprocess startup command (MCP adds ``--transport streamable-http``)
* The internal dict name (``_dynamic_agents`` / ``_dynamic_servers``)

This module provides :class:`BasePythonPackageDeployer` which implements
all shared logic.  Subclasses provide three abstract hooks:

``_get_service_env()``
    Returns the env-var dict for the spawned subprocess.

``_get_server_py_path()``
    Returns the ``Path`` to ``server.py`` inside the cloned repo.

``_get_start_command()``
    Returns the full ``argv`` list for ``subprocess.Popen``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, IO, List, Optional

from oai_platform_core.deployers.base import BaseDeployer

logger = logging.getLogger(__name__)


class BasePythonPackageDeployer(BaseDeployer):
    """Shared Python-package deployer for OAI registry packages.

    Manages git clone/pull, venv creation, uv-based dependency
    installation, and subprocess lifecycle for both agent and MCP server
    deployments.
    """

    def __init__(
        self,
        seed_config: Dict[str, Any],
        base_dir: str,
        agent_base_url: str,
        agent_local_registry_url: str,
    ) -> None:
        self._seed_config = dict(seed_config)
        self._dynamic_services: Dict[str, Dict[str, Any]] = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url
        self.used_ports: List[int] = []
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.log_files: Dict[str, IO[Any]] = {}

    # ------------------------------------------------------------------
    # BaseDeployer — lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        logger.info("PythonPackageDeployer initialized at %s", self.base_dir)
        for service_name in self._seed_config:
            cfg = dict(self._seed_config[service_name])
            cfg["repo_name"] = self._get_repo_name(cfg["source"])
            cfg["port"] = self.find_available_port()
            self._dynamic_services[service_name] = cfg
            self._start_service(service_name)

    async def shutdown(self) -> None:
        for service_name in list(self.running_processes):
            self._stop_service(service_name)
        logger.info("PythonPackageDeployer shutdown complete.")

    def find_available_port(self) -> int:
        all_services = {**self._seed_config, **self._dynamic_services}
        for config in all_services.values():
            if isinstance(config, dict) and config.get("port") is not None:
                self.used_ports.append(config["port"])

        port = 8000
        for _ in range(10_000):
            if port not in self.used_ports and port not in (8080, 8081, 8082):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", port))
                        return port
                except OSError:
                    pass
            port += 1

        raise RuntimeError("Could not find an available port starting from 8000")

    def image_exists(self, service_name: str, version: str) -> bool:
        """Return ``True`` if the service's local repo directory exists."""
        config = self._dynamic_services.get(service_name)
        if not config:
            return False
        repo_name = config.get("repo_name")
        if not repo_name:
            return False
        return (self.base_dir / repo_name).exists()

    # ------------------------------------------------------------------
    # Shared deploy helpers
    # ------------------------------------------------------------------

    async def _deploy_service(
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
    ) -> str:
        """Non-streaming deploy: collect all stream lines and return them."""
        output: List[str] = []
        async for line in self._stream_deploy_service(
            service_name, source_url, framework, env, description,
            tags, port, current_version, refresh_repo, no_build,
        ):
            output.append(line)
        return "".join(output)

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
        """Async generator: clone/pull, install deps, start service."""
        if port is None:
            port = self.find_available_port()
            self.used_ports.append(port)

        repo_name = self._get_repo_name(source_url)
        service_dir = self.base_dir / repo_name

        self._dynamic_services[service_name] = {
            "port": port,
            "source": source_url,
            "framework": framework,
            "env": env or {},
            "repo_name": repo_name,
            "current_version": current_version,
        }

        # 1. Clone or pull
        if not service_dir.exists():
            yield f"Cloning {source_url} to {service_dir}...\n"
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", source_url, str(service_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in _stream_proc(proc):
                yield line

            if current_version and current_version != "latest":
                yield f"Checking out version {current_version}...\n"
                proc = await asyncio.create_subprocess_exec(
                    "git", "-C", str(service_dir), "checkout", current_version,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await proc.wait()

        elif refresh_repo and not no_build:
            yield f"Pulling latest from {source_url}...\n"
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(service_dir), "fetch", "--all",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await proc.wait()

            target = (
                current_version
                if current_version and current_version != "latest"
                else "origin/main"
            )
            yield f"Checking out {target}...\n"
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(service_dir), "checkout", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in _stream_proc(proc):
                yield line

        # 2. Create venv
        venv_dir = service_dir / ".venv"
        if not venv_dir.exists() and not no_build:
            yield f"Creating virtual environment in {venv_dir}...\n"
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "venv", str(venv_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in _stream_proc(proc):
                yield line

        # 3. Install deps
        if not no_build:
            pip_exe = venv_dir / "bin" / "pip"
            uv_exe = venv_dir / "bin" / "uv"

            yield "Installing uv...\n"
            proc = await asyncio.create_subprocess_exec(
                str(pip_exe), "install", "uv",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in _stream_proc(proc):
                yield line

            yield "Installing requirements via uv...\n"
            req_file = str(service_dir / "requirements.txt")
            cmd = [str(uv_exe), "pip", "install", "-r", req_file]
            logger.info("Running command: %s", " ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in _stream_proc(proc):
                logger.info("[%s uv] %s", service_name, line.rstrip())
                yield line

        # 4. Start service
        yield f"Starting '{service_name}' on port {port}...\n"
        try:
            self._start_service(service_name)
            yield f"\n✅ '{service_name}' deployed and started successfully!\n"
        except Exception as exc:
            yield f"\n❌ Failed to start '{service_name}': {exc}\n"
            raise

    # ------------------------------------------------------------------
    # Internal start / stop / remove (generic, use these from subclasses)
    # ------------------------------------------------------------------

    def _start_service(self, service_name: str) -> str:
        """Start the service subprocess.  Called by ``start_agent`` / ``start_server``."""
        if service_name in self.running_processes:
            return "Already running"

        config = self._dynamic_services.get(service_name)
        if not config:
            raise ValueError(
                f"'{service_name}' configuration not found in PythonPackageDeployer."
            )

        repo_name = config["repo_name"]
        port = config["port"]
        service_dir = self.base_dir / repo_name
        venv_bin = service_dir / ".venv" / "bin"
        venv_python = venv_bin / "python"

        if not venv_python.exists():
            venv_python = Path(sys.executable)

        env = os.environ.copy()
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(service_dir / ".venv")

        service_env = self._get_service_env(
            service_name=service_name,
            port=port,
            base_url=self._base_url,
            local_registry_url=self._local_registry_url,
            env_overrides=config.get("env", {}),
            deployment_mode="python_package",
        )
        env.update({k: str(v) for k, v in service_env.items()})

        server_py = self._get_server_py_path(service_dir, service_name)
        cmd = self._get_start_command(venv_python, server_py, port)
        logger.info("Starting service subprocess: %s", " ".join(str(c) for c in cmd))

        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f"{service_name}.log", "a")
        self.log_files[service_name] = log_file

        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        self.running_processes[service_name] = proc
        return "Started"

    def _stop_service(self, service_name: str) -> str:
        """Terminate the service subprocess."""
        proc = self.running_processes.get(service_name)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self.running_processes[service_name]

            log_file = self.log_files.pop(service_name, None)
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
            return "Stopped"
        return "Not running"

    def _remove_service(self, service_name: str) -> str:
        """Stop and deregister the service."""
        self._stop_service(service_name)
        self._dynamic_services.pop(service_name, None)
        return "Removed"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _get_repo_name(source_url: str) -> str:
        name = source_url.split("/")[-1]
        return name[:-4] if name.endswith(".git") else name

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_service_env(
        self,
        service_name: str,
        port: int,
        base_url: str,
        local_registry_url: str,
        env_overrides: Optional[Dict[str, Any]],
        deployment_mode: str,
    ) -> Dict[str, str]:
        """Return the env-var dict for the service subprocess."""

    @abstractmethod
    def _get_server_py_path(self, service_dir: Path, service_name: str) -> Path:
        """Return the path to ``server.py`` inside *service_dir*."""

    @abstractmethod
    def _get_start_command(
        self, venv_python: Path, server_py: Path, port: int
    ) -> List[str]:
        """Return the full ``argv`` list for ``subprocess.Popen``."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _stream_proc(proc: asyncio.subprocess.Process) -> AsyncGenerator[str, None]:
    """Yield decoded lines from *proc* stdout until EOF, then await finish."""
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        yield line.decode("utf-8", errors="replace")
    await proc.wait()
