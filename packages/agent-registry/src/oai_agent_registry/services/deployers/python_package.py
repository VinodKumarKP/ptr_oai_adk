import os
import sys
import socket
import asyncio
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator, IO

from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.utils.env_vars import get_common_agent_env

logger = logging.getLogger(__name__)

class PythonPackageDeployer(BaseDeployer):
    """
    Deployer that clones a git repository, creates a python virtual environment,
    installs dependencies, and runs the agent locally as a python subprocess.
    """

    def start_infra_services(self) -> None:
        pass

    def __init__(
        self,
        seed_config: Dict[str, Any],
        base_dir: str,
        agent_base_url: str,
        agent_local_registry_url: str,
    ):
        self._seed_config = dict(seed_config)
        self._dynamic_agents: Dict[str, Dict[str, Any]] = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = agent_base_url
        self._local_registry_url = agent_local_registry_url
        self.used_ports: List[int] = []
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.log_files: Dict[str, IO[Any]] = {}

    async def initialize(self) -> None:
        logger.info(f"PythonPackageDeployer initialized at {self.base_dir}")
        for agent_name in self._seed_config.keys():
            self._dynamic_agents[agent_name] = self._seed_config[agent_name]
            self._dynamic_agents[agent_name]['repo_name'] = self._get_repo_name(self._seed_config[agent_name]['source'])
            self._dynamic_agents[agent_name]['port'] = self.find_available_port()
            self.start_agent(agent_name)

    async def shutdown(self) -> None:
        for agent_name in list(self.running_processes.keys()):
            self.stop_agent(agent_name)
        logger.info("PythonPackageDeployer shutdown complete.")

    def find_available_port(self) -> int:
        all_agents = {**self._seed_config, **self._dynamic_agents}
        for config in all_agents.values():
            if isinstance(config, dict) and "port" in config and config["port"] is not None:
                self.used_ports.append(config["port"])

        port = 8000
        max_attempts = 10000

        for _ in range(max_attempts):
            if port not in self.used_ports and port not in [8080, 8081, 8082]:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", port))
                        return port
                except OSError:
                    pass
            port += 1

        raise RuntimeError(f"Could not find an available port starting from 8000")

    def _get_repo_name(self, source_url: str) -> str:
        name = source_url.split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name

    def image_exists(self, agent_name: str, version: str) -> bool:
        # Since this deployer clones repos and creates venvs, we check if the directory exists
        # Versioning would require checking out specific tags, which could be implemented in future.
        config = self._dynamic_agents.get(agent_name)
        if not config:
            return False
        repo_name = config.get("repo_name")
        if not repo_name:
            return False
        agent_dir = self.base_dir / repo_name
        return agent_dir.exists()

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
        output = []
        async for line in self.stream_deploy_agent(
            agent_name, source_url, framework, env, description, tags, port, current_version, refresh_repo, no_build
        ):
            output.append(line)
        return "".join(output)

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
        
        if port is None:
            port = self.find_available_port()
            self.used_ports.append(port)

        repo_name = self._get_repo_name(source_url)
        agent_dir = self.base_dir / repo_name

        self._dynamic_agents[agent_name] = {
            "port": port,
            "source": source_url,
            "framework": framework,
            "env": env or {},
            "repo_name": repo_name,
            "current_version": current_version
        }

        # 1. Clone or Pull
        if not agent_dir.exists():
            yield f"Cloning {source_url} to {agent_dir}...\n"
            cmd = ["git", "clone", source_url, str(agent_dir)]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await process.stdout.readline()
                if not line: break
                yield line.decode('utf-8', errors='replace')
            await process.wait()
            
            if current_version and current_version != "latest":
                yield f"Checking out version {current_version}...\n"
                cmd = ["git", "-C", str(agent_dir), "checkout", current_version]
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                await process.wait()
                
        elif refresh_repo and not no_build:
            yield f"Pulling latest from {source_url}...\n"
            cmd = ["git", "-C", str(agent_dir), "fetch", "--all"]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            await process.wait()
            
            target = current_version if current_version and current_version != "latest" else "origin/main"
            yield f"Checking out {target}...\n"
            cmd = ["git", "-C", str(agent_dir), "checkout", target]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await process.stdout.readline()
                if not line: break
                yield line.decode('utf-8', errors='replace')
            await process.wait()

        # 2. Create venv
        venv_dir = agent_dir / ".venv"
        if not venv_dir.exists() and not no_build:
            yield f"Creating virtual environment in {venv_dir}...\n"
            cmd = [sys.executable, "-m", "venv", str(venv_dir)]
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await process.stdout.readline()
                if not line: break
                yield line.decode('utf-8', errors='replace')
            await process.wait()

        # 3. Install requirements
        if not no_build:
            yield "Installing requirements...\n"
            pip_exe = venv_dir / "bin" / "pip"

            yield "Installing uv...\n"
            uv_install_cmd = [str(pip_exe), "install", "uv"]
            process = await asyncio.create_subprocess_exec(*uv_install_cmd, stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await process.stdout.readline()
                if not line: break
                yield line.decode('utf-8', errors='replace')
            await process.wait()

            yield "Installing requirements via uv...\n"
            uv_exe = venv_dir / "bin" / "uv"
            cmd = [str(uv_exe), "pip", "install", "-r", str(agent_dir / "requirements.txt")]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            while True:
                line = await process.stdout.readline()
                if not line: break
                decoded = line.decode('utf-8', errors='replace')
                # Log the output to the main logger
                logger.info(f"[{agent_name} pip] {decoded.strip()}")
                # Yield it back to the stream
                yield decoded
            await process.wait()

        # 4. Start agent
        yield f"Starting Python Package Agent '{agent_name}' on port {port}...\n"
        try:
            self.start_agent(agent_name)
            yield f"\n✅ Agent '{agent_name}' deployed and started successfully!\n"
        except Exception as e:
            yield f"\n❌ Failed to start agent '{agent_name}': {str(e)}\n"
            raise

    def start_agent(self, agent_name: str) -> str:
        if agent_name in self.running_processes:
            return "Already running"

        config = self._dynamic_agents.get(agent_name)
        if not config:
            raise ValueError(f"Agent '{agent_name}' configuration not found in Python Package Deployer.")

        repo_name = config["repo_name"]
        port = config["port"]
        agent_dir = self.base_dir / repo_name
        venv_bin = agent_dir / ".venv" / "bin"
        venv_python = venv_bin / "python"
        
        if not venv_python.exists():
            # Fallback to system python if venv wasn't created properly
            venv_python = Path(sys.executable)

        env = os.environ.copy()
        
        # Activate virtual environment in the subprocess by prepending its bin to PATH
        if venv_bin.exists():
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            env["VIRTUAL_ENV"] = str(agent_dir / ".venv")
        
        agent_env = get_common_agent_env(
            agent_name=agent_name,
            port=port,
            base_url=self._base_url,
            local_registry_url=self._local_registry_url,
            env_overrides=config.get("env", {}),
            deployment_mode='python_package'
        )
        
        for k, v in agent_env.items():
            env[k] = str(v)

        server_py = agent_dir / "agentic_registry_agents" / "agents" / agent_name / "server.py"
        
        if not server_py.exists():
            # Sometimes the repo structure might just be server.py in root or differently nested.
            # Look for server.py in root as a fallback.
            fallback_py = agent_dir / "server.py"
            if fallback_py.exists():
                server_py = fallback_py

        cmd = [str(venv_python), str(server_py), "--port", str(port)]
        logger.info(f"Starting python package agent: {' '.join(cmd)}")

        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"{agent_name}.log"
        
        log_file = open(log_file_path, "a")
        self.log_files[agent_name] = log_file

        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        self.running_processes[agent_name] = proc
        return "Started"

    def stop_agent(self, agent_name: str) -> str:
        proc = self.running_processes.get(agent_name)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            del self.running_processes[agent_name]
            
            log_file = self.log_files.get(agent_name)
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
                del self.log_files[agent_name]

            return "Stopped"
        return "Not running"

    def remove_agent(self, agent_name: str) -> str:
        self.stop_agent(agent_name)
        if agent_name in self._dynamic_agents:
            del self._dynamic_agents[agent_name]
        return "Removed"
