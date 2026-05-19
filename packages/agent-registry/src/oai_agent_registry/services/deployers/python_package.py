"""
Agent-registry PythonPackageDeployer.

Extends :class:`oai_platform_core.deployers.python_package_base.BasePythonPackageDeployer`
with agent-specific hooks (env-var builder, ``server.py`` path convention,
subprocess startup command) and satisfies the
:class:`oai_agent_registry.services.deployers.base.BaseDeployer` abstract
contract (``deploy_agent``, ``stream_deploy_agent``, ``start_agent``, …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.python_package_base import BasePythonPackageDeployer

from oai_agent_registry.services.deployers.base import BaseDeployer
from oai_agent_registry.utils.env_vars import get_common_agent_env


class PythonPackageDeployer(BasePythonPackageDeployer, BaseDeployer):  # type: ignore[misc]
    """Python-package deployer for the agent registry.

    Clones a Git repository, creates a virtualenv, installs dependencies
    with uv, and runs the agent as a subprocess.
    """

    # ------------------------------------------------------------------
    # Abstract hook — environment vars
    # ------------------------------------------------------------------

    def _get_service_env(
        self,
        service_name: str,
        port: int,
        base_url: str,
        local_registry_url: str,
        env_overrides: Optional[Dict[str, Any]],
        deployment_mode: str,
    ) -> Dict[str, str]:
        return get_common_agent_env(
            agent_name=service_name,
            port=port,
            base_url=base_url,
            local_registry_url=local_registry_url,
            env_overrides=env_overrides or {},
            deployment_mode=deployment_mode,
        )

    # ------------------------------------------------------------------
    # Abstract hook — server.py path inside the cloned repo
    # ------------------------------------------------------------------

    def _get_server_py_path(self, service_dir: Path, service_name: str) -> Path:
        primary = (
            service_dir / "agentic_registry_agents" / "agents" / service_name / "server.py"
        )
        if primary.exists():
            return primary
        # Fallback: server.py at repo root
        return service_dir / "server.py"

    # ------------------------------------------------------------------
    # Abstract hook — subprocess startup command
    # ------------------------------------------------------------------

    def _get_start_command(
        self, venv_python: Path, server_py: Path, port: int
    ) -> List[str]:
        return [str(venv_python), str(server_py), "--port", str(port)]

    # ------------------------------------------------------------------
    # BaseDeployer abstract methods — agent-specific names
    # ------------------------------------------------------------------

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
        return await self._deploy_service(
            agent_name, source_url, framework, env, description,
            tags, port, current_version, refresh_repo, no_build,
        )

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
        return self._start_service(agent_name)

    def stop_agent(self, agent_name: str) -> str:
        return self._stop_service(agent_name)

    def remove_agent(self, agent_name: str) -> str:
        return self._remove_service(agent_name)

    def start_infra_services(self) -> None:
        """No-op: the Python-package deployer has no Docker infra to start."""
