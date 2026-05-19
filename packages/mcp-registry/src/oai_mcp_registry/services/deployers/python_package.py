"""
MCP-registry PythonPackageDeployer.

Extends :class:`oai_platform_core.deployers.python_package_base.BasePythonPackageDeployer`
with MCP-server-specific hooks (env-var builder, ``server.py`` path convention,
subprocess startup command with ``--transport streamable-http``) and satisfies
the :class:`oai_mcp_registry.services.deployers.base.BaseDeployer` abstract
contract (``deploy_server``, ``stream_deploy_server``, ``start_server``, …).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.python_package_base import BasePythonPackageDeployer

from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.utils.env_vars import get_common_server_env


class PythonPackageDeployer(BasePythonPackageDeployer, BaseDeployer):  # type: ignore[misc]
    """Python-package deployer for the MCP registry.

    Clones a Git repository, creates a virtualenv, installs dependencies
    with uv, and runs the MCP server as a subprocess.
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
        return get_common_server_env(
            server_name=service_name,
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
            service_dir / "mcp_registry_servers" / "servers" / service_name / "server.py"
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
        return [
            str(venv_python), str(server_py),
            "--port", str(port),
            "--transport", "streamable-http",
        ]

    # ------------------------------------------------------------------
    # BaseDeployer abstract methods — server-specific names
    # ------------------------------------------------------------------

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
        return await self._deploy_service(
            server_name, source_url, framework, env, description,
            tags, port, current_version, refresh_repo, no_build,
        )

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
        return self._start_service(server_name)

    def stop_server(self, server_name: str) -> str:
        return self._stop_service(server_name)

    def remove_server(self, server_name: str) -> str:
        return self._remove_service(server_name)
