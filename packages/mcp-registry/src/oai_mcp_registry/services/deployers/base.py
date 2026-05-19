"""
MCP-registry deployer base.

Inherits the four framework-agnostic abstract methods from the canonical
:py:class:`oai_platform_core.deployers.base.BaseDeployer` and adds the
MCP-server-specific deploy / lifecycle methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.base import BaseDeployer

__all__ = ["BaseDeployer"]


class BaseDeployer(BaseDeployer):  # type: ignore[no-redef]
    """Abstract base for all MCP-server deployment strategies.

    Extends the platform-core base with MCP-server-specific abstract
    methods.  All concrete deployers (``DockerComposeManager``,
    ``PythonPackageDeployer``, …) inherit from this class.
    """

    @abstractmethod
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
        """Deploy a single MCP server."""

    @abstractmethod
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
        """Deploy a single MCP server and stream the build/deploy output."""

    @abstractmethod
    def start_server(self, server_name: str) -> str:
        """Start a previously deployed but stopped server."""

    @abstractmethod
    def stop_server(self, server_name: str) -> str:
        """Stop a running server without destroying it."""

    @abstractmethod
    def remove_server(self, server_name: str) -> str:
        """Completely remove and tear down a server."""
