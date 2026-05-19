"""
Agent-registry deployer base.

Inherits the four framework-agnostic abstract methods from the canonical
:py:class:`oai_platform_core.deployers.base.BaseDeployer` and adds the
agent-specific deploy / lifecycle methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from oai_platform_core.deployers.base import BaseDeployer

__all__ = ["BaseDeployer"]


class BaseDeployer(BaseDeployer):  # type: ignore[no-redef]
    """Abstract base for all agent deployment strategies.

    Extends the platform-core base with agent-specific abstract methods.
    All concrete deployers (``DockerComposeManager``,
    ``PythonPackageDeployer``, …) inherit from this class.
    """

    @abstractmethod
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
        """Deploy a single agent."""

    @abstractmethod
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
        """Deploy a single agent and stream the build/deploy output."""

    @abstractmethod
    def start_agent(self, agent_name: str) -> str:
        """Start a previously deployed but stopped agent."""

    @abstractmethod
    def stop_agent(self, agent_name: str) -> str:
        """Stop a running agent without destroying it."""

    @abstractmethod
    def remove_agent(self, agent_name: str) -> str:
        """Completely remove and tear down an agent."""

    @abstractmethod
    def start_infra_services(self) -> None:
        """Start infrastructure services required by agents (DBs, queues, …)."""
