"""
oai_platform_core.deployers.base — canonical BaseDeployer ABC.

All deployment strategies (Docker Compose, Kubernetes, Python Package, …)
implement this interface.  The four methods declared here are **identical**
across every registry package; package-specific abstract methods
(``deploy_agent`` / ``deploy_server``, etc.) are added by per-package
sub-classes that inherit from this base.

Usage
-----
    # In oai_agent_registry:
    from oai_platform_core.deployers.base import BaseDeployer

    class AgentBaseDeployer(BaseDeployer):
        @abstractmethod
        async def deploy_agent(self, agent_name: str, ...) -> str: ...

    # Concrete implementation:
    class DockerComposeManager(AgentBaseDeployer): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseDeployer"]


class BaseDeployer(ABC):
    """Abstract base for all OAI deployment strategies.

    Sub-classes must implement every abstract method declared here.
    Package-specific registries extend this class with additional abstract
    methods that carry the appropriate naming convention (``deploy_agent``
    vs ``deploy_server``, etc.).
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the deployment environment (e.g. start base services)."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the deployment environment (e.g. stop all services)."""

    # ------------------------------------------------------------------
    # Port allocation
    # ------------------------------------------------------------------

    @abstractmethod
    def find_available_port(self) -> int:
        """Return an unused port for a new service deployment."""

    # ------------------------------------------------------------------
    # Artifact inspection
    # ------------------------------------------------------------------

    @abstractmethod
    def image_exists(self, service_name: str, version: str) -> bool:
        """Return ``True`` if the deployment artifact for *version* exists locally."""
