from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator

class BaseDeployer(ABC):
    """
    Abstract base class for all deployment strategies (Docker Compose, Kubernetes, Python Package, etc.).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the deployment environment (e.g., start base services)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the deployment environment (e.g., stop all services)."""
        pass

    @abstractmethod
    def find_available_port(self) -> int:
        """Find an available port for a new agent."""
        pass

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
        pass

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
        """Deploy a single agent and stream the output."""
        pass

    @abstractmethod
    def start_agent(self, agent_name: str) -> str:
        """Start a previously deployed but stopped agent."""
        pass

    @abstractmethod
    def stop_agent(self, agent_name: str) -> str:
        """Stop a running agent (pause without destroying)."""
        pass

    @abstractmethod
    def remove_agent(self, agent_name: str) -> str:
        """Completely remove and teardown an agent."""
        pass

    @abstractmethod
    def image_exists(self, agent_name: str, version: str) -> bool:
        """Check if the deployment artifact for the given version already exists locally."""
        pass


    @abstractmethod
    def start_infra_services(self) -> None:
        """Start any necessary infrastructure services (e.g., databases, message queues)."""
        pass