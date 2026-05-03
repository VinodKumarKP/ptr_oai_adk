from typing import Any, Dict
from oai_agent_registry.services.base_deployer import BaseDeployer
from oai_agent_registry.services.docker_compose_manager import DockerComposeManager

class DeployerFactory:
    """Factory to instantiate the correct deployment manager based on the configured mode."""

    @staticmethod
    def get_deployer(
        mode: str,
        seed_config: Dict[str, Any],
        compose_output_path: str,
        base_compose_path: str,
        agent_base_url: str,
        agent_local_registry_url: str,
    ) -> BaseDeployer:
        if mode == "docker":
            return DockerComposeManager(
                seed_config=seed_config,
                compose_output_path=compose_output_path,
                base_compose_path=base_compose_path,
                agent_base_url=agent_base_url,
                agent_local_registry_url=agent_local_registry_url,
            )
        elif mode == "kubernetes":
            raise NotImplementedError("Kubernetes deployment mode is not yet implemented.")
        elif mode == "python_package":
            raise NotImplementedError("Python package deployment mode is not yet implemented.")
        else:
            raise ValueError(f"Unknown deployment mode: {mode}. Supported modes: docker, kubernetes, python_package.")
