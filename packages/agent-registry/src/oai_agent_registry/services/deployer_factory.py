import os
from typing import Any, Dict
from oai_agent_registry.services.base_deployer import BaseDeployer
from oai_agent_registry.services.docker_compose_manager import DockerComposeManager
from oai_agent_registry.services.python_package_deployer import PythonPackageDeployer

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
            # Determine the python package resources directory relative to the docker output path
            base_dir = os.path.abspath(os.path.join(os.path.dirname(compose_output_path), "..", "python"))
            port = agent_local_registry_url.split(':')[-1]
            agent_local_registry_url = f"http://localhost:{port}"
            return PythonPackageDeployer(
                seed_config=seed_config,
                base_dir=base_dir,
                agent_base_url=agent_base_url,
                agent_local_registry_url=agent_local_registry_url,
            )
        else:
            raise ValueError(f"Unknown deployment mode: {mode}. Supported modes: docker, kubernetes, python_package.")
