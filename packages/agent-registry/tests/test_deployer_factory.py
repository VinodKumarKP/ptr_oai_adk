import pytest
from oai_agent_registry.services.deployers.factory import DeployerFactory
from oai_agent_registry.services.deployers.docker_compose import DockerComposeManager
from oai_agent_registry.services.deployers.python_package import PythonPackageDeployer

def test_deployer_factory_docker(tmp_path):
    compose_path = tmp_path / "compose.yaml"
    base_path = tmp_path / "base.yaml"
    deployer = DeployerFactory.get_deployer(
        mode="docker",
        seed_config={},
        compose_output_path=str(compose_path),
        base_compose_path=str(base_path),
        agent_base_url="http://base",
        agent_local_registry_url="http://local:8000"
    )
    assert isinstance(deployer, DockerComposeManager)

def test_deployer_factory_kubernetes(tmp_path):
    compose_path = tmp_path / "compose.yaml"
    base_path = tmp_path / "base.yaml"
    with pytest.raises(NotImplementedError, match="Kubernetes deployment mode is not yet implemented"):
        DeployerFactory.get_deployer(
            mode="kubernetes",
            seed_config={},
            compose_output_path=str(compose_path),
            base_compose_path=str(base_path),
            agent_base_url="http://base",
            agent_local_registry_url="http://local:8000"
        )

def test_deployer_factory_python_package(tmp_path):
    # Nested compose_output_path so we can do dirname, .., python and create it inside tmp_path
    compose_dir = tmp_path / "subdir"
    compose_dir.mkdir()
    compose_path = compose_dir / "compose.yaml"
    base_path = tmp_path / "base.yaml"
    deployer = DeployerFactory.get_deployer(
        mode="python_package",
        seed_config={},
        compose_output_path=str(compose_path),
        base_compose_path=str(base_path),
        agent_base_url="http://base",
        agent_local_registry_url="http://local:8000"
    )
    assert isinstance(deployer, PythonPackageDeployer)

def test_deployer_factory_unknown(tmp_path):
    compose_path = tmp_path / "compose.yaml"
    base_path = tmp_path / "base.yaml"
    with pytest.raises(ValueError, match="Unknown deployment mode"):
        DeployerFactory.get_deployer(
            mode="unknown",
            seed_config={},
            compose_output_path=str(compose_path),
            base_compose_path=str(base_path),
            agent_base_url="http://base",
            agent_local_registry_url="http://local:8000"
        )
