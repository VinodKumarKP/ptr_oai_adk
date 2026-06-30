"""Unit tests for oai_mcp_registry.services.deployers."""

import pytest
from unittest.mock import MagicMock, patch, call

from oai_mcp_registry.services.deployers.base import BaseDeployer
from oai_mcp_registry.services.deployers.factory import DeployerFactory


class TestDeployerFactory:
    def test_get_deployer_docker(self, tmp_path):
        """Test getting docker deployer."""
        with patch("oai_mcp_registry.services.deployers.factory.DockerComposeManager") as mock_docker:
            mock_docker.return_value = MagicMock()
            deployer = DeployerFactory.get_deployer(
                mode="docker",
                seed_config={},
                compose_output_path=str(tmp_path / "docker-compose.yaml"),
                base_compose_path=str(tmp_path / "base.yaml"),
                agent_base_url="http://localhost:8081",
                agent_local_registry_url="http://host.docker.internal:8081"
            )
            assert mock_docker.called

    def test_get_deployer_python_package(self, tmp_path):
        """Test getting python_package deployer."""
        with patch("oai_mcp_registry.services.deployers.factory.PythonPackageDeployer") as mock_python:
            mock_python.return_value = MagicMock()
            deployer = DeployerFactory.get_deployer(
                mode="python_package",
                seed_config={},
                compose_output_path=str(tmp_path / "docker-compose.yaml"),
                base_compose_path=str(tmp_path / "base.yaml"),
                agent_base_url="http://localhost:8081",
                agent_local_registry_url="http://host.docker.internal:8081"
            )
            assert mock_python.called

    def test_get_deployer_kubernetes_not_implemented(self, tmp_path):
        """Test that kubernetes deployer raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            DeployerFactory.get_deployer(
                mode="kubernetes",
                seed_config={},
                compose_output_path=str(tmp_path / "docker-compose.yaml"),
                base_compose_path=str(tmp_path / "base.yaml"),
                agent_base_url="http://localhost:8081",
                agent_local_registry_url="http://host.docker.internal:8081"
            )

    def test_get_deployer_invalid_mode(self, tmp_path):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            DeployerFactory.get_deployer(
                mode="invalid_mode",
                seed_config={},
                compose_output_path=str(tmp_path / "docker-compose.yaml"),
                base_compose_path=str(tmp_path / "base.yaml"),
                agent_base_url="http://localhost:8081",
                agent_local_registry_url="http://host.docker.internal:8081"
            )
