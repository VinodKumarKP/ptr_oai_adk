"""Unit tests for oai_mcp_registry.models."""

import pytest
from pydantic import ValidationError

from oai_mcp_registry.models import (
    EnvVarRequirement,
    MCPServerDiscoveryItem,
    MCPServerDiscoveryResult,
    BulkMCPServerRegistrationRequest,
    BulkMCPServerRegistrationResult,
    ServerConfig,
    ServerRegistration,
    ServerDeregistration,
    UpdateServerEnvVarsRequest,
    RegistryConfig,
    AppConfig,
    McpServerLifecycleAction,
    ServerAction,
    ServerActionHistory,
)


class TestEnvVarRequirement:
    def test_create_minimal(self):
        env_var = EnvVarRequirement(name="API_KEY")
        assert env_var.name == "API_KEY"
        assert env_var.description is None
        assert env_var.default is None
        assert env_var.required is False
        assert env_var.sensitive is False

    def test_create_with_all_fields(self):
        env_var = EnvVarRequirement(
            name="API_KEY",
            description="API Key for authentication",
            default="default_key",
            required=True,
            sensitive=True
        )
        assert env_var.name == "API_KEY"
        assert env_var.description == "API Key for authentication"
        assert env_var.default == "default_key"
        assert env_var.required is True
        assert env_var.sensitive is True


class TestMCPServerDiscoveryItem:
    def test_create_minimal(self):
        item = MCPServerDiscoveryItem(name="server1")
        assert item.name == "server1"
        assert item.description is None
        assert item.status == "available"
        assert item.tags == []
        assert item.sensitive_vars == []

    def test_create_with_all_fields(self):
        item = MCPServerDiscoveryItem(
            name="server1",
            description="Test server",
            tags=["test", "prod"],
            port=8000,
            source="https://github.com/test/repo",
            status="available",
            config_file="config.yaml",
            error=None,
            env_vars={"KEY": "value"},
            sensitive_vars=["KEY"]
        )
        assert item.name == "server1"
        assert item.description == "Test server"
        assert item.tags == ["test", "prod"]
        assert item.env_vars == {"KEY": "value"}


class TestMCPServerDiscoveryResult:
    def test_create(self):
        result = MCPServerDiscoveryResult(
            git_repository_url="https://github.com/test/repo",
            total_found=3,
            available_to_register=2,
            already_registered=1,
            invalid=0,
            servers=[
                MCPServerDiscoveryItem(name="server1"),
                MCPServerDiscoveryItem(name="server2"),
            ]
        )
        assert result.git_repository_url == "https://github.com/test/repo"
        assert result.total_found == 3
        assert result.available_to_register == 2
        assert len(result.servers) == 2


class TestBulkMCPServerRegistrationRequest:
    def test_create_minimal(self):
        req = BulkMCPServerRegistrationRequest(
            git_repository_url="https://github.com/test/repo",
            server_names=["server1", "server2"]
        )
        assert req.git_repository_url == "https://github.com/test/repo"
        assert req.server_names == ["server1", "server2"]
        assert req.deployment_mode == "docker"
        assert req.auth_token is None

    def test_create_with_all_fields(self):
        req = BulkMCPServerRegistrationRequest(
            git_repository_url="https://github.com/test/repo",
            server_names=["server1"],
            deployment_mode="kubernetes",
            auth_token="token123",
            config_path="custom/path",
            server_env_overrides={"server1": {"KEY": "value"}}
        )
        assert req.deployment_mode == "kubernetes"
        assert req.auth_token == "token123"
        assert req.server_env_overrides == {"server1": {"KEY": "value"}}


class TestBulkMCPServerRegistrationResult:
    def test_create(self):
        result = BulkMCPServerRegistrationResult(
            total_registered=2,
            successful=["server1", "server2"],
            failed=[]
        )
        assert result.total_registered == 2
        assert result.successful == ["server1", "server2"]
        assert result.failed == []


class TestServerConfig:
    def test_create_minimal(self):
        config = ServerConfig()
        assert config.endpoint is None
        assert config.enabled is True
        assert config.description == "A proxied MCP server"
        assert config.tags == []

    def test_create_with_all_fields(self):
        config = ServerConfig(
            endpoint="http://localhost:8000",
            enabled=True,
            port=8000,
            description="Test server",
            registered_via="registry",
            source="https://github.com/test/repo",
            tags=["test"],
            current_version="1.0.0",
            available_versions=["1.0.0", "2.0.0"],
            deployment_mode="kubernetes",
            env_vars={"KEY": "value"},
            sensitive_vars=["KEY"]
        )
        assert config.endpoint == "http://localhost:8000"
        assert config.port == 8000
        assert config.registered_via == "registry"


class TestServerRegistration:
    def test_create_minimal(self):
        reg = ServerRegistration(name="test_server")
        assert reg.name == "test_server"
        assert reg.description == "A proxied MCP server"
        assert reg.active is True
        assert reg.registered_via == "dynamic"
        assert reg.deployment_mode == "unknown"

    def test_create_with_all_fields(self):
        reg = ServerRegistration(
            name="test_server",
            description="Test",
            endpoint="http://localhost:8000",
            port=8000,
            active=True,
            registered_via="registry",
            source="https://github.com/test",
            tags=["test"],
            current_version="1.0.0",
            available_versions=["1.0.0"],
            deployment_mode="docker",
            env_vars={"KEY": "val"},
            sensitive_vars=["KEY"]
        )
        assert reg.deployment_mode == "docker"
        assert reg.registered_via == "registry"


class TestServerDeregistration:
    def test_create(self):
        dereg = ServerDeregistration(name="test_server")
        assert dereg.name == "test_server"


class TestUpdateServerEnvVarsRequest:
    def test_create_minimal(self):
        req = UpdateServerEnvVarsRequest()
        assert req.env_vars == {}
        assert req.sensitive_vars == []

    def test_create_with_data(self):
        req = UpdateServerEnvVarsRequest(
            env_vars={"KEY1": "value1", "KEY2": "value2"},
            sensitive_vars=["KEY1"]
        )
        assert req.env_vars == {"KEY1": "value1", "KEY2": "value2"}
        assert req.sensitive_vars == ["KEY1"]


class TestRegistryConfig:
    def test_create_with_defaults(self):
        config = RegistryConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8081
        assert config.enable_auto_discovery is False
        assert config.start_port == 8000
        assert config.end_port == 8100
        assert config.enable_cors is True
        assert config.auto_start_infra is False

    def test_create_with_custom_values(self):
        config = RegistryConfig(
            host="localhost",
            port=9000,
            enable_cors=False,
            enable_auto_discovery=True
        )
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.enable_cors is False
        assert config.enable_auto_discovery is True


class TestAppConfig:
    def test_create_minimal(self):
        config = AppConfig(servers={})
        assert config.servers == {}
        assert isinstance(config.registry, RegistryConfig)

    def test_create_with_servers(self):
        config = AppConfig(
            servers={
                "server1": ServerConfig(endpoint="http://localhost:8000"),
                "server2": ServerConfig(endpoint="http://localhost:8001")
            }
        )
        assert len(config.servers) == 2
        assert "server1" in config.servers


class TestMcpServerLifecycleAction:
    def test_create_minimal(self):
        action = McpServerLifecycleAction(action="start")
        assert action.action == "start"
        assert action.version is None
        assert action.stream_output is False

    def test_create_with_all_fields(self):
        action = McpServerLifecycleAction(
            action="update",
            version="2.0.0",
            stream_output=True
        )
        assert action.action == "update"
        assert action.version == "2.0.0"
        assert action.stream_output is True

    @pytest.mark.parametrize("action", [
        "start", "stop", "redeploy", "refresh", "restart",
        "rebuild", "upgrade", "update", "downgrade", "delete"
    ])
    def test_valid_actions(self, action):
        act = McpServerLifecycleAction(action=action)
        assert act.action == action


class TestServerAction:
    def test_create(self):
        action = ServerAction(
            id=1,
            server_name="test_server",
            action="start",
            version="1.0.0",
            created_at="2024-01-01T00:00:00Z"
        )
        assert action.id == 1
        assert action.server_name == "test_server"
        assert action.action == "start"


class TestServerActionHistory:
    def test_create(self):
        actions = [
            ServerAction(
                id=1,
                server_name="test_server",
                action="start",
                version=None,
                created_at="2024-01-01T00:00:00Z"
            )
        ]
        history = ServerActionHistory(
            server_name="test_server",
            total_count=1,
            actions=actions
        )
        assert history.server_name == "test_server"
        assert history.total_count == 1
        assert len(history.actions) == 1
