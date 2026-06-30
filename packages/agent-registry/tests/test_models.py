"""Unit tests for oai_agent_registry.models."""

import pytest
from pydantic import ValidationError

from oai_agent_registry.models import (
    EnvVarRequirement,
    AgentDiscoveryItem,
    AgentDiscoveryResult,
    BulkAgentRegistrationRequest,
    BulkAgentRegistrationResult,
    AgentConfig,
    AgentRegistration,
    AgentDeregistration,
    AgentLifecycleAction,
    RegistryConfig,
    AgentAction,
    AgentActionHistory,
    Config,
    UpdateServerEnvVarsRequest,
)


class TestEnvVarRequirement:
    def test_create_with_defaults(self):
        env_var = EnvVarRequirement(name="API_KEY")
        assert env_var.name == "API_KEY"
        assert env_var.default is None
        assert env_var.required is True
        assert env_var.sensitive is False

    def test_create_with_all_fields(self):
        env_var = EnvVarRequirement(
            name="API_KEY",
            default="default_key",
            required=False,
            sensitive=True
        )
        assert env_var.name == "API_KEY"
        assert env_var.default == "default_key"
        assert env_var.required is False
        assert env_var.sensitive is True


class TestAgentDiscoveryItem:
    def test_create_minimal(self):
        item = AgentDiscoveryItem(name="agent1")
        assert item.name == "agent1"
        assert item.description is None
        assert item.framework is None
        assert item.status == "available"
        assert item.tags == []
        assert item.prompts == []
        assert item.required_env == []

    def test_create_with_all_fields(self):
        item = AgentDiscoveryItem(
            name="agent1",
            description="Test agent",
            framework="langgraph",
            agent_type="chat",
            tags=["test", "prod"],
            prompts=["prompt1"],
            port=8000,
            source="https://github.com/test/repo",
            status="available",
            config_file="config.yaml",
            error=None,
            required_env=[EnvVarRequirement(name="API_KEY", sensitive=True)]
        )
        assert item.name == "agent1"
        assert item.description == "Test agent"
        assert item.framework == "langgraph"
        assert item.tags == ["test", "prod"]
        assert len(item.required_env) == 1


class TestAgentDiscoveryResult:
    def test_create(self):
        result = AgentDiscoveryResult(
            git_repository_url="https://github.com/test/repo",
            total_found=3,
            available_to_register=2,
            already_registered=1,
            invalid=0,
            agents=[
                AgentDiscoveryItem(name="agent1"),
                AgentDiscoveryItem(name="agent2"),
            ]
        )
        assert result.git_repository_url == "https://github.com/test/repo"
        assert result.total_found == 3
        assert result.available_to_register == 2
        assert len(result.agents) == 2


class TestBulkAgentRegistrationRequest:
    def test_create_minimal(self):
        req = BulkAgentRegistrationRequest(
            git_repository_url="https://github.com/test/repo",
            agent_names=["agent1", "agent2"]
        )
        assert req.git_repository_url == "https://github.com/test/repo"
        assert req.agent_names == ["agent1", "agent2"]
        assert req.framework is None
        assert req.deployment_mode == "docker"
        assert req.auth_token is None

    def test_create_with_all_fields(self):
        req = BulkAgentRegistrationRequest(
            git_repository_url="https://github.com/test/repo",
            agent_names=["agent1"],
            framework="langgraph",
            deployment_mode="kubernetes",
            auth_token="token123",
            config_path="custom/path",
            env_vars={"KEY": "value"}
        )
        assert req.framework == "langgraph"
        assert req.deployment_mode == "kubernetes"
        assert req.auth_token == "token123"
        assert req.config_path == "custom/path"
        assert req.env_vars == {"KEY": "value"}

    def test_invalid_framework(self):
        with pytest.raises(ValidationError):
            BulkAgentRegistrationRequest(
                git_repository_url="https://github.com/test/repo",
                agent_names=["agent1"],
                framework="invalid_framework"
            )

    def test_invalid_deployment_mode(self):
        with pytest.raises(ValidationError):
            BulkAgentRegistrationRequest(
                git_repository_url="https://github.com/test/repo",
                agent_names=["agent1"],
                deployment_mode="invalid_mode"
            )


class TestBulkAgentRegistrationResult:
    def test_create(self):
        result = BulkAgentRegistrationResult(
            total_registered=2,
            successful=["agent1", "agent2"],
            failed=[]
        )
        assert result.total_registered == 2
        assert result.successful == ["agent1", "agent2"]
        assert result.failed == []

    def test_with_failures(self):
        result = BulkAgentRegistrationResult(
            total_registered=1,
            successful=["agent1"],
            failed=[{"agent2": "Error message"}]
        )
        assert result.total_registered == 1
        assert len(result.failed) == 1


class TestAgentConfig:
    def test_create_minimal(self):
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.endpoint is None
        assert config.enabled is True
        assert config.timeout == 300
        assert config.framework is None
        assert config.tags == []
        assert config.env_vars is None

    def test_create_with_all_fields(self):
        config = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            enabled=True,
            timeout=600,
            description="Test",
            port=8000,
            source="https://github.com/test/repo",
            framework="langgraph",
            prompts=["prompt1"],
            tags=["test"],
            current_version="1.0.0",
            available_versions=["1.0.0", "2.0.0"],
            registered_via="dynamic",
            deployment_mode="kubernetes",
            env_vars={"KEY": "value"},
            sensitive_vars=["KEY"]
        )
        assert config.name == "test_agent"
        assert config.endpoint == "http://localhost:8000"
        assert config.timeout == 600
        assert config.tags == ["test"]

    def test_invalid_framework(self):
        with pytest.raises(ValidationError):
            AgentConfig(name="test", framework="invalid")


class TestAgentRegistration:
    def test_create_minimal(self):
        reg = AgentRegistration(name="test_agent")
        assert reg.name == "test_agent"
        assert reg.description is None
        assert reg.active is True
        assert reg.registered_via == "dynamic"
        assert reg.deployment_mode == "unknown"

    def test_create_with_all_fields(self):
        reg = AgentRegistration(
            name="test_agent",
            description="Test",
            endpoint="http://localhost:8000",
            port=8000,
            source="https://github.com/test",
            active=True,
            registered_via="registry",
            framework="langgraph",
            prompts=["p1"],
            tags=["test"],
            current_version="1.0.0",
            available_versions=["1.0.0"],
            deployment_mode="docker",
            env_vars={"KEY": "val"},
            sensitive_vars=["KEY"]
        )
        assert reg.framework == "langgraph"
        assert reg.registered_via == "registry"
        assert reg.deployment_mode == "docker"


class TestAgentDeregistration:
    def test_create(self):
        dereg = AgentDeregistration(name="test_agent")
        assert dereg.name == "test_agent"


class TestAgentLifecycleAction:
    def test_create_minimal(self):
        action = AgentLifecycleAction(action="start")
        assert action.action == "start"
        assert action.version is None
        assert action.stream_output is False

    def test_create_with_all_fields(self):
        action = AgentLifecycleAction(
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
        act = AgentLifecycleAction(action=action)
        assert act.action == action

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            AgentLifecycleAction(action="invalid_action")


class TestRegistryConfig:
    def test_create_with_defaults(self):
        config = RegistryConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8081
        assert config.default_timeout == 300
        assert config.enable_cors is True
        assert config.log_requests is True
        assert config.strip_prefix is True
        assert config.enable_auto_discovery is False
        assert config.start_port == 8000
        assert config.end_port == 8200
        assert config.auth_enabled is False
        assert config.force_auth is False
        assert config.api_key is None
        assert config.deployment_mode == "docker"
        assert config.max_version == 10
        assert config.auto_start_infra is False

    def test_create_with_custom_values(self):
        config = RegistryConfig(
            host="localhost",
            port=9000,
            default_timeout=600,
            enable_cors=False,
            auth_enabled=True,
            api_key="secret_key",
            deployment_mode="kubernetes"
        )
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.default_timeout == 600
        assert config.enable_cors is False
        assert config.auth_enabled is True
        assert config.api_key == "secret_key"
        assert config.deployment_mode == "kubernetes"


class TestAgentAction:
    def test_create(self):
        action = AgentAction(
            id=1,
            agent_name="test_agent",
            action="start",
            version="1.0.0",
            created_at="2024-01-01T00:00:00Z"
        )
        assert action.id == 1
        assert action.agent_name == "test_agent"
        assert action.action == "start"
        assert action.version == "1.0.0"


class TestAgentActionHistory:
    def test_create(self):
        actions = [
            AgentAction(
                id=1,
                agent_name="test_agent",
                action="start",
                version=None,
                created_at="2024-01-01T00:00:00Z"
            )
        ]
        history = AgentActionHistory(
            agent_name="test_agent",
            total_count=1,
            actions=actions
        )
        assert history.agent_name == "test_agent"
        assert history.total_count == 1
        assert len(history.actions) == 1


class TestConfig:
    def test_create_minimal(self):
        config = Config(agents={})
        assert config.agents == {}
        assert isinstance(config.registry, RegistryConfig)

    def test_create_with_agents(self):
        config = Config(
            agents={
                "agent1": "http://localhost:8000",
                "agent2": {
                    "endpoint": "http://localhost:8001",
                    "enabled": True
                }
            }
        )
        assert len(config.agents) == 2


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
