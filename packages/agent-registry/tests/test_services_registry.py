"""Unit tests for oai_agent_registry.services.registry."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch, call

from oai_agent_registry.models import (
    AgentConfig,
    AgentRegistration,
    AgentDeregistration,
    RegistryConfig,
)
from oai_agent_registry.services.registry import (
    AgentRegistry,
    _mask_sensitive_env,
    _sse,
)
from fastapi import HTTPException


class TestMaskSensitiveEnv:
    def test_mask_with_sensitive_vars(self):
        env_vars = {
            "API_KEY": "secret_key",
            "DATABASE_URL": "postgres://localhost",
            "PUBLIC_VAR": "public_value"
        }
        sensitive_vars = ["API_KEY", "DATABASE_URL"]

        result = _mask_sensitive_env(env_vars, sensitive_vars)

        assert result["API_KEY"] == "***"
        assert result["DATABASE_URL"] == "***"
        assert result["PUBLIC_VAR"] == "public_value"

    def test_mask_with_no_sensitive_vars(self):
        env_vars = {
            "API_KEY": "secret_key",
            "PUBLIC_VAR": "public_value"
        }

        result = _mask_sensitive_env(env_vars, [])

        assert result["API_KEY"] == "secret_key"
        assert result["PUBLIC_VAR"] == "public_value"

    def test_mask_with_none_env_vars(self):
        result = _mask_sensitive_env(None, ["KEY"])
        assert result == {}

    def test_mask_with_none_sensitive_vars(self):
        env_vars = {"API_KEY": "secret"}
        result = _mask_sensitive_env(env_vars, None)
        assert result["API_KEY"] == "secret"

    def test_mask_with_missing_sensitive_var(self):
        env_vars = {"API_KEY": "secret"}
        sensitive_vars = ["MISSING_KEY"]

        result = _mask_sensitive_env(env_vars, sensitive_vars)

        assert result["API_KEY"] == "secret"


class TestSSE:
    def test_sse_wrapper(self):
        """Test that _sse creates a StreamingResponse."""
        async def mock_generator():
            yield "test_data"

        response = _sse(mock_generator())

        assert response.media_type == "text/event-stream"
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"


class TestAgentRegistryInit:
    def test_init_with_config_path(self, mock_config_file):
        """Test registry initialization with config file."""
        registry = AgentRegistry(config_path=mock_config_file)

        assert registry.config_path == mock_config_file
        assert registry.agents is not None
        assert registry.registry_config is not None
        assert isinstance(registry.registry_config, RegistryConfig)

    def test_init_creates_empty_deployers(self, mock_config_file):
        """Test that init creates empty deployers dict."""
        registry = AgentRegistry(config_path=mock_config_file)
        assert isinstance(registry.deployers, dict)

    def test_init_sets_default_config_path(self, monkeypatch):
        """Test that default config path is used when none provided."""
        monkeypatch.delenv("REGISTRY_CONFIG_PATH", raising=False)
        registry = AgentRegistry()
        assert "registry_config.json" in registry.config_path


class TestAgentRegistryLoadConfig:
    def test_load_config_from_file(self, mock_config_file):
        """Test loading config from file."""
        registry = AgentRegistry(config_path=mock_config_file)

        assert registry.config is not None
        assert len(registry.agents) > 0
        assert "test_agent" in registry.agents

    def test_load_config_missing_file(self, tmp_path):
        """Test loading config when file doesn't exist."""
        missing_path = str(tmp_path / "nonexistent.json")
        registry = AgentRegistry(config_path=missing_path)

        assert registry.config is not None
        assert len(registry.agents) == 0

    def test_load_config_with_string_endpoints(self, tmp_path):
        """Test loading config with string endpoints."""
        config_file = tmp_path / "config.json"
        config_data = {
            "agents": {
                "agent1": "http://localhost:8000",
            }
        }
        config_file.write_text(json.dumps(config_data))

        registry = AgentRegistry(config_path=str(config_file))

        assert "agent1" in registry.agents
        assert registry.agents["agent1"].endpoint == "http://localhost:8000"

    def test_load_config_with_dict_agents(self, tmp_path):
        """Test loading config with dict agent configs."""
        config_file = tmp_path / "config.json"
        config_data = {
            "agents": {
                "agent1": {
                    "endpoint": "http://localhost:8000",
                    "framework": "langgraph"
                }
            }
        }
        config_file.write_text(json.dumps(config_data))

        registry = AgentRegistry(config_path=str(config_file))

        assert registry.agents["agent1"].framework == "langgraph"
        assert registry.agents["agent1"].registered_via == "config"


class TestAgentRegistryMasks:
    def test_get_deployer_returns_default(self, registry_instance):
        """Test that _get_deployer returns docker by default."""
        registry_instance.deployers["docker"] = MagicMock()

        deployer = registry_instance._get_deployer("unknown")

        assert deployer == registry_instance.deployers["docker"]

    def test_get_deployer_returns_specific(self, registry_instance):
        """Test that _get_deployer returns specific deployer."""
        docker_deployer = MagicMock()
        python_deployer = MagicMock()
        registry_instance.deployers["docker"] = docker_deployer
        registry_instance.deployers["python_package"] = python_deployer

        deployer = registry_instance._get_deployer("python_package")

        assert deployer == python_deployer

    def test_agent_deploy_kwargs(self, registry_instance, agent_config):
        """Test building deploy kwargs."""
        kwargs = registry_instance._agent_deploy_kwargs(
            "test_agent",
            agent_config,
            extra_param="value"
        )

        assert kwargs["agent_name"] == "test_agent"
        assert kwargs["source_url"] == (agent_config.source or "")
        assert kwargs["framework"] == agent_config.framework
        assert kwargs["extra_param"] == "value"

    def test_lifecycle_json(self, registry_instance):
        """Test building lifecycle JSON response."""
        registry_instance.agents["test_agent"] = MagicMock(enabled=True)

        response = registry_instance._lifecycle_json("test_agent", "start")

        assert response.body
        data = json.loads(response.body)
        assert data["agent"] == "test_agent"
        assert data["action"] == "start"
        assert data["enabled"] is True


class TestAgentRegistryEndpointHostPort:
    def test_endpoint_with_url_scheme(self):
        """Test parsing endpoint with URL scheme."""
        cfg = MagicMock(endpoint="http://localhost:8000", port=None)

        host, port = AgentRegistry._endpoint_host_port(cfg)

        assert host == "host.docker.internal"
        assert port == 8000

    def test_endpoint_without_scheme(self):
        """Test parsing endpoint without URL scheme."""
        cfg = MagicMock(endpoint="localhost:8000", port=None)

        host, port = AgentRegistry._endpoint_host_port(cfg)

        assert host == "host.docker.internal"

    def test_loopback_hosts_rewrite(self):
        """Test that loopback hosts are rewritten."""
        for loopback in ["localhost", "127.0.0.1", "0.0.0.0"]:
            cfg = MagicMock(endpoint=f"http://{loopback}:8000", port=None)
            host, port = AgentRegistry._endpoint_host_port(cfg)
            assert host == "host.docker.internal"

    def test_external_host_not_rewritten(self):
        """Test that external hosts are not rewritten."""
        cfg = MagicMock(endpoint="http://example.com:8000", port=None)

        host, port = AgentRegistry._endpoint_host_port(cfg)

        assert host == "example.com"

    def test_no_endpoint(self):
        """Test when no endpoint is provided."""
        cfg = MagicMock(endpoint=None, port=8000)

        host, port = AgentRegistry._endpoint_host_port(cfg)

        assert host is None


class TestAgentRegistryPrometheustargets:
    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("os.replace")
    @patch("json.dump")
    def test_write_prometheus_targets(self, mock_dump, mock_replace, mock_makedirs, mock_open, registry_instance):
        """Test writing Prometheus targets."""
        registry_instance._build_dir = "/tmp/build"
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        registry_instance.agents["agent1"] = MagicMock(
            enabled=True,
            endpoint="http://localhost:8000",
            port=8000
        )

        registry_instance._write_prometheus_targets()

        mock_makedirs.assert_called()
        assert mock_dump.called or mock_open.called


class TestAgentRegistryGetInfo:
    @pytest.mark.asyncio
    async def test_get_info(self, registry_instance):
        """Test getting registry info."""
        registry_instance.agents["test_agent"] = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            enabled=True,
            description="Test",
            framework="langgraph",
            registered_via="dynamic",
            current_version="1.0.0",
            available_versions=["1.0.0"],
            deployment_mode="docker",
            env_vars={"KEY": "value"},
            sensitive_vars=["KEY"]
        )

        with patch.dict("os.environ", {"AGENT_BASE_URL": "http://localhost", "AGENT_BASE_URL_PORT": "8081"}):
            response = await registry_instance.get_info()

            assert response.status_code == 200
            data = json.loads(response.body)
            assert data["registry"]["total_agents"] == 1
            assert data["registry"]["enabled_agents"] == 1


class TestAgentRegistryHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy_agent(self, registry_instance):
        """Test health check with healthy agent."""
        registry_instance.agents["test_agent"] = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            enabled=True
        )
        registry_instance.client.get = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "ok"})
            )
        )

        response = await registry_instance.health_check()

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["agents"]["test_agent"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_disabled_agent(self, registry_instance):
        """Test health check with disabled agent."""
        registry_instance.agents["test_agent"] = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            enabled=False
        )

        response = await registry_instance.health_check()

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["agents"]["test_agent"]["status"] == "disabled"


class TestAgentRegistryRegisterAgent:
    @pytest.mark.asyncio
    async def test_register_new_agent(self, registry_instance, agent_registration):
        """Test registering a new agent."""
        registry_instance.db_logger.get_agent_details = AsyncMock(return_value={})

        response = await registry_instance.register_agent(agent_registration)

        assert response.status_code == 200
        assert "test_agent" in registry_instance.agents

    @pytest.mark.asyncio
    async def test_register_existing_agent(self, registry_instance, agent_registration, agent_config):
        """Test updating an existing agent."""
        registry_instance.agents["test_agent"] = agent_config
        registry_instance.db_logger.get_agent_details = AsyncMock(return_value={})
        registry_instance._build_dir = "/tmp"

        response = await registry_instance.register_agent(agent_registration)

        assert response.status_code == 200


class TestAgentRegistryDeregisterAgent:
    @pytest.mark.asyncio
    async def test_deregister_agent(self, registry_instance):
        """Test deregistering an agent."""
        agent_config = AgentConfig(name="test_agent", endpoint="http://localhost:8000", enabled=True)
        registry_instance.agents["test_agent"] = agent_config
        registry_instance._build_dir = "/tmp"

        deregistration = AgentDeregistration(name="test_agent")
        response = await registry_instance.deregister_agent(deregistration)

        assert response.status_code == 200
        assert registry_instance.agents["test_agent"].enabled is False

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_agent(self, registry_instance):
        """Test deregistering nonexistent agent raises error."""
        deregistration = AgentDeregistration(name="nonexistent")

        with pytest.raises(HTTPException) as exc_info:
            await registry_instance.deregister_agent(deregistration)

        assert exc_info.value.status_code == 404


class TestAgentRegistryLifecycleActions:
    @pytest.mark.asyncio
    async def test_action_stop(self, registry_instance):
        """Test stop action."""
        agent_config = AgentConfig(name="test_agent", endpoint="http://localhost:8000", enabled=True)
        registry_instance.agents["test_agent"] = agent_config
        registry_instance.deployers["docker"] = MagicMock()

        response = await registry_instance._action_stop("test_agent", agent_config, registry_instance.deployers["docker"])

        assert registry_instance.agents["test_agent"].enabled is False

    @pytest.mark.asyncio
    async def test_action_start(self, registry_instance):
        """Test start action."""
        agent_config = AgentConfig(name="test_agent", endpoint="http://localhost:8000", enabled=False)
        registry_instance.agents["test_agent"] = agent_config
        registry_instance.deployers["docker"] = MagicMock()

        response = await registry_instance._action_start("test_agent", agent_config, registry_instance.deployers["docker"])

        assert registry_instance.agents["test_agent"].enabled is True

    @pytest.mark.asyncio
    async def test_action_delete(self, registry_instance):
        """Test delete action."""
        agent_config = AgentConfig(name="test_agent", endpoint="http://localhost:8000")
        registry_instance.agents["test_agent"] = agent_config
        registry_instance.deployers["docker"] = MagicMock()

        response = await registry_instance._action_delete("test_agent", agent_config, registry_instance.deployers["docker"])

        assert "test_agent" not in registry_instance.agents

    @pytest.mark.asyncio
    async def test_execute_lifecycle_action_unknown_agent(self, registry_instance):
        """Test executing action on unknown agent."""
        with pytest.raises(HTTPException) as exc_info:
            await registry_instance.execute_lifecycle_action("unknown", "start")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_lifecycle_action_invalid_action(self, registry_instance):
        """Test executing invalid action."""
        registry_instance.agents["test_agent"] = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            deployment_mode="docker"
        )
        registry_instance.deployers["docker"] = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await registry_instance.execute_lifecycle_action("test_agent", "invalid_action")

        assert exc_info.value.status_code == 400


class TestAgentRegistryProxyRequest:
    @pytest.mark.asyncio
    async def test_proxy_unknown_agent(self, registry_instance):
        """Test proxying to unknown agent."""
        request = MagicMock()
        request.body = AsyncMock(return_value=b"")

        with patch("oai_agent_registry.services.registry._validate_token"):
            with pytest.raises(HTTPException) as exc_info:
                await registry_instance.proxy_request("unknown", "test", request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_proxy_disabled_agent(self, registry_instance):
        """Test proxying to disabled agent."""
        registry_instance.agents["test_agent"] = AgentConfig(
            name="test_agent",
            endpoint="http://localhost:8000",
            enabled=False
        )
        request = MagicMock()
        request.body = AsyncMock(return_value=b"")

        with patch("oai_agent_registry.services.registry._validate_token"):
            with pytest.raises(HTTPException) as exc_info:
                await registry_instance.proxy_request("test_agent", "test", request)

            assert exc_info.value.status_code == 503


class TestAgentRegistryBuildSeedConfigs:
    def test_build_seed_configs_from_agents(self, registry_instance):
        """Test building seed configs from agents."""
        registry_instance.agents["agent1"] = AgentConfig(
            name="agent1",
            endpoint="http://localhost:8000",
            port=8000,
            source="https://github.com/test/repo",
            framework="langgraph",
            deployment_mode="docker"
        )

        seeds = registry_instance._build_seed_configs_from_agents()

        assert "docker" in seeds
        assert "agent1" in seeds["docker"]
        assert seeds["docker"]["agent1"]["port"] == 8000
