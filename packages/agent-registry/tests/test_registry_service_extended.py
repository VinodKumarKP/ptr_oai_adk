"""Extended unit tests for oai_agent_registry.services.registry."""

import json
import asyncio
from fastapi import Request
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from oai_agent_registry.models import (
    AgentConfig,
    AgentRegistration,
    AgentDeregistration,
    Config,
    RegistryConfig,
)
from oai_agent_registry.services.registry import AgentRegistry


@pytest.fixture
def test_registry_instance(mock_config_file):
    """Create a test registry with minimal mocking."""
    registry = AgentRegistry(config_path=mock_config_file)
    registry.db_logger = AsyncMock()
    registry.db_logger.is_active = False
    registry.client = AsyncMock()
    registry._build_dir = "/tmp/test_build"
    return registry


class TestRegistryLoadConfig:
    def test_load_config_with_bad_json(self, tmp_path):
        """Test loading config with malformed JSON."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(Exception):
            registry = AgentRegistry(config_path=str(config_file))


class TestRegistryDiscoverAgents:
    @pytest.mark.asyncio
    async def test_discover_agents_success(self, test_registry_instance):
        """Test successful agent discovery."""
        test_registry_instance.client.get = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "agent_name": "discovered_agent",
                    "description": "Discovered",
                    "framework": "langgraph",
                    "source": "http://example.com"
                })
            )
        )
        test_registry_instance.db_logger.log_agent_registration = AsyncMock()

        await test_registry_instance.discover_agents()

        assert "discovered_agent" in test_registry_instance.agents


class TestRegistryReloadConfig:
    @pytest.mark.asyncio
    async def test_reload_config(self, test_registry_instance):
        """Test reloading configuration."""
        test_registry_instance._sync_agents_to_db = AsyncMock()
        test_registry_instance._check_agent_statuses = AsyncMock()

        response = await test_registry_instance.reload_config()

        assert response.status_code == 200
        test_registry_instance._sync_agents_to_db.assert_called()


class TestRegistryPersistAgent:
    @pytest.mark.asyncio
    async def test_persist_agent_to_db_success(self, test_registry_instance):
        """Test persisting agent to database."""
        agent_config = AgentConfig(
            name="test",
            endpoint="http://localhost:8000",
            enabled=True,
            framework="langgraph"
        )
        test_registry_instance.db_logger.log_agent_registration = AsyncMock()

        result = await test_registry_instance._persist_agent_to_db("test", agent_config)

        assert result is True
        test_registry_instance.db_logger.log_agent_registration.assert_called()

    @pytest.mark.asyncio
    async def test_persist_agent_to_db_failure(self, test_registry_instance):
        """Test failing to persist agent to database."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.db_logger.log_agent_registration = AsyncMock(
            side_effect=Exception("DB Error")
        )

        result = await test_registry_instance._persist_agent_to_db("test", agent_config)

        assert result is False


class TestRegistryGetMergedValues:
    @pytest.mark.asyncio
    async def test_get_merged_agent_values(self, test_registry_instance):
        """Test merging new registration with existing values."""
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.get_agent_details = AsyncMock(return_value={
            "endpoint_url": "http://old:8000",
            "port": 8000,
            "framework": "old_framework"
        })

        registration = AgentRegistration(
            name="test",
            endpoint="http://new:8000",
            framework="langgraph"
        )

        merged = await test_registry_instance._get_merged_agent_values("test", registration, "dynamic")

        assert merged["endpoint"] == "http://new:8000"
        assert merged["framework"] == "langgraph"
        assert merged["registered_via"] == "dynamic"

    @pytest.mark.asyncio
    async def test_get_merged_agent_values_preserve_old(self, test_registry_instance):
        """Test that old values are preserved when new ones are None."""
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.get_agent_details = AsyncMock(return_value={
            "endpoint_url": "http://old:8000",
            "port": 8000,
            "available_versions": ["1.0", "2.0"]
        })

        registration = AgentRegistration(name="test")

        merged = await test_registry_instance._get_merged_agent_values("test", registration, "dynamic")

        assert merged["endpoint"] == "http://old:8000"
        assert merged["port"] == 8000
        assert merged["available_versions"] == ["1.0", "2.0"]


class TestRegistrySyncAgentsToDb:
    @pytest.mark.asyncio
    async def test_sync_agents_to_db(self, test_registry_instance):
        """Test syncing config agents to database."""
        test_registry_instance.agents["config_agent"] = AgentConfig(
            name="config_agent",
            endpoint="http://localhost:8000",
            registered_via="config"
        )
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.log_agent_registration = AsyncMock()

        await test_registry_instance._sync_agents_to_db()

        test_registry_instance.db_logger.log_agent_registration.assert_called()


class TestRegistryLoadAgentsFromDb:
    @pytest.mark.asyncio
    async def test_load_agents_from_db(self, test_registry_instance):
        """Test loading agents from database."""
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.get_all_agents = AsyncMock(return_value=[
            {
                "agent_name": "db_agent",
                "endpoint_url": "http://localhost:8000",
                "port": 8000,
                "framework": "langgraph",
                "registered_via": "dynamic"
            }
        ])

        await test_registry_instance._load_agents_from_db()

        assert "db_agent" in test_registry_instance.agents


class TestRegistryCheckAgentStatuses:
    @pytest.mark.asyncio
    async def test_check_agent_statuses_healthy(self, test_registry_instance):
        """Test checking healthy agent statuses."""
        test_registry_instance.agents["test"] = AgentConfig(
            name="test",
            endpoint="http://localhost:8000",
            enabled=False,
            timeout=300
        )
        test_registry_instance.client.get = AsyncMock(
            return_value=MagicMock(status_code=200)
        )

        await test_registry_instance._check_agent_statuses()

        assert test_registry_instance.agents["test"].enabled is True

    @pytest.mark.asyncio
    async def test_check_agent_statuses_unhealthy(self, test_registry_instance):
        """Test checking unhealthy agent statuses."""
        test_registry_instance.agents["test"] = AgentConfig(
            name="test",
            endpoint="http://localhost:8000",
            enabled=True,
            timeout=300
        )
        test_registry_instance.client.get = AsyncMock(
            return_value=MagicMock(status_code=500)
        )
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.deregister_agent = AsyncMock()

        await test_registry_instance._check_agent_statuses()

        assert test_registry_instance.agents["test"].enabled is False


class TestRegistryActionVersionSwitch:
    @pytest.mark.asyncio
    async def test_action_version_switch_no_version(self, test_registry_instance):
        """Test version switch without version parameter."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")

        with pytest.raises(Exception):  # HTTPException
            await test_registry_instance._action_version_switch(
                "test", agent_config, None, None, False
            )

    @pytest.mark.asyncio
    async def test_action_version_switch_with_version(self, test_registry_instance):
        """Test version switch with version parameter."""
        agent_config = AgentConfig(
            name="test",
            endpoint="http://localhost:8000",
            current_version="1.0.0",
            available_versions=["1.0.0"]
        )
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.log_agent_action = AsyncMock()

        response = await test_registry_instance._action_version_switch(
            "test", agent_config, None, "2.0.0", False
        )

        assert response.status_code == 200
        assert agent_config.current_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_action_version_switch_with_deployer(self, test_registry_instance):
        """Test version switch action with mock deployer (streaming and non-streaming)."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000", current_version="1.0.0")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        
        mock_deployer = MagicMock()
        mock_deployer.deploy_agent = AsyncMock()
        mock_deployer.image_exists.return_value = False
        
        async def fake_stream(*args, **kwargs):
            yield "Switching version..."
        mock_deployer.stream_deploy_agent = fake_stream
        
        # Non-streaming
        res = await test_registry_instance._action_version_switch("test", agent_config, mock_deployer, "2.0.0", False)
        assert res.status_code == 200
        mock_deployer.deploy_agent.assert_called_once()
        
        # Streaming
        res = await test_registry_instance._action_version_switch("test", agent_config, mock_deployer, "2.0.0", True)
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Agent 'test' switching to version 2.0.0...\n\n" in body
        assert "data: Switching version...\n\n" in body


class TestRegistryActionRestart:
    @pytest.mark.asyncio
    async def test_action_restart_no_deployer(self, test_registry_instance):
        """Test restart without deployer."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config

        response = await test_registry_instance._action_restart(
            "test", agent_config, None, False
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_action_restart_with_deployer(self, test_registry_instance):
        """Test restart action with mock deployer (streaming and non-streaming)."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        
        mock_deployer = MagicMock()
        mock_deployer.deploy_agent = AsyncMock()
        
        # Non-streaming
        res = await test_registry_instance._action_restart("test", agent_config, mock_deployer, False)
        assert res.status_code == 200
        mock_deployer.deploy_agent.assert_called_once()
        
        # Streaming
        res = await test_registry_instance._action_restart("test", agent_config, mock_deployer, True)
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Restarting 'test'...\n\n" in body
        assert "data: ✓ Agent 'test' restarted.\n\n" in body


class TestRegistryActionRebuild:
    @pytest.mark.asyncio
    async def test_action_rebuild_no_deployer(self, test_registry_instance):
        """Test rebuild without deployer."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config

        response = await test_registry_instance._action_rebuild(
            "test", agent_config, None, False
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_action_rebuild_with_deployer(self, test_registry_instance):
        """Test rebuild action with mock deployer (streaming and non-streaming)."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        
        mock_deployer = MagicMock()
        mock_deployer.deploy_agent = AsyncMock()
        mock_deployer.remove_agent = MagicMock()
        
        async def fake_stream(*args, **kwargs):
            yield "Rebuilding..."
        mock_deployer.stream_deploy_agent = fake_stream
        
        # Non-streaming
        res = await test_registry_instance._action_rebuild("test", agent_config, mock_deployer, False)
        assert res.status_code == 200
        
        # Give asyncio tasks a chance to run
        await asyncio.sleep(0.01)
        mock_deployer.deploy_agent.assert_called_once()
        
        # Streaming
        res = await test_registry_instance._action_rebuild("test", agent_config, mock_deployer, True)
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Agent 'test' stopped. Rebuilding image...\n\n" in body
        assert "data: Rebuilding...\n\n" in body


class TestRegistryActionRedeploy:
    @pytest.mark.asyncio
    async def test_action_redeploy_no_deployer(self, test_registry_instance):
        """Test redeploy without deployer."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config

        response = await test_registry_instance._action_redeploy(
            "test", agent_config, None, False
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_action_redeploy_with_deployer(self, test_registry_instance):
        """Test redeploy action with mock deployer (streaming and non-streaming)."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        
        mock_deployer = MagicMock()
        mock_deployer.deploy_agent = AsyncMock()
        mock_deployer.remove_agent = MagicMock()
        
        # Non-streaming
        res = await test_registry_instance._action_redeploy("test", agent_config, mock_deployer, False)
        assert res.status_code == 200
        
        # Give asyncio tasks a chance to run
        await asyncio.sleep(0.01)
        mock_deployer.deploy_agent.assert_called_once()
        
        # Streaming
        res = await test_registry_instance._action_redeploy("test", agent_config, mock_deployer, True)
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Starting async redeploy for agent 'test'...\n\n" in body
        assert "data: ✅ Redeploy task initiated for 'test'.\n\n" in body


class TestRegistryUpdateEnvVars:
    @pytest.mark.asyncio
    async def test_update_server_env_vars_unknown_agent(self, test_registry_instance):
        """Test updating env vars for unknown agent."""
        with pytest.raises(Exception):  # HTTPException
            await test_registry_instance.update_server_env_vars(
                "unknown", {"KEY": "value"}, ["KEY"]
            )

    @pytest.mark.asyncio
    async def test_update_server_env_vars_known_agent(self, test_registry_instance):
        """Test updating env vars for known agent."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance.db_logger.is_active = True
        test_registry_instance.db_logger.log_agent_action = AsyncMock()

        response = await test_registry_instance.update_server_env_vars(
            "test", {"KEY": "value"}, ["KEY"]
        )

        assert agent_config.env_vars == {"KEY": "value"}
        assert agent_config.sensitive_vars == ["KEY"]

    @pytest.mark.asyncio
    async def test_update_server_env_vars_with_deployer(self, test_registry_instance):
        """Test updating env vars with mock deployer."""
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000", deployment_mode="docker")
        test_registry_instance.agents["test"] = agent_config
        test_registry_instance._persist_agent_to_db = AsyncMock(return_value=True)
        test_registry_instance._write_prometheus_targets = MagicMock()
        
        mock_deployer = MagicMock()
        async def fake_stream(*args, **kwargs):
            yield "Updating env..."
        mock_deployer.stream_deploy_agent = fake_stream
        test_registry_instance._get_deployer = MagicMock(return_value=mock_deployer)
        
        res = await test_registry_instance.update_server_env_vars(
            "test", {"KEY": "value"}, ["KEY"]
        )
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Saved 1 variable(s) for 'test'.\n\n" in body
        assert "data: Updating env...\n\n" in body
        assert "data: ✓ Agent 'test' redeployed with updated environment variables.\n\n" in body


class TestRegistryCleanResponseHeaders:
    def test_clean_response_headers(self):
        """Test cleaning response headers."""
        from httpx import Headers

        headers = Headers({
            "content-type": "application/json",
            "content-length": "100",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "custom-header": "value"
        })

        cleaned = AgentRegistry._clean_response_headers(headers)

        assert "content-type" in cleaned
        assert "custom-header" in cleaned
        assert "content-encoding" not in cleaned
        assert "content-length" not in cleaned
        assert "transfer-encoding" not in cleaned
        assert "connection" not in cleaned


class FakeRequest:
    def __init__(self, method, path, headers, body=b""):
        self.method = method
        self.url = MagicMock()
        self.url.path = path
        self.url.query = ""
        self.headers = headers
        self.client = MagicMock()
        self.client.host = "127.0.0.1"
        self._body = body

    async def stream(self):
        yield self._body

    async def body(self):
        return self._body


class TestRegistryProxyRequest:
    @pytest.mark.asyncio
    async def test_proxy_request_unknown_agent(self, test_registry_instance):
        """Test proxying request to an unknown agent."""
        import os
        from fastapi import HTTPException
        mock_request = FakeRequest("GET", "/unknown/path", {})
        with patch.dict(os.environ, {"AGENT_AUTH_ENABLED": "false"}):
            with pytest.raises(HTTPException) as exc:
                await test_registry_instance.proxy_request("unknown", "path", mock_request)
            assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_proxy_request_no_endpoint(self, test_registry_instance):
        """Test proxying request to agent with no endpoint."""
        import os
        from fastapi import HTTPException
        agent_config = AgentConfig(name="test", enabled=False)
        test_registry_instance.agents["test"] = agent_config
        mock_request = FakeRequest("GET", "/test/path", {})
        with patch.dict(os.environ, {"AGENT_AUTH_ENABLED": "false"}):
            with pytest.raises(HTTPException) as exc:
                await test_registry_instance.proxy_request("test", "path", mock_request)
            assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_proxy_regular_request_success(self, test_registry_instance):
        """Test regular HTTP proxy request success."""
        import os
        from starlette.datastructures import Headers
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8080")
        test_registry_instance.agents["test"] = agent_config
        
        headers = Headers({"content-type": "application/json", "host": "localhost"})
        mock_request = FakeRequest("POST", "/test/api/path", headers, b'{"key": "value"}')
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"status": "ok"}'
        
        test_registry_instance.client.request = AsyncMock(return_value=mock_response)
        
        with patch.dict(os.environ, {"AGENT_AUTH_ENABLED": "false"}):
            response = await test_registry_instance.proxy_request("test", "api/path", mock_request)
        assert response.status_code == 200
        assert response.body == b'{"status": "ok"}'

    @pytest.mark.asyncio
    @patch("oai_agent_registry.services.registry.httpx.AsyncClient")
    async def test_proxy_streaming_request_success(self, mock_client_cls, test_registry_instance):
        """Test streaming HTTP proxy request success."""
        import os
        from starlette.datastructures import Headers
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8080")
        test_registry_instance.agents["test"] = agent_config
        
        headers = Headers({"accept": "text/event-stream", "host": "localhost"})
        mock_request = FakeRequest("GET", "/test/stream/path", headers)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        
        async def fake_response_stream():
            yield b"data: event 1\n\n"
        mock_response.aiter_bytes = fake_response_stream
        
        mock_client = MagicMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        with patch.dict(os.environ, {"AGENT_AUTH_ENABLED": "false"}):
            response = await test_registry_instance.proxy_request("test", "stream/path", mock_request)
        assert response.status_code == 200
        body = [chunk async for chunk in response.body_iterator]
        assert b"data: event 1\n\n" in body


class TestRegistryLifecycleAndRegistration:
    @pytest.mark.asyncio
    async def test_register_agent_non_streaming(self, test_registry_instance):
        """Test register_agent in non-streaming mode."""
        mock_deployer = MagicMock()
        mock_deployer.find_available_port.return_value = 1234
        mock_deployer.deploy_agent = AsyncMock()
        
        test_registry_instance._get_deployer = MagicMock(return_value=mock_deployer)
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        test_registry_instance._get_merged_agent_values = AsyncMock(return_value={
            "name": "test_agent",
            "endpoint": "http://localhost:1234",
            "deployment_mode": "docker",
        })
        
        reg = AgentRegistration(
            name="test_agent",
            deployment_mode="docker",
            registered_via="registry",
        )
        
        res = await test_registry_instance.register_agent(reg, stream_output=False)
        assert res.status_code == 200
        
        await asyncio.sleep(0.01)
        mock_deployer.deploy_agent.assert_called_once()
        test_registry_instance._persist_agent_to_db.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_register_agent_streaming(self, test_registry_instance):
        """Test register_agent in streaming mode."""
        mock_deployer = MagicMock()
        mock_deployer.find_available_port.return_value = 1234
        
        async def fake_stream_deploy(*args, **kwargs):
            yield "Step 1"
            yield "Step 2"
        mock_deployer.stream_deploy_agent = fake_stream_deploy
        
        test_registry_instance._get_deployer = MagicMock(return_value=mock_deployer)
        test_registry_instance._persist_agent_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        test_registry_instance._get_merged_agent_values = AsyncMock(return_value={
            "name": "test_agent",
            "endpoint": "http://localhost:1234",
            "deployment_mode": "docker",
        })
        
        reg = AgentRegistration(
            name="test_agent",
            deployment_mode="docker",
            registered_via="registry",
        )
        
        res = await test_registry_instance.register_agent(reg, stream_output=True)
        assert res.status_code == 200
        body = [chunk async for chunk in res.body_iterator]
        assert "data: Step 1\n\n" in body
        assert "data: Step 2\n\n" in body
        assert "data: ✅ Agent 'test_agent' registered successfully.\n\n" in body

    @pytest.mark.asyncio
    async def test_deregister_agent_unknown(self, test_registry_instance):
        """Test deregistering unknown agent raises 404."""
        from fastapi import HTTPException
        dereg = AgentDeregistration(name="unknown")
        with pytest.raises(HTTPException) as exc:
            await test_registry_instance.deregister_agent(dereg)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_info_and_health_check(self, test_registry_instance):
        """Test get_info, health_check and reload_config."""
        test_registry_instance.db_logger.get_history = AsyncMock(return_value=[])
        
        res = await test_registry_instance.get_info()
        assert res.status_code == 200
        
        res = await test_registry_instance.health_check()
        assert res.status_code == 200
        
        with patch.object(test_registry_instance, "load_config") as mock_load:
            res = await test_registry_instance.reload_config()
            assert res.status_code == 200
            mock_load.assert_called_once()


class TestRegistryBuildSeedConfigs:
    def test_build_seed_configs_from_agents(self, test_registry_instance):
        """Test building seed configurations from agents."""
        agent_config = AgentConfig(
            name="test",
            endpoint="http://localhost:8000",
            deployment_mode="docker",
            framework="openai",
            env_vars={"KEY": "val"},
            current_version="1.0.0"
        )
        test_registry_instance.agents["test"] = agent_config
        seeds = test_registry_instance._build_seed_configs_from_agents()
        assert "docker" in seeds
        assert "test" in seeds["docker"]
        assert seeds["docker"]["test"]["env"] == {"KEY": "val"}


class TestRegistryInitialize:
    @pytest.mark.asyncio
    @patch("oai_agent_registry.services.registry.DeployerFactory.get_deployer")
    @patch("oai_agent_registry.services.infra_manager.InfraManager.wait_for_postgres")
    async def test_initialize_with_auto_start_infra(self, mock_wait, mock_get_deployer, test_registry_instance):
        """Test initialize and _auto_start_infra flow."""
        # 1. Configure registry for auto start infra
        test_registry_instance.registry_config.auto_start_infra = True
        test_registry_instance.registry_config.infra_startup_timeout = 10
        
        # 2. Mock early deployer
        mock_early_deployer = MagicMock()
        mock_early_deployer.start_infra_services = MagicMock()
        mock_get_deployer.return_value = mock_early_deployer
        
        # 3. Mock other methods
        test_registry_instance.db_logger = AsyncMock()
        test_registry_instance.db_logger.is_active = True
        test_registry_instance._load_agents_from_db = AsyncMock()
        test_registry_instance._sync_agents_to_db = AsyncMock()
        test_registry_instance._write_prometheus_targets = MagicMock()
        test_registry_instance._check_agent_statuses = AsyncMock()
        
        # 4. Run initialize
        await test_registry_instance.initialize()
        
        # 5. Assertions
        mock_wait.assert_called_once_with(timeout=10)
        mock_early_deployer.start_infra_services.assert_any_call()
        test_registry_instance.db_logger.initialize.assert_called_once()


class TestRegistryShutdown:
    @pytest.mark.asyncio
    async def test_shutdown(self, test_registry_instance):
        """Test shutdown sequence."""
        test_registry_instance.client = AsyncMock()
        test_registry_instance.db_logger = AsyncMock()
        
        agent_config = AgentConfig(name="test", endpoint="http://localhost:8000", registered_via="config", deployment_mode="docker")
        test_registry_instance.agents = {"test": agent_config}
        
        mock_deployer = MagicMock()
        mock_deployer.shutdown = AsyncMock()
        test_registry_instance.deployers["docker"] = mock_deployer
        test_registry_instance._get_deployer = MagicMock(return_value=mock_deployer)
        
        await test_registry_instance.shutdown()
        
        test_registry_instance.client.aclose.assert_called_once()
        mock_deployer.remove_agent.assert_called_once_with("test")
        mock_deployer.shutdown.assert_called_once()
        test_registry_instance.db_logger.close.assert_called_once()
