"""Extended unit tests for oai_agent_registry.services.registry."""

import json
import asyncio
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
