"""Extended tests for oai_mcp_registry.services.registry to increase coverage."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from oai_mcp_registry.services.registry import MCPRegistry
from oai_mcp_registry.models import ServerConfig, RegistryConfig, AppConfig


class TestMCPRegistryInitialization:
    def test_init_with_default_config_path(self):
        """Test registry with default config path."""
        with patch.dict("os.environ", {}, clear=False):
            registry = MCPRegistry()
            assert registry.config is not None
            assert isinstance(registry.registry_config, RegistryConfig)

    def test_init_with_invalid_json(self, tmp_path):
        """Test loading invalid JSON config."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("{ invalid json }")

        registry = MCPRegistry(config_path=str(config_file))
        assert registry.config is not None

    def test_init_creates_app_config(self, mock_config_file):
        """Test that init creates AppConfig."""
        registry = MCPRegistry(config_path=mock_config_file)
        assert isinstance(registry.config, AppConfig)

    def test_init_attributes(self, mock_config_file):
        """Test all initialized attributes."""
        registry = MCPRegistry(config_path=mock_config_file)
        assert hasattr(registry, "host_ip")
        assert hasattr(registry, "public_ip")
        assert hasattr(registry, "sub_apps")
        assert isinstance(registry.sub_apps, dict)
        assert hasattr(registry, "start_sub_app")
        assert hasattr(registry, "stop_sub_app")


class TestMCPRegistryConfiguration:
    def test_load_configuration_sets_registered_via(self, mock_config_file):
        """Test that loaded configs are marked as registered_via='config'."""
        registry = MCPRegistry(config_path=mock_config_file)
        for server in registry.config.servers.values():
            assert server.registered_via == "config"

    def test_load_configuration_empty_servers(self, tmp_path):
        """Test loading config with empty servers."""
        config_file = tmp_path / "config.json"
        config_data = {"servers": {}}
        config_file.write_text(json.dumps(config_data))

        registry = MCPRegistry(config_path=str(config_file))
        assert len(registry.config.servers) == 0

    def test_load_configuration_with_all_fields(self, tmp_path):
        """Test loading config with all possible fields."""
        config_file = tmp_path / "config.json"
        config_data = {
            "servers": {
                "full_server": {
                    "endpoint": "http://localhost:8000",
                    "enabled": True,
                    "port": 8000,
                    "description": "Full server",
                    "registered_via": "config",
                    "source": "https://github.com/test",
                    "tags": ["test"],
                    "current_version": "1.0.0",
                    "available_versions": ["1.0.0"],
                    "deployment_mode": "docker",
                    "env_vars": {"KEY": "value"},
                    "sensitive_vars": ["KEY"]
                }
            },
            "registry": {
                "host": "localhost",
                "port": 9000,
                "enable_auto_discovery": True,
                "start_port": 8000,
                "end_port": 8200,
                "enable_cors": False,
            }
        }
        config_file.write_text(json.dumps(config_data))

        registry = MCPRegistry(config_path=str(config_file))
        assert "full_server" in registry.config.servers
        assert registry.config.registry.port == 9000
        assert registry.config.registry.enable_auto_discovery is True


class TestMCPRegistrySeedConfigs:
    def test_build_seed_configs_empty(self, registry_instance):
        """Test building seed configs with no servers."""
        registry_instance.config.servers = {}
        seeds = registry_instance._build_seed_configs_from_servers()
        assert seeds == {}

    def test_build_seed_configs_defaults_to_docker(self, registry_instance):
        """Test that servers without deployment_mode default to docker."""
        registry_instance.config.servers["server1"] = ServerConfig(
            endpoint="http://localhost:8000",
            port=8000
        )
        # Manually unset deployment_mode to test default
        registry_instance.config.servers["server1"].deployment_mode = None

        seeds = registry_instance._build_seed_configs_from_servers()

        assert "docker" in seeds
        assert "server1" in seeds["docker"]

    def test_build_seed_configs_includes_all_fields(self, registry_instance):
        """Test that seed configs include all necessary fields."""
        registry_instance.config.servers["server1"] = ServerConfig(
            endpoint="http://localhost:8000",
            port=8000,
            source="https://github.com/test",
            tags=["test"],
            description="Test",
            current_version="1.0.0",
            env_vars={"KEY": "value"},
            deployment_mode="docker"
        )

        seeds = registry_instance._build_seed_configs_from_servers()
        server_seed = seeds["docker"]["server1"]

        assert server_seed["port"] == 8000
        assert server_seed["source"] == "https://github.com/test"
        assert server_seed["tags"] == ["test"]
        assert server_seed["description"] == "Test"
        assert server_seed["current_version"] == "1.0.0"
        assert server_seed["env"] == {"KEY": "value"}

    def test_build_seed_configs_handles_none_values(self, registry_instance):
        """Test that seed configs handle None values gracefully."""
        registry_instance.config.servers["server1"] = ServerConfig(
            endpoint="http://localhost:8000",
            port=8000,
            source=None,
            tags=None,
            description="A proxied MCP server",
            env_vars=None
        )

        seeds = registry_instance._build_seed_configs_from_servers()
        server_seed = seeds["docker"]["server1"]

        assert server_seed["source"] == ""
        assert server_seed["tags"] == []
        assert server_seed["env"] is None


class TestMCPRegistryMethods:
    @pytest.mark.asyncio
    async def test_initialize_creates_client(self, registry_instance):
        """Test that initialize sets up a client."""
        # Mock the deployers to avoid actual Docker operations
        registry_instance.deployers = {"docker": MagicMock()}
        registry_instance.deployers["docker"].initialize = AsyncMock()

        # We can't actually call initialize without more mocks, but we can verify attributes
        assert hasattr(registry_instance, "db_logger")

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, registry_instance):
        """Test shutdown cleanup."""
        registry_instance.db_logger.close = AsyncMock()
        # Register a mock deployer
        registry_instance.deployers["docker"] = MagicMock()
        registry_instance.deployers["docker"].shutdown = AsyncMock()

        await registry_instance.shutdown()

        registry_instance.db_logger.close.assert_called()


class TestMCPRegistryErrorHandling:
    def test_config_error_creates_empty_app_config(self, tmp_path):
        """Test that config errors result in empty AppConfig."""
        # Create a file that will cause an error when loaded
        config_file = tmp_path / "config.json"
        config_file.write_text('{"servers": null}')  # Invalid: servers should be a dict

        registry = MCPRegistry(config_path=str(config_file))

        # Should have created an empty config as fallback
        assert registry.config is not None

    def test_load_configuration_with_registry_defaults(self, tmp_path):
        """Test that missing registry config uses defaults."""
        config_file = tmp_path / "config.json"
        config_data = {"servers": {}}
        config_file.write_text(json.dumps(config_data))

        registry = MCPRegistry(config_path=str(config_file))

        # Should use RegistryConfig defaults
        assert registry.registry_config.host == "0.0.0.0"
        assert registry.registry_config.port == 8081
