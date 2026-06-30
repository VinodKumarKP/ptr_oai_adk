"""Unit tests for oai_mcp_registry.services.registry."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from oai_mcp_registry.models import (
    ServerConfig,
    ServerRegistration,
    ServerDeregistration,
    RegistryConfig,
)
from oai_mcp_registry.services.registry import (
    MCPRegistry,
    _sse,
)
from fastapi import HTTPException


class TestSSE:
    def test_sse_wrapper(self):
        """Test that _sse creates a StreamingResponse."""
        async def mock_generator():
            yield "test_data"

        response = _sse(mock_generator())

        assert response.media_type == "text/event-stream"


class TestMCPRegistryInit:
    def test_init_with_config_path(self, mock_config_file):
        """Test registry initialization with config file."""
        registry = MCPRegistry(config_path=mock_config_file)

        assert registry.config_path == mock_config_file
        assert registry.config is not None
        assert registry.registry_config is not None
        assert isinstance(registry.registry_config, RegistryConfig)

    def test_init_creates_empty_deployers(self, mock_config_file):
        """Test that init creates empty deployers dict."""
        registry = MCPRegistry(config_path=mock_config_file)
        assert isinstance(registry.deployers, dict)

    def test_init_has_db_logger(self, mock_config_file):
        """Test that init creates db_logger."""
        registry = MCPRegistry(config_path=mock_config_file)
        assert hasattr(registry, "db_logger")


class TestMCPRegistryLoadConfig:
    def test_load_config_from_file(self, mock_config_file):
        """Test loading config from file."""
        registry = MCPRegistry(config_path=mock_config_file)

        assert registry.config is not None
        assert len(registry.config.servers) > 0
        assert "test_server" in registry.config.servers

    def test_load_config_missing_file(self, tmp_path):
        """Test loading config when file doesn't exist."""
        missing_path = str(tmp_path / "nonexistent.json")
        registry = MCPRegistry(config_path=missing_path)

        assert registry.config is not None
        assert len(registry.config.servers) == 0


class TestMCPRegistryBuildSeedConfigs:
    def test_build_seed_configs_from_servers(self, registry_instance):
        """Test building seed configs from servers."""
        registry_instance.config.servers["server1"] = ServerConfig(
            endpoint="http://localhost:8000",
            port=8000,
            source="https://github.com/test/repo",
            deployment_mode="docker"
        )

        seeds = registry_instance._build_seed_configs_from_servers()

        assert "docker" in seeds
        assert "server1" in seeds["docker"]
        assert seeds["docker"]["server1"]["port"] == 8000

    def test_build_seed_configs_multiple_modes(self, registry_instance):
        """Test building seed configs with multiple deployment modes."""
        registry_instance.config.servers["server1"] = ServerConfig(
            endpoint="http://localhost:8000",
            deployment_mode="docker"
        )
        registry_instance.config.servers["server2"] = ServerConfig(
            endpoint="http://localhost:8001",
            deployment_mode="kubernetes"
        )

        seeds = registry_instance._build_seed_configs_from_servers()

        assert "docker" in seeds
        assert "kubernetes" in seeds
