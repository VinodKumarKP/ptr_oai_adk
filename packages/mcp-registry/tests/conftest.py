"""Shared pytest configuration and fixtures for MCP registry tests."""

import asyncio
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from oai_mcp_registry.models import (
    ServerConfig,
    RegistryConfig,
    AppConfig,
    ServerRegistration,
)
from oai_mcp_registry.services.registry import MCPRegistry


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "servers": {
            "test_server": {
                "endpoint": "http://localhost:8000",
                "enabled": False,
                "port": 8000,
                "description": "Test server",
                "tags": ["test"],
            }
        },
        "registry": {
            "host": "0.0.0.0",
            "port": 8081,
            "enable_cors": True,
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def registry_instance(mock_config_file):
    """Create a registry instance with mocked database."""
    registry = MCPRegistry(config_path=mock_config_file)

    # Mock the database logger
    registry.db_logger = AsyncMock()
    registry.db_logger.is_active = False
    registry.db_logger.initialize = AsyncMock()
    registry.db_logger.close = AsyncMock()
    registry.db_logger.log_server_registration = AsyncMock()
    registry.db_logger.log_server_action = AsyncMock()
    registry.db_logger.deregister_server = AsyncMock()
    registry.db_logger.delete_server = AsyncMock()
    registry.db_logger.get_all_servers = AsyncMock(return_value=[])
    registry.db_logger.get_server_details = AsyncMock(return_value={})
    registry.db_logger.get_server_actions = AsyncMock(return_value=[])
    registry.db_logger.get_server_action_count = AsyncMock(return_value=0)

    return registry


@pytest.fixture
def server_registration():
    """Create a sample server registration payload."""
    return ServerRegistration(
        name="test_server",
        description="Test server",
        endpoint="http://localhost:8000",
        port=8000,
        tags=["test"],
        env_vars={"API_KEY": "secret"},
        sensitive_vars=["API_KEY"],
    )


@pytest.fixture
def server_config():
    """Create a sample server config."""
    return ServerConfig(
        endpoint="http://localhost:8000",
        enabled=True,
        port=8000,
        description="Test server",
        tags=["test"],
        env_vars={"API_KEY": "secret"},
        sensitive_vars=["API_KEY"],
    )
