"""Shared pytest configuration and fixtures."""

import asyncio
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from oai_agent_registry.models import AgentConfig, RegistryConfig, Config, AgentRegistration
from oai_agent_registry.services.registry import AgentRegistry


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
        "agents": {
            "test_agent": {
                "endpoint": "http://localhost:8000",
                "enabled": False,
                "port": 8000,
                "framework": "langgraph",
                "description": "Test agent",
                "tags": ["test"],
            }
        },
        "registry": {
            "host": "0.0.0.0",
            "port": 8081,
            "enable_cors": True,
            "default_timeout": 300,
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
    registry = AgentRegistry(config_path=mock_config_file)

    # Mock the database logger
    registry.db_logger = AsyncMock()
    registry.db_logger.is_active = False
    registry.db_logger.initialize = AsyncMock()
    registry.db_logger.close = AsyncMock()
    registry.db_logger.log_agent_registration = AsyncMock()
    registry.db_logger.log_agent_action = AsyncMock()
    registry.db_logger.deregister_agent = AsyncMock()
    registry.db_logger.delete_agent = AsyncMock()
    registry.db_logger.get_all_agents = AsyncMock(return_value=[])
    registry.db_logger.get_agent_details = AsyncMock(return_value={})
    registry.db_logger.get_agent_actions = AsyncMock(return_value=[])
    registry.db_logger.get_agent_action_count = AsyncMock(return_value=0)

    # Initialize HTTP client
    registry.client = AsyncMock(spec=httpx.AsyncClient)
    registry.client.aclose = AsyncMock()

    return registry


@pytest.fixture
def agent_registration():
    """Create a sample agent registration payload."""
    return AgentRegistration(
        name="test_agent",
        description="Test agent",
        endpoint="http://localhost:8000",
        port=8000,
        framework="langgraph",
        tags=["test"],
        env_vars={"API_KEY": "secret"},
        sensitive_vars=["API_KEY"],
    )


@pytest.fixture
def agent_config():
    """Create a sample agent config."""
    return AgentConfig(
        name="test_agent",
        endpoint="http://localhost:8000",
        enabled=True,
        port=8000,
        framework="langgraph",
        description="Test agent",
        tags=["test"],
        env_vars={"API_KEY": "secret"},
        sensitive_vars=["API_KEY"],
    )
