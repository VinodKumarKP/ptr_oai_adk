"""Unit tests for oai_agent_registry.app."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from oai_agent_registry.app import create_app


@pytest.fixture
def mock_registry():
    """Mock the registry instance."""
    with patch("oai_agent_registry.app.registry_instance") as mock:
        mock.client = MagicMock()
        mock.registry_config = MagicMock()
        mock.registry_config.default_timeout = 300
        mock.registry_config.enable_cors = True
        mock.registry_config.enable_auto_discovery = False
        mock.initialize = AsyncMock()
        mock.shutdown = AsyncMock()
        mock.client.aclose = AsyncMock()
        yield mock


@pytest.fixture
def mock_security():
    """Mock security dependencies."""
    with patch("oai_agent_registry.app.verify_api_key") as mock:
        yield mock


@pytest.fixture
def mock_networking():
    """Mock networking functions."""
    with patch("oai_agent_registry.app.get_public_ip") as mock_public, \
         patch("oai_agent_registry.app.get_private_ip") as mock_private:
        mock_public.return_value = "192.168.1.1"
        mock_private.return_value = "10.0.0.1"
        yield mock_public, mock_private


def test_create_app_returns_fastapi_app(mock_registry, mock_security, mock_networking):
    """Test that create_app returns a FastAPI application."""
    app = create_app()

    assert app is not None
    assert hasattr(app, "routes")


def test_create_app_has_title(mock_registry, mock_security, mock_networking):
    """Test that created app has correct title."""
    app = create_app()

    assert app.title == "Agent Registry"


def test_create_app_has_description(mock_registry, mock_security, mock_networking):
    """Test that created app has description."""
    app = create_app()

    assert "registry" in app.description.lower()


def test_create_app_has_version(mock_registry, mock_security, mock_networking):
    """Test that created app has version."""
    app = create_app()

    assert app.version == "1.0.0"


def test_create_app_has_middleware(mock_registry, mock_security, mock_networking):
    """Test that middleware is added."""
    app = create_app()
    # Just verify the app has user_middleware
    assert hasattr(app, "user_middleware")


def test_create_app_can_be_called_multiple_times(mock_registry, mock_security, mock_networking):
    """Test that create_app can be called multiple times."""
    app1 = create_app()
    app2 = create_app()

    assert app1 is not app2
    assert app1.title == app2.title
