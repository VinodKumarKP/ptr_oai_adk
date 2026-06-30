"""Unit tests for oai_mcp_registry.app."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from oai_mcp_registry.app import REGISTRY_ROUTES


def test_registry_routes_defined():
    """Test that REGISTRY_ROUTES is properly defined."""
    assert "register" in REGISTRY_ROUTES
    assert "deregister" in REGISTRY_ROUTES
    assert "info" in REGISTRY_ROUTES
    assert "health" in REGISTRY_ROUTES
    assert "reload-config" in REGISTRY_ROUTES
    assert "lifecycle" in REGISTRY_ROUTES
    assert "docs" in REGISTRY_ROUTES
    assert "openapi.json" in REGISTRY_ROUTES
    assert "redoc" in REGISTRY_ROUTES


def test_registry_routes_is_frozenset():
    """Test that REGISTRY_ROUTES is a frozenset."""
    assert isinstance(REGISTRY_ROUTES, frozenset)
