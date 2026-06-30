"""Unit tests for oai_agent_registry.dependencies."""

import pytest

from oai_agent_registry.dependencies import get_registry, registry_instance
from oai_agent_registry.services.registry import AgentRegistry


class TestDependencies:
    def test_registry_instance_is_agent_registry(self):
        """Test that registry_instance is an AgentRegistry."""
        assert isinstance(registry_instance, AgentRegistry)

    def test_get_registry_returns_instance(self):
        """Test that get_registry returns the registry_instance."""
        result = get_registry()
        assert result is registry_instance

    def test_get_registry_is_callable(self):
        """Test that get_registry is callable."""
        assert callable(get_registry)

    def test_registry_instance_has_agents(self):
        """Test that registry_instance has agents attribute."""
        assert hasattr(registry_instance, "agents")
        assert isinstance(registry_instance.agents, dict)

    def test_registry_instance_has_config(self):
        """Test that registry_instance has config attribute."""
        assert hasattr(registry_instance, "config")

    def test_registry_instance_has_registry_config(self):
        """Test that registry_instance has registry_config attribute."""
        assert hasattr(registry_instance, "registry_config")

    def test_registry_instance_has_db_logger(self):
        """Test that registry_instance has db_logger attribute."""
        assert hasattr(registry_instance, "db_logger")

    def test_registry_instance_has_deployers(self):
        """Test that registry_instance has deployers attribute."""
        assert hasattr(registry_instance, "deployers")
        assert isinstance(registry_instance.deployers, dict)

    def test_multiple_get_registry_calls_return_same_instance(self):
        """Test that multiple calls to get_registry return the same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_registry_instance_has_client(self):
        """Test that registry_instance has client attribute."""
        assert hasattr(registry_instance, "client")

    def test_registry_instance_has_start_time(self):
        """Test that registry_instance has start_time attribute."""
        assert hasattr(registry_instance, "start_time")
        assert isinstance(registry_instance.start_time, float)
