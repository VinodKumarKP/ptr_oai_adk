"""Unit tests for oai_agent_registry.exceptions."""

import pytest
from oai_agent_registry.exceptions import (
    AgentRegistryException,
    AgentNotFoundException,
    AgentNotEnabledException,
    AuthenticationException,
)


class TestAgentRegistryException:
    def test_raise_base_exception(self):
        with pytest.raises(AgentRegistryException):
            raise AgentRegistryException("Test error")

    def test_exception_message(self):
        exc = AgentRegistryException("Test message")
        assert str(exc) == "Test message"

    def test_exception_inheritance(self):
        assert issubclass(AgentRegistryException, Exception)


class TestAgentNotFoundException:
    def test_raise_exception(self):
        with pytest.raises(AgentNotFoundException):
            raise AgentNotFoundException("Agent not found")

    def test_is_agent_registry_exception(self):
        assert issubclass(AgentNotFoundException, AgentRegistryException)

    def test_can_catch_as_base_exception(self):
        with pytest.raises(AgentRegistryException):
            raise AgentNotFoundException("Agent not found")


class TestAgentNotEnabledException:
    def test_raise_exception(self):
        with pytest.raises(AgentNotEnabledException):
            raise AgentNotEnabledException("Agent is disabled")

    def test_is_agent_registry_exception(self):
        assert issubclass(AgentNotEnabledException, AgentRegistryException)

    def test_can_catch_as_base_exception(self):
        with pytest.raises(AgentRegistryException):
            raise AgentNotEnabledException("Agent is disabled")


class TestAuthenticationException:
    def test_raise_exception(self):
        with pytest.raises(AuthenticationException):
            raise AuthenticationException("Auth failed")

    def test_is_agent_registry_exception(self):
        assert issubclass(AuthenticationException, AgentRegistryException)

    def test_can_catch_as_base_exception(self):
        with pytest.raises(AgentRegistryException):
            raise AuthenticationException("Auth failed")

    def test_multiple_inheritance(self):
        exc = AuthenticationException("Test")
        assert isinstance(exc, AgentRegistryException)
