"""Unit tests for oai_agent_registry.utils.env_vars."""

import pytest
from unittest.mock import patch

from oai_agent_registry.utils.env_vars import get_common_agent_env


class TestGetCommonAgentEnv:
    def test_docker_deployment_mode(self):
        """Test env vars for docker deployment."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://host.docker.internal:8081",
            deployment_mode="docker"
        )

        assert env["AGENT_NAME"] == "test_agent"
        assert env["PORT"] == "8000"
        assert env["AGENT_AUTH_ENABLED"] == "true"
        assert env["AGENT_BASE_URL"] == "http://localhost"
        assert env["AGENT_REGISTRY_URL"] == "http://host.docker.internal:8081"
        assert env["FORCE_AUTH"] == "false"
        assert env["PROMETHEUS_ENABLED"] == "true"
        assert env["OTEL_SERVICE_NAME"] == "test_agent"
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://jaeger:4317"
        assert "agent-valkey" in env.get("REDIS_HOST", "")
        assert "agent_logs_db" in env.get("LOGGING_DB_HOST", "")

    def test_local_deployment_mode(self):
        """Test env vars for local deployment."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            deployment_mode="local"
        )

        assert env["AGENT_NAME"] == "test_agent"
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:4317"
        assert "localhost" in env.get("REDIS_HOST", "")
        assert "localhost" in env.get("LOGGING_DB_HOST", "")

    def test_with_env_overrides(self):
        """Test that env_overrides are applied."""
        overrides = {
            "CUSTOM_VAR": "custom_value",
            "AGENT_NAME": "overridden_name",
            "AGENT_AUTH_ENABLED": "false"
        }
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides=overrides
        )

        assert env["CUSTOM_VAR"] == "custom_value"
        assert env["AGENT_NAME"] == "overridden_name"
        assert env["AGENT_AUTH_ENABLED"] == "false"

    def test_with_none_overrides(self):
        """Test that None env_overrides are handled gracefully."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides=None
        )

        assert env["AGENT_NAME"] == "test_agent"
        assert "AGENT_BASE_URL" in env

    def test_trusted_cidrs_set(self):
        """Test that TRUSTED_CIDRS is properly set."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert "TRUSTED_CIDRS" in env
        assert "127.0.0.0/8" in env["TRUSTED_CIDRS"]
        assert "::1/128" in env["TRUSTED_CIDRS"]
        assert "172.16.0.0/12" in env["TRUSTED_CIDRS"]

    def test_return_type_is_dict(self):
        """Test that return value is a dictionary."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert isinstance(env, dict)

    def test_all_values_are_strings(self):
        """Test that all values are strings."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides={"INT_VAL": 123, "FLOAT_VAL": 45.67}
        )

        for key, value in env.items():
            assert isinstance(value, str), f"{key} is not a string: {type(value)}"

    def test_observability_defaults(self):
        """Test that observability defaults are set."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert env["PROMETHEUS_ENABLED"] == "true"
        assert env["OTEL_SERVICE_NAME"] == "test_agent"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in env

    def test_port_parameter_used(self):
        """Test that port parameter is properly used."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=9999,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert env["PORT"] == "9999"

    def test_agent_name_parameter_used(self):
        """Test that agent_name parameter is properly used."""
        env = get_common_agent_env(
            agent_name="my_special_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert env["AGENT_NAME"] == "my_special_agent"
        assert env["OTEL_SERVICE_NAME"] == "my_special_agent"

    def test_base_url_parameter_used(self):
        """Test that base_url parameter is properly used."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://example.com",
            local_registry_url="http://localhost:8081"
        )

        assert env["AGENT_BASE_URL"] == "http://example.com"

    def test_registry_url_parameter_used(self):
        """Test that local_registry_url parameter is properly used."""
        env = get_common_agent_env(
            agent_name="test_agent",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://custom-registry:9000"
        )

        assert env["AGENT_REGISTRY_URL"] == "http://custom-registry:9000"
