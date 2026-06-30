"""Unit tests for oai_mcp_registry.utils.env_vars."""

import pytest

from oai_mcp_registry.utils.env_vars import get_common_server_env


class TestGetCommonServerEnv:
    def test_docker_deployment_mode(self):
        """Test env vars for docker deployment."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://host.docker.internal:8081",
            deployment_mode="docker"
        )

        assert env["MCP_SERVER_NAME"] == "test_server"
        assert env["PORT"] == "8000"
        assert env["MCP_AUTH_ENABLED"] == "true"
        assert env["AUTH_ENABLED"] == "true"
        assert env["MCP_BASE_URL"] == "http://localhost"
        assert env["MCP_REGISTRY_URL"] == "http://host.docker.internal:8081"
        assert env["FORCE_AUTH"] == "true"
        assert "mcp-valkey" in env.get("REDIS_HOST", "")
        assert "mcp_logs_db" in env.get("LOGGING_DB_HOST", "")

    def test_local_deployment_mode(self):
        """Test env vars for local deployment."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            deployment_mode="local"
        )

        assert env["MCP_SERVER_NAME"] == "test_server"
        assert "localhost" in env.get("REDIS_HOST", "")
        assert "localhost" in env.get("LOGGING_DB_HOST", "")

    def test_with_env_overrides(self):
        """Test that env_overrides are applied."""
        overrides = {
            "CUSTOM_VAR": "custom_value",
            "MCP_SERVER_NAME": "overridden_name",
            "MCP_AUTH_ENABLED": "false"
        }
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides=overrides
        )

        assert env["CUSTOM_VAR"] == "custom_value"
        assert env["MCP_SERVER_NAME"] == "overridden_name"
        assert env["MCP_AUTH_ENABLED"] == "false"

    def test_with_none_overrides(self):
        """Test that None env_overrides are handled gracefully."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides=None
        )

        assert env["MCP_SERVER_NAME"] == "test_server"
        assert "MCP_BASE_URL" in env

    def test_return_type_is_dict(self):
        """Test that return value is a dictionary."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert isinstance(env, dict)

    def test_all_values_are_strings(self):
        """Test that all values are strings."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081",
            env_overrides={"INT_VAL": 123, "FLOAT_VAL": 45.67}
        )

        for key, value in env.items():
            assert isinstance(value, str), f"{key} is not a string: {type(value)}"

    def test_server_name_parameter_used(self):
        """Test that server_name parameter is properly used."""
        env = get_common_server_env(
            server_name="my_special_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert env["MCP_SERVER_NAME"] == "my_special_server"

    def test_port_parameter_used(self):
        """Test that port parameter is properly used."""
        env = get_common_server_env(
            server_name="test_server",
            port=9999,
            base_url="http://localhost",
            local_registry_url="http://localhost:8081"
        )

        assert env["PORT"] == "9999"

    def test_base_url_parameter_used(self):
        """Test that base_url parameter is properly used."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://example.com",
            local_registry_url="http://localhost:8081"
        )

        assert env["MCP_BASE_URL"] == "http://example.com"

    def test_registry_url_parameter_used(self):
        """Test that local_registry_url parameter is properly used."""
        env = get_common_server_env(
            server_name="test_server",
            port=8000,
            base_url="http://localhost",
            local_registry_url="http://custom-registry:9000"
        )

        assert env["MCP_REGISTRY_URL"] == "http://custom-registry:9000"
