"""
Tests for oai_platform_core.deployers.env_utils.build_deployment_env
"""
import os
import pytest
from unittest.mock import patch
from oai_platform_core.deployers.env_utils import build_deployment_env


class TestBuildDeploymentEnv:
    def test_required_keys_present(self):
        env = build_deployment_env(
            port=9000,
            redis_host="redis-host",
            logging_db_host="db-host",
            logging_db_name="mydb",
        )
        required = {
            "PORT", "AWS_REGION", "REDIS_HOST", "REDIS_PORT",
            "LOGGING_DB_HOST", "LOGGING_DB_PORT", "LOGGING_DB_USER",
            "LOGGING_DB_PASSWORD", "LOGGING_DB_NAME",
            "DB_LOGGING_ENABLED", "DB_POOL_MAX_SIZE",
            "DB_POOL_TIMEOUT", "DB_POOL_MIN_SIZE",
        }
        assert required.issubset(env.keys())

    def test_port_is_stringified(self):
        env = build_deployment_env(9001, "rh", "dbh", "db")
        assert env["PORT"] == "9001"

    def test_redis_host_used(self):
        env = build_deployment_env(9000, "my-redis", "dbh", "db")
        assert env["REDIS_HOST"] == "my-redis"

    def test_logging_db_values_set(self):
        env = build_deployment_env(9000, "rh", "pg-host", "logs_db")
        assert env["LOGGING_DB_HOST"] == "pg-host"
        assert env["LOGGING_DB_NAME"] == "logs_db"

    def test_redis_port_fixed_at_6379(self):
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["REDIS_PORT"] == "6379"

    def test_logging_db_port_fixed_at_5432(self):
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["LOGGING_DB_PORT"] == "5432"

    def test_env_overrides_applied(self):
        env = build_deployment_env(
            9000, "rh", "dbh", "db",
            env_overrides={"CUSTOM_KEY": "custom_value", "PORT": "1234"},
        )
        assert env["CUSTOM_KEY"] == "custom_value"
        # Overrides win over positional args
        assert env["PORT"] == "1234"

    def test_env_overrides_values_coerced_to_str(self):
        env = build_deployment_env(9000, "rh", "dbh", "db",
                                   env_overrides={"MY_INT": 42})
        assert env["MY_INT"] == "42"

    def test_no_overrides_when_none(self):
        env = build_deployment_env(9000, "rh", "dbh", "db", env_overrides=None)
        assert "CUSTOM_KEY" not in env

    def test_aws_region_from_env(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["AWS_REGION"] == "eu-west-1"

    def test_aws_region_default_when_not_set(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["AWS_REGION"] == "us-east-1"

    def test_db_logging_enabled_true(self):
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["DB_LOGGING_ENABLED"] == "true"

    def test_pool_defaults(self):
        env = build_deployment_env(9000, "rh", "dbh", "db")
        assert env["DB_POOL_MAX_SIZE"] == "2"
        assert env["DB_POOL_MIN_SIZE"] == "1"
        assert env["DB_POOL_TIMEOUT"] == "60"

    def test_returns_dict_of_strings(self):
        env = build_deployment_env(9000, "rh", "dbh", "db")
        for v in env.values():
            assert isinstance(v, str)
