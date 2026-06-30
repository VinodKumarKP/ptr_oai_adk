"""Unit tests for oai_mcp_registry.services.db.database_logger."""

import json
import logging
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from oai_mcp_registry.services.db.database_logger import RegistryDatabaseLogger


def make_mock_backend(name="sqlite"):
    backend = AsyncMock()
    backend.name = name
    backend.initialize = AsyncMock(return_value=True)
    backend.execute = AsyncMock()
    backend.fetch_one = AsyncMock(return_value=None)
    backend.fetch = AsyncMock(return_value=[])
    backend.close = AsyncMock()
    backend.MCP_REGISTRY_UPSERT = "UPSERT SQL"
    backend.MCP_REGISTRY_DEACTIVATE = "DEACTIVATE SQL"
    backend.MCP_REGISTRY_DELETE = "DELETE SQL"
    backend.MCP_REGISTRY_SELECT_ONE = "SELECT ONE SQL"
    backend.MCP_REGISTRY_SELECT_ALL = "SELECT ALL SQL"
    backend.MCP_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT ACTIVE SQL"
    backend.SERVER_ACTION_INSERT = "ACTION INSERT SQL"
    backend.SERVER_ACTION_SELECT_ALL = "ACTION SELECT SQL"
    backend.SERVER_ACTION_SELECT_FILTERED = "ACTION FILTERED SQL"
    backend.SERVER_ACTION_COUNT = "ACTION COUNT SQL"
    return backend


class TestRegistryDatabaseLoggerInit:

    def test_init_defaults(self):
        logger = RegistryDatabaseLogger()
        assert logger.is_active is False
        assert logger._backend is None
        assert not logger._db_logging_enabled

    def test_init_with_custom_backends(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        assert len(logger._backends) == 1

    def test_init_with_logger(self):
        log = logging.getLogger("test")
        logger = RegistryDatabaseLogger(logger=log)
        assert logger.logger is log


class TestRegistryDatabaseLoggerInitialize:

    @pytest.mark.asyncio
    async def test_initialize_disabled_by_env(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])

        with patch.dict(os.environ, {"REGISTRY_DB_LOGGING_ENABLED": "false"}):
            await logger.initialize()

        assert logger.is_active is False
        backend.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_with_active_backend(self):
        backend = make_mock_backend()
        backend.initialize = AsyncMock(return_value=True)
        logger = RegistryDatabaseLogger(backends=[backend])

        with patch.dict(os.environ, {"REGISTRY_DB_LOGGING_ENABLED": "true"}):
            await logger.initialize()

        assert logger.is_active is True
        assert logger._backend is backend

    @pytest.mark.asyncio
    async def test_initialize_fallback_on_failure(self):
        bad_backend = make_mock_backend()
        bad_backend.initialize = AsyncMock(return_value=False)
        good_backend = make_mock_backend()
        good_backend.initialize = AsyncMock(return_value=True)

        logger = RegistryDatabaseLogger(backends=[bad_backend, good_backend])

        with patch.dict(os.environ, {"REGISTRY_DB_LOGGING_ENABLED": "true"}):
            await logger.initialize()

        assert logger._backend is good_backend
        assert logger.is_active is True

    @pytest.mark.asyncio
    async def test_initialize_all_backends_fail(self):
        backend = make_mock_backend()
        backend.initialize = AsyncMock(return_value=False)
        logger = RegistryDatabaseLogger(backends=[backend])

        with patch.dict(os.environ, {"REGISTRY_DB_LOGGING_ENABLED": "true"}):
            await logger.initialize()

        assert logger.is_active is False


class TestLogServerRegistration:

    @pytest.mark.asyncio
    async def test_log_registration_when_ready(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.log_server_registration(
            server_name="srv",
            endpoint_url="http://localhost:8000",
            port=8000,
            description="Test server",
            active=True,
            registered_via="dynamic",
            source="https://github.com/test",
            tags=["tag1"],
            current_version="1.0",
            available_versions=["1.0"],
            deployment_mode="docker",
            env_vars={"K": "V"},
            sensitive_vars=["K"],
        )

        backend.execute.assert_called_once()
        args = backend.execute.call_args
        assert args[0][0] == "UPSERT SQL"

    @pytest.mark.asyncio
    async def test_log_registration_skips_when_not_ready(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = False
        logger.is_active = False

        await logger.log_server_registration(
            server_name="srv",
            endpoint_url="http://localhost:8000",
            port=8000,
        )

        backend.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_registration_raises_on_db_error(self):
        backend = make_mock_backend()
        backend.execute = AsyncMock(side_effect=RuntimeError("db error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        with pytest.raises(RuntimeError):
            await logger.log_server_registration(
                server_name="srv",
                endpoint_url="http://localhost:8000",
                port=8000,
            )

    @pytest.mark.asyncio
    async def test_log_registration_null_env_vars(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.log_server_registration(
            server_name="srv",
            endpoint_url="http://localhost:8000",
            port=8000,
            env_vars=None,
            sensitive_vars=None,
            tags=None,
            available_versions=None,
        )

        backend.execute.assert_called_once()


class TestDeregisterServer:

    @pytest.mark.asyncio
    async def test_deregister_sqlite(self):
        backend = make_mock_backend(name="sqlite")
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.deregister_server("srv")

        backend.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_deregister_postgres(self):
        from oai_mcp_registry.services.db.database_logger import PostgresBackend
        backend = make_mock_backend(name="postgres")
        # Make isinstance check work
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        with patch("oai_mcp_registry.services.db.database_logger.isinstance", return_value=True):
            await logger.deregister_server("srv")

        backend.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_deregister_skips_when_not_ready(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])

        await logger.deregister_server("srv")

        backend.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_deregister_raises_on_error(self):
        backend = make_mock_backend()
        backend.execute = AsyncMock(side_effect=RuntimeError("db error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        with pytest.raises(RuntimeError):
            await logger.deregister_server("srv")


class TestDeleteServer:

    @pytest.mark.asyncio
    async def test_delete_server(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.delete_server("srv")

        backend.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_server_raises_on_error(self):
        backend = make_mock_backend()
        backend.execute = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        with pytest.raises(RuntimeError):
            await logger.delete_server("srv")


class TestGetServerDetails:

    @pytest.mark.asyncio
    async def test_get_server_details_found(self):
        backend = make_mock_backend()
        row = {
            "server_name": "srv",
            "endpoint_url": "http://localhost:8000",
            "port": 8000,
            "active": True,
            "tags": '["tag1"]',
            "available_versions": '["1.0"]',
            "env_vars": '{"K":"V"}',
            "sensitive_vars": '["K"]',
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        backend.fetch_one = AsyncMock(return_value=row)
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_details("srv")

        assert result["server_name"] == "srv"
        assert result["tags"] == ["tag1"]
        assert result["env_vars"] == {"K": "V"}

    @pytest.mark.asyncio
    async def test_get_server_details_not_found(self):
        backend = make_mock_backend()
        backend.fetch_one = AsyncMock(return_value=None)
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_details("ghost")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_server_details_returns_none_when_not_ready(self):
        logger = RegistryDatabaseLogger()
        result = await logger.get_server_details("srv")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_server_details_handles_error(self):
        backend = make_mock_backend()
        backend.fetch_one = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_details("srv")

        assert result is None


class TestGetAllServers:

    @pytest.mark.asyncio
    async def test_get_all_servers(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(return_value=[
            {"server_name": "srv1", "tags": "[]", "available_versions": "[]",
             "env_vars": "{}", "sensitive_vars": "[]", "active": True,
             "created_at": None, "updated_at": None},
        ])
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_all_servers()

        assert len(result) == 1
        assert result[0]["server_name"] == "srv1"

    @pytest.mark.asyncio
    async def test_get_all_servers_returns_empty_when_not_ready(self):
        logger = RegistryDatabaseLogger()
        result = await logger.get_all_servers()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_servers_handles_error(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_all_servers()

        assert result == []


class TestGetActiveDynamicServers:

    @pytest.mark.asyncio
    async def test_get_active_dynamic_servers(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(return_value=[
            {"server_name": "dyn_srv", "tags": "[]", "available_versions": "[]",
             "env_vars": "{}", "sensitive_vars": "[]", "active": True,
             "created_at": None, "updated_at": None},
        ])
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_active_dynamic_servers()

        assert len(result) == 1


class TestLogServerAction:

    @pytest.mark.asyncio
    async def test_log_action(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.log_server_action("srv", "start", "1.0")

        backend.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_action_skips_when_not_ready(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])

        await logger.log_server_action("srv", "start")

        backend.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_action_raises_on_error(self):
        backend = make_mock_backend()
        backend.execute = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        with pytest.raises(RuntimeError):
            await logger.log_server_action("srv", "start")


class TestGetServerActions:

    @pytest.mark.asyncio
    async def test_get_server_actions_unfiltered(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(return_value=[
            {"id": 1, "server_name": "srv", "action": "start", "version": "1.0",
             "created_at": datetime.now(timezone.utc)},
        ])
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_actions("srv")

        assert len(result) == 1
        assert result[0]["action"] == "start"
        assert isinstance(result[0]["created_at"], str)

    @pytest.mark.asyncio
    async def test_get_server_actions_filtered(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(return_value=[])
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_actions("srv", action_type="start")

        backend.fetch.assert_called_once_with("ACTION FILTERED SQL", ("srv", "start"))

    @pytest.mark.asyncio
    async def test_get_server_actions_with_limit(self):
        backend = make_mock_backend()
        rows = [{"id": i, "server_name": "srv", "action": "start", "version": None,
                 "created_at": None} for i in range(10)]
        backend.fetch = AsyncMock(return_value=rows)
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_actions("srv", limit=5)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_get_server_actions_returns_empty_when_not_ready(self):
        logger = RegistryDatabaseLogger()
        result = await logger.get_server_actions("srv")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_server_actions_handles_error(self):
        backend = make_mock_backend()
        backend.fetch = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        result = await logger.get_server_actions("srv")

        assert result == []


class TestGetServerActionCount:

    @pytest.mark.asyncio
    async def test_get_action_count(self):
        backend = make_mock_backend()
        backend.fetch_one = AsyncMock(return_value={"count": 42})
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        count = await logger.get_server_action_count("srv")

        assert count == 42

    @pytest.mark.asyncio
    async def test_get_action_count_returns_zero_when_not_ready(self):
        logger = RegistryDatabaseLogger()
        count = await logger.get_server_action_count("srv")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_action_count_handles_none_row(self):
        backend = make_mock_backend()
        backend.fetch_one = AsyncMock(return_value=None)
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        count = await logger.get_server_action_count("srv")

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_action_count_handles_error(self):
        backend = make_mock_backend()
        backend.fetch_one = AsyncMock(side_effect=RuntimeError("error"))
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        count = await logger.get_server_action_count("srv")

        assert count == 0


class TestClose:

    @pytest.mark.asyncio
    async def test_close_resets_state(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        await logger.close()

        backend.close.assert_called_once()
        assert logger._backend is None
        assert logger.is_active is False

    @pytest.mark.asyncio
    async def test_close_when_no_backend(self):
        logger = RegistryDatabaseLogger()
        await logger.close()  # should not raise


class TestDeserializeRow:

    def test_deserialize_timestamps(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        now = datetime.now(timezone.utc)
        row = {"created_at": now, "updated_at": now, "server_name": "srv"}
        result = logger._deserialize_row(row)

        assert isinstance(result["created_at"], str)
        assert isinstance(result["updated_at"], str)

    def test_deserialize_sqlite_active_bool(self):
        backend = make_mock_backend(name="sqlite")
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        row = {"active": 1, "tags": "[]", "available_versions": "[]"}
        result = logger._deserialize_row(row)

        assert result["active"] is True

    def test_deserialize_json_fields(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        row = {
            "tags": '["a","b"]',
            "available_versions": '["1.0","2.0"]',
            "env_vars": '{"KEY":"val"}',
            "sensitive_vars": '["KEY"]',
        }
        result = logger._deserialize_row(row)

        assert result["tags"] == ["a", "b"]
        assert result["available_versions"] == ["1.0", "2.0"]
        assert result["env_vars"] == {"KEY": "val"}
        assert result["sensitive_vars"] == ["KEY"]

    def test_deserialize_invalid_json_uses_empty(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._db_logging_enabled = True
        logger.is_active = True
        logger._backend = backend

        row = {
            "tags": "not json",
            "available_versions": "also not json",
            "env_vars": "bad",
            "sensitive_vars": "bad2",
        }
        result = logger._deserialize_row(row)

        assert result["tags"] == []
        assert result["available_versions"] == []
        assert result["env_vars"] == {}
        assert result["sensitive_vars"] == []

    def test_deserialize_empty_row(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._backend = backend

        result = logger._deserialize_row({})
        assert result == {}

    def test_deserialize_null_env_vars(self):
        backend = make_mock_backend()
        logger = RegistryDatabaseLogger(backends=[backend])
        logger._backend = backend

        row = {"env_vars": None, "sensitive_vars": None}
        result = logger._deserialize_row(row)

        assert result["env_vars"] == {}
        assert result["sensitive_vars"] == []

    def test_isoformat_with_datetime(self):
        now = datetime.now(timezone.utc)
        result = RegistryDatabaseLogger._isoformat(now)
        assert isinstance(result, str)

    def test_isoformat_with_string(self):
        result = RegistryDatabaseLogger._isoformat("2024-01-01")
        assert result == "2024-01-01"

    def test_isoformat_with_none(self):
        result = RegistryDatabaseLogger._isoformat(None)
        assert result is None
