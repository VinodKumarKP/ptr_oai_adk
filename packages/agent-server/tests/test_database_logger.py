"""
Test suite for database_logger.py (refactored version).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import sys
import types
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Locate the module under test
_MODULE_PATH = './src/oai_agent_server/utils/database_logger.py'

def run(coro):
    return asyncio.run(coro)

def silent_logger() -> logging.Logger:
    log = logging.getLogger("test_silent")
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log

class _FakeDBConn:
    def __init__(self, captured: list):
        self._captured = captured
        self.row_factory = None
    async def execute(self, sql, params=()):
        self._captured.append({"sql": sql, "params": params})
    async def executemany(self, sql, params_seq):
        for p in params_seq:
            self._captured.append({"sql": sql, "params": p, "via": "executemany"})
    async def commit(self): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass

class _FakeCursorCM:
    def __init__(self, calls, sql, params):
        self._calls = calls
        self.sql = sql
        self.params = params
    def __await__(self):
        self._calls.append({"sql": self.sql, "params": self.params})
        async def _f(): return self
        return _f().__await__()
    async def __aenter__(self):
        self._calls.append({"sql": self.sql, "params": self.params})
        return self
    async def __aexit__(self, *args): pass
    async def fetchall(self): return []
    async def fetchone(self): return None

class _FakeAiosqlite(types.ModuleType):
    class Row(dict): pass
    def __init__(self):
        super().__init__("aiosqlite")
        self.calls: List[Dict] = []
        self.Row = _FakeAiosqlite.Row
    def connect(self, path: str):
        outer = self
        @asynccontextmanager
        async def _ctx():
            conn = _FakeDBConn(outer.calls)
            conn.row_factory = _FakeAiosqlite.Row
            def _execute(sql, params=()):
                return _FakeCursorCM(outer.calls, sql, params)
            conn.execute = _execute
            yield conn
        return _ctx()

class _FakeAsyncpg(types.ModuleType):
    def __init__(self):
        super().__init__("asyncpg")
        self.exceptions = types.SimpleNamespace(
            InvalidPasswordError=OSError, CannotConnectNowError=OSError,
            TooManyConnectionsError=OSError, ConnectionDoesNotExistError=OSError,
        )
        self.PostgresError = OSError
    async def create_pool(self, *args, **kwargs):
        raise OSError("no real postgres in tests")

def load_module(fake_aiosqlite: Optional[_FakeAiosqlite] = None, fake_asyncpg: Optional[_FakeAsyncpg] = None):
    stubs: dict = {}
    if fake_aiosqlite is not None: stubs["aiosqlite"] = fake_aiosqlite
    if fake_asyncpg is not None:
        stubs["asyncpg"] = fake_asyncpg
        stubs["asyncpg.pool"] = types.SimpleNamespace(Pool=None)
    spec = importlib.util.spec_from_file_location("database_logger_under_test", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(mod)
    return mod

def make_mock_backend(succeeds: bool = True, name: str = "mock", fetch_result: Optional[List] = None, fetch_one_result: Optional[Dict] = None) -> MagicMock:
    b = MagicMock()
    b.name = name
    b.initialize = AsyncMock(return_value=succeeds)
    b.execute = AsyncMock()
    b.execute_many = AsyncMock()
    b.fetch = AsyncMock(return_value=fetch_result if fetch_result is not None else [])
    b.fetch_one = AsyncMock(return_value=fetch_one_result)
    b.close = AsyncMock()
    b.CHAT_LOGS_INSERT = "INSERT_CHAT"
    b.ACTIVITY_LOG_INSERT = "INSERT_ACTIVITY"
    b.PLACEHOLDER = "$"
    return b

@pytest.fixture()
def mod():
    return load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())

@pytest.fixture()
def backend():
    return make_mock_backend()

@pytest.fixture()
def active_db(mod, backend):
    with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
        db = mod.DatabaseLogger(backends=[backend], logger=silent_logger())
        run(db.initialize())
    return db

def chat_log_row(**overrides) -> Dict:
    row = {"id": 1, "timestamp": "2024-01-01T00:00:00+00:00", "agent_name": "agent", "session_id": "s1", "user_id": "u1", "endpoint": "/chat", "input_message": '{"role": "user"}', "output_response": '{"role": "assistant"}', "request_headers": '{"Content-Type": "application/json"}', "model_info": None, "token_usage": None, "total_tokens": 42, "response_time_ms": 123.4, "status": "success", "error_message": None, "created_at": "2024-01-01T00:00:00+00:00"}
    row.update(overrides)
    return row

def activity_row(**overrides) -> Dict:
    row = {"id": 1, "timestamp": "2024-01-01T00:00:00+00:00", "agent_name": "agent", "session_id": "s1", "user_id": "u1", "endpoint": "/chat/stream", "chunk_sequence": 0, "chunk_content": '{"text": "hello"}', "chunk_text": "hello", "serialization_warning": None, "request_headers": '{"Content-Type": "application/json"}', "created_at": "2024-01-01T00:00:00+00:00"}
    row.update(overrides)
    return row

def stats_row(**overrides) -> Dict:
    row = {"total_interactions": 10, "unique_sessions": 5, "unique_users": 3, "avg_response_time": 200.0, "max_response_time": 500.0, "min_response_time": 50.0, "total_tokens_sum": 1000, "successful_interactions": 9, "failed_interactions": 1}
    row.update(overrides)
    return row

def make_read_db(fetch_result=None, fetch_one_result=None):
    mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
    b = make_mock_backend(fetch_result=fetch_result, fetch_one_result=fetch_one_result)
    with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
        db = mod.DatabaseLogger(backends=[b], logger=silent_logger())
        run(db.initialize())
    return mod, db, b

def make_chunks(n: int = 3) -> List[Dict]:
    return [{"chunk_sequence": i, "chunk_content": {"type": "text_delta", "text": f"word{i}"}, "chunk_text": f"word{i}", "serialization_warning": None} for i in range(n)]

class TestSerializeForJson:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.serialize = load_module().DatabaseLogger._serialize_for_json
    def test_none(self): assert self.serialize(None) is None
    def test_primitives(self):
        for v in (42, 3.14, True, "hello"): assert self.serialize(v) == v
    def test_list(self): assert self.serialize([1, "a", None]) == [1, "a", None]
    def test_tuple_becomes_list(self): assert self.serialize((1, 2)) == [1, 2]
    def test_dict(self): assert self.serialize({"k": 1}) == {"k": 1}
    def test_dict_keys_coerced_to_str(self): assert "1" in self.serialize({1: "v"})
    def test_nested(self): assert self.serialize({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}
    def test_max_depth(self):
        deep: dict = {}
        cursor = deep
        for _ in range(15): cursor["x"] = {}; cursor = cursor["x"]
        assert self.serialize(deep, max_depth=10) is not None
    def test_plain_object_via_dict(self):
        class Obj:
            def __init__(self): self.a = 1; self.b = "two"
        assert self.serialize(Obj()) == {"a": 1, "b": "two"}
    def test_pydantic_v2_model_dump_preferred_over_dict(self):
        call_order: List[str] = []
        class FakePydanticV2:
            def model_dump(self): call_order.append("model_dump"); return {"from": "model_dump"}
            def dict(self): call_order.append("dict"); return {"from": "dict"}
            @property
            def __dict__(self): call_order.append("__dict__"); return {"from": "__dict__"}
        result = self.serialize(FakePydanticV2())
        assert result == {"from": "model_dump"}
        assert call_order[0] == "model_dump"
    def test_pydantic_v1_dict_fallback(self):
        class FakePydanticV1:
            def dict(self): return {"source": "v1"}
        assert self.serialize(FakePydanticV1()) == {"source": "v1"}
    def test_unserializable_falls_back_to_str(self):
        class Weird:
            def __str__(self): return "weird"
        assert self.serialize(Weird()) == "weird"

class TestToJson:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.to_json = load_module().DatabaseLogger._to_json
    def test_none_returns_none(self): assert self.to_json(None) is None
    def test_dict_returns_json_string(self):
        result = self.to_json({"a": 1})
        assert isinstance(result, str)
        assert json.loads(result) == {"a": 1}
    def test_list(self): assert json.loads(self.to_json([1, 2, 3])) == [1, 2, 3]

class TestRedactHeaders:
    @pytest.fixture(autouse=True)
    def setup(self):
        mod = load_module()
        self.redact = mod.DatabaseLogger._redact_headers
        self.sentinel = mod._REDACTED_SENTINEL
    def test_none_passthrough(self): assert self.redact(None) is None
    def test_empty_dict_passthrough(self): assert self.redact({}) == {}
    def test_authorization_redacted(self): assert self.redact({"Authorization": "Bearer secret"})["Authorization"] == self.sentinel
    def test_cookie_redacted(self): assert self.redact({"Cookie": "session=abc"})["Cookie"] == self.sentinel
    def test_x_api_key_redacted(self): assert self.redact({"X-Api-Key": "key123"})["X-Api-Key"] == self.sentinel
    def test_case_insensitive_redaction(self):
        result = self.redact({"AUTHORIZATION": "s1", "authorization": "s2"})
        assert result["AUTHORIZATION"] == self.sentinel
        assert result["authorization"] == self.sentinel
    def test_safe_headers_preserved(self):
        result = self.redact({"Content-Type": "application/json", "X-Request-Id": "123"})
        assert result["Content-Type"] == "application/json"
        assert result["X-Request-Id"] == "123"
    def test_mixed_headers(self):
        result = self.redact({"Authorization": "tok", "Content-Type": "application/json", "X-Api-Key": "k"})
        assert result["Authorization"] == self.sentinel
        assert result["Content-Type"] == "application/json"
        assert result["X-Api-Key"] == self.sentinel
    def test_original_dict_not_mutated(self):
        original = {"Authorization": "secret"}
        self.redact(original)
        assert original["Authorization"] == "secret"

class TestExtractTotalTokens:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.extract = load_module().DatabaseLogger._extract_total_tokens
    def test_none_returns_none(self): assert self.extract(None) is None
    def test_snake_case_key(self): assert self.extract({"total_tokens": 100}) == 100
    def test_camel_case_key(self): assert self.extract({"totalTokens": 200}) == 200
    def test_snake_case_preferred(self): assert self.extract({"total_tokens": 10, "totalTokens": 20}) == 10
    def test_non_dict_returns_none(self): assert self.extract("not a dict") is None

class TestLoggingDisabled:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
    def _make_db(self):
        return self.mod.DatabaseLogger(backends=[], logger=silent_logger())
    def test_is_active_false_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            db = self._make_db()
            run(db.initialize())
        assert not db.is_active
    def test_log_interaction_silent_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            db = self._make_db()
            run(db.initialize())
            run(db.log_interaction(interaction_id="1", agent_name="a", session_id="s", user_id="u", endpoint="/ep", input_message={}, output_response={}))
    def test_log_stream_chunk_silent_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            db = self._make_db()
            run(db.initialize())
            run(db.log_stream_chunks_batch(interaction_id="1", agent_name="agent", session_id="sid", user_id="uid", endpoint="/ep", chunks=[{"chunk_sequence": 0, "chunk_content": {}}]))
    def test_close_safe_when_no_backend(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            db = self._make_db()
            run(db.initialize())
            run(db.close())
        assert not db.is_active

class TestFallbackChain:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
    def test_first_succeeding_backend_used(self):
        b1 = make_mock_backend(False, "b1")
        b2 = make_mock_backend(True, "b2")
        b3 = make_mock_backend(True, "b3")
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
            db = self.mod.DatabaseLogger(backends=[b1, b2, b3], logger=silent_logger())
            run(db.initialize())
        assert db.is_active
        assert db._backend is b2
        b3.initialize.assert_not_called()
    def test_all_fail_leaves_inactive(self):
        b1, b2 = make_mock_backend(False, "b1"), make_mock_backend(False, "b2")
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
            db = self.mod.DatabaseLogger(backends=[b1, b2], logger=silent_logger())
            run(db.initialize())
        assert not db.is_active
        assert db._backend is None
    def test_no_backends_leaves_inactive_use_default_sqllite(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
            db = self.mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
        assert db.is_active

class TestLogInteraction:
    @pytest.fixture(autouse=True)
    def setup(self, mod, backend, active_db):
        self.mod = mod
        self.backend = backend
        self.db = active_db
    def _call(self, **kwargs):
        defaults = dict(
            interaction_id="test-id", agent_name="agent", session_id="sess-1", user_id="user-1", endpoint="/chat",
            input_message={"role": "user", "content": "hello"},
            output_response={"role": "assistant", "content": "hi"},
        )
        defaults.update(kwargs)
        run(self.db.log_interaction(**defaults))
    def test_execute_called_once(self): self._call(); self.backend.execute.assert_called_once()
    def test_query_is_chat_insert(self):
        self._call()
        query, _ = self.backend.execute.call_args[0]
        assert query == "INSERT_CHAT"
    def test_param_count(self):
        self._call()
        _, params = self.backend.execute.call_args[0]
        assert len(params) == 15
    def test_agent_name_in_params(self):
        self._call(agent_name="my-agent")
        _, params = self.backend.execute.call_args[0]
        assert params[2] == "my-agent"
    def test_session_id_in_params(self):
        self._call(session_id="s-xyz")
        _, params = self.backend.execute.call_args[0]
        assert params[3] == "s-xyz"
    def test_input_message_serialised_to_json(self):
        self._call(input_message={"role": "user", "content": "hi"})
        _, params = self.backend.execute.call_args[0]
        assert json.loads(params[6])["role"] == "user"
    def test_token_usage_snake_case_extracted(self):
        self._call(token_usage={"total_tokens": 77})
        _, params = self.backend.execute.call_args[0]
        assert params[11] == 77
    def test_token_usage_camel_case_extracted(self):
        self._call(token_usage={"totalTokens": 88})
        _, params = self.backend.execute.call_args[0]
        assert params[11] == 88
    def test_status_default_success(self):
        self._call()
        _, params = self.backend.execute.call_args[0]
        assert params[13] == "success"
    def test_custom_status(self):
        self._call(status="error", error_message="oops")
        _, params = self.backend.execute.call_args[0]
        assert params[13] == "error"
        assert params[14] == "oops"
    def test_authorization_header_redacted(self):
        self._call(request_headers={"Authorization": "Bearer tok", "Content-Type": "application/json"})
        _, params = self.backend.execute.call_args[0]
        headers = json.loads(params[8])
        assert headers["Authorization"] == self.mod._REDACTED_SENTINEL
        assert headers["Content-Type"] == "application/json"
    def test_timestamp_is_timezone_aware(self):
        self._call()
        _, params = self.backend.execute.call_args[0]
        assert params[1].tzinfo == timezone.utc
    def test_backend_exception_does_not_propagate(self):
        self.backend.execute = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError):
            self._call()
    def test_none_request_headers_stores_none(self):
        self._call(request_headers=None)
        _, params = self.backend.execute.call_args[0]
        assert params[8] is None

class TestLogStreamChunk:
    @pytest.fixture(autouse=True)
    def setup(self, mod, backend, active_db):
        self.mod = mod
        self.backend = backend
        self.db = active_db
    def _call(self, **kwargs):
        defaults = dict(
            interaction_id="1", agent_name="agent", session_id="sess-1", user_id="user-1",
            endpoint="/chat", chunks=[{"chunk_sequence": 3, "chunk_content": {"type": "text_delta", "text": "Hello"}}]
        )
        defaults.update(kwargs)
        run(self.db.log_stream_chunks_batch(**defaults))
    def test_execute_called_once(self): self._call(); self.backend.execute_many.assert_called_once()
    def test_query_is_activity_insert(self):
        self._call()
        query, _ = self.backend.execute_many.call_args[0]
        assert query == "INSERT_ACTIVITY"
    def test_param_count(self):
        self._call()
        _, params = self.backend.execute_many.call_args[0]
        assert len(params[0]) == 11
    def test_chunk_sequence_in_params(self):
        self._call(chunks=[{"chunk_sequence": 7, "chunk_content": {}}])
        _, params = self.backend.execute_many.call_args[0]
        assert params[0][6] == 7
    def test_chunk_content_serialised(self):
        self._call(chunks=[{"chunk_sequence": 0, "chunk_content": {"text": "hi"}}])
        _, params = self.backend.execute_many.call_args[0]
        assert json.loads(params[0][7]) == {"text": "hi"}
    def test_serialization_warning_passed_through(self):
        self._call(chunks=[{"chunk_sequence": 0, "chunk_content": {}, "serialization_warning": "truncated"}])
        _, params = self.backend.execute_many.call_args[0]
        assert params[0][9] == "truncated"
    def test_authorization_header_redacted(self):
        self._call(request_headers={"Authorization": "Bearer x"})
        _, params = self.backend.execute_many.call_args[0]
        assert json.loads(params[0][10])["Authorization"] == self.mod._REDACTED_SENTINEL
    def test_backend_exception_does_not_propagate(self):
        self.backend.execute_many = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError):
            self._call()
    def test_timestamp_timezone_aware(self):
        self._call()
        _, params = self.backend.execute_many.call_args[0]
        assert params[0][1].tzinfo is not None

class TestClose:
    @pytest.fixture(autouse=True)
    def setup(self, backend, active_db):
        self.backend = backend
        self.db = active_db
    def test_close_calls_backend_close(self): run(self.db.close()); self.backend.close.assert_called_once()
    def test_is_active_false_after_close(self): run(self.db.close()); assert not self.db.is_active
    def test_backend_nulled_after_close(self): run(self.db.close()); assert self.db._backend is None
    def test_double_close_safe(self): run(self.db.close()); run(self.db.close())

class TestSQLiteBackendIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fake_aiosqlite = _FakeAiosqlite()
        self.mod = load_module(fake_aiosqlite=self.fake_aiosqlite, fake_asyncpg=_FakeAsyncpg())
    def test_initialize_succeeds(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/test.db"}):
            assert run(backend.initialize(silent_logger())) is True
    def test_schema_created_on_initialize(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/test_schema.db"}):
            run(backend.initialize(silent_logger()))
        sqls = [c["sql"] for c in self.fake_aiosqlite.calls]
        assert any("chat_logs" in s for s in sqls)
        assert any("agent_activity_log" in s for s in sqls)
    def test_execute_records_call(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/test_exec.db"}):
            run(backend.initialize(silent_logger()))
            run(backend.execute("INSERT INTO foo VALUES (?)", ("bar",)))
        sqls = [c["sql"] for c in self.fake_aiosqlite.calls]
        assert "INSERT INTO foo VALUES (?)" in sqls
    def test_db_path_from_env(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/custom.db"}):
            run(backend.initialize(silent_logger()))
        assert backend._db_path == "/tmp/custom.db"
    def test_db_dir_env_sets_path(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_DIR": "/tmp/mydir"}, clear=False):
            os.environ.pop("SQLITE_DB_PATH", None)
            run(backend.initialize(silent_logger()))
        assert backend._db_path == "/tmp/mydir/agent_logs.db"
    def test_db_path_takes_precedence_over_db_dir(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/full.db", "SQLITE_DB_DIR": "/ignored"}):
            run(backend.initialize(silent_logger()))
        assert backend._db_path == "/tmp/full.db"
    def test_close_nulls_path(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/close_test.db"}):
            run(backend.initialize(silent_logger()))
        run(backend.close())
        assert backend._db_path is None

class TestPostgresBackendFallsBack:
    def test_unavailable_when_asyncpg_missing(self):
        with patch.dict(sys.modules, {"asyncpg": None}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite())
            assert run(mod.PostgresBackend().initialize(silent_logger())) is False
    def test_connection_failure_returns_false(self):
        mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
        assert run(mod.PostgresBackend().initialize(silent_logger())) is False

class TestSQLiteBackendFallsBack:
    def test_unavailable_when_aiosqlite_missing(self):
        with patch.dict(sys.modules, {"aiosqlite": None}):
            mod = load_module(fake_asyncpg=_FakeAsyncpg())
            backend = mod.SQLiteBackend()
            with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/x.db"}):
                assert run(backend.initialize(silent_logger())) is False

class TestCustomBackend:
    def test_custom_backend_used(self):
        mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
        executed: list = []
        class MemoryBackend(mod.DatabaseBackend):
            name = "memory"; CHAT_LOGS_INSERT = "MEMORY_CHAT"; ACTIVITY_LOG_INSERT = "MEMORY_ACTIVITY"; PLACEHOLDER = "$"
            async def initialize(self, app_logger): return True
            async def execute(self, query, params): executed.append((query, params))
            async def execute_many(self, query, params_seq):
                for p in params_seq: executed.append((query, p))
            async def fetch(self, query, params): return []
            async def fetch_one(self, query, params): return None
            async def close(self): pass
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "true"}):
            db = mod.DatabaseLogger(backends=[MemoryBackend()], logger=silent_logger())
            run(db.initialize())
            run(db.log_interaction(interaction_id="1", agent_name="a", session_id="s", user_id="u", endpoint="/ep", input_message="in", output_response="out"))
        assert len(executed) == 1
        assert executed[0][0] == "MEMORY_CHAT"

class TestLogStreamChunksBatch:
    @pytest.fixture(autouse=True)
    def setup(self, mod, backend, active_db):
        self.mod = mod
        self.backend = backend
        self.db = active_db
    def _call(self, chunks=None, **kwargs):
        defaults = dict(interaction_id="1", agent_name="agent", session_id="sess-1", user_id="user-1", endpoint="/chat", chunks=chunks if chunks is not None else make_chunks())
        defaults.update(kwargs)
        run(self.db.log_stream_chunks_batch(**defaults))
    def test_execute_many_called_once(self): self._call(); self.backend.execute_many.assert_called_once()
    def test_single_execute_not_called(self): self._call(); self.backend.execute.assert_not_called()
    def test_query_is_activity_insert(self):
        self._call()
        query, _ = self.backend.execute_many.call_args[0]
        assert query == "INSERT_ACTIVITY"
    def test_params_seq_length_matches_chunks(self):
        self._call(chunks=make_chunks(5))
        _, params_seq = self.backend.execute_many.call_args[0]
        assert len(params_seq) == 5
    def test_each_row_has_correct_param_count(self):
        self._call(chunks=make_chunks(3))
        _, params_seq = self.backend.execute_many.call_args[0]
        assert all(len(row) == 11 for row in params_seq)
    def test_chunk_sequence_values_preserved(self):
        self._call(chunks=make_chunks(4))
        _, params_seq = self.backend.execute_many.call_args[0]
        assert [row[6] for row in params_seq] == [0, 1, 2, 3]
    def test_chunk_content_serialised_per_row(self):
        self._call(chunks=make_chunks(2))
        _, params_seq = self.backend.execute_many.call_args[0]
        for i, row in enumerate(params_seq):
            assert json.loads(row[7])["text"] == f"word{i}"
    def test_chunk_text_preserved(self):
        self._call(chunks=make_chunks(2))
        _, params_seq = self.backend.execute_many.call_args[0]
        for i, row in enumerate(params_seq):
            assert row[8] == f"word{i}"
    def test_serialization_warning_passed_through(self):
        chunks = [{"chunk_sequence": 0, "chunk_content": {}, "serialization_warning": "warn!"}]
        self._call(chunks=chunks)
        _, params_seq = self.backend.execute_many.call_args[0]
        assert params_seq[0][9] == "warn!"
    def test_optional_fields_default_to_none(self):
        chunks = [{"chunk_sequence": 0, "chunk_content": {"text": "hi"}}]
        self._call(chunks=chunks)
        _, params_seq = self.backend.execute_many.call_args[0]
        assert params_seq[0][8] is None
        assert params_seq[0][9] is None
    def test_shared_headers_applied_to_every_row(self):
        self._call(chunks=make_chunks(3), request_headers={"X-Request-Id": "abc"})
        _, params_seq = self.backend.execute_many.call_args[0]
        for row in params_seq:
            assert json.loads(row[10])["X-Request-Id"] == "abc"
    def test_authorization_header_redacted_in_every_row(self):
        self._call(chunks=make_chunks(2), request_headers={"Authorization": "Bearer tok"})
        _, params_seq = self.backend.execute_many.call_args[0]
        for row in params_seq:
            assert json.loads(row[10])["Authorization"] == self.mod._REDACTED_SENTINEL
    def test_timestamps_are_timezone_aware(self):
        self._call(chunks=make_chunks(3))
        _, params_seq = self.backend.execute_many.call_args[0]
        assert all(row[1].tzinfo is not None for row in params_seq)
    def test_shared_fields_consistent_across_rows(self):
        self._call(chunks=make_chunks(3), agent_name="my-agent", session_id="s-99", user_id="u-1", endpoint="/stream")
        _, params_seq = self.backend.execute_many.call_args[0]
        for row in params_seq:
            assert row[2] == "my-agent"
            assert row[3] == "s-99"
            assert row[4] == "u-1"
            assert row[5] == "/stream"
    def test_empty_chunks_is_noop(self):
        self._call(chunks=[])
        self.backend.execute_many.assert_not_called()
        self.backend.execute.assert_not_called()
    def test_single_chunk_works(self):
        self._call(chunks=make_chunks(1))
        _, params_seq = self.backend.execute_many.call_args[0]
        assert len(params_seq) == 1
    def test_backend_exception_does_not_propagate(self):
        self.backend.execute_many = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError):
            self._call()
    def test_disabled_logging_is_noop(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
            db = mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
            run(db.log_stream_chunks_batch(interaction_id="1", agent_name="a", session_id="s", user_id="u", endpoint="/ep", chunks=make_chunks()))

class TestExecuteMany:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fake_aiosqlite = _FakeAiosqlite()
        self.mod = load_module(fake_aiosqlite=self.fake_aiosqlite, fake_asyncpg=_FakeAsyncpg())
    def test_execute_many_empty_is_noop(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/em_test.db"}):
            run(backend.initialize(silent_logger()))
            initial = len(self.fake_aiosqlite.calls)
            run(backend.execute_many("INSERT INTO x VALUES (?)", []))
        assert len(self.fake_aiosqlite.calls) == initial
    def test_execute_many_records_all_rows(self):
        backend = self.mod.SQLiteBackend()
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/em_test2.db"}):
            run(backend.initialize(silent_logger()))
            run(backend.execute_many("INSERT INTO x VALUES (?)", [("a",), ("b",), ("c",)]))
        batch = [c for c in self.fake_aiosqlite.calls if c.get("via") == "executemany"]
        assert len(batch) == 3
        assert sorted(c["params"][0] for c in batch) == ["a", "b", "c"]

class TestResolveDbPath:
    @pytest.fixture(autouse=True)
    def setup(self):
        mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
        self.resolve = mod.SQLiteBackend._resolve_db_path
    def _clean(self):
        env = os.environ.copy()
        env.pop("SQLITE_DB_PATH", None)
        env.pop("SQLITE_DB_DIR", None)
        return env
    def test_fallback_when_nothing_set(self):
        with patch.dict(os.environ, self._clean(), clear=True):
            assert self.resolve() == "agent_logs.db"
    def test_sqlite_db_path_used(self):
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/data/logs.db"}):
            assert self.resolve() == "/data/logs.db"
    def test_sqlite_db_dir_appends_default_filename(self):
        env = self._clean()
        env["SQLITE_DB_DIR"] = "/var/log/myapp"
        with patch.dict(os.environ, env, clear=True):
            assert self.resolve() == os.path.join("/var/log/myapp", "agent_logs.db")
    def test_sqlite_db_path_overrides_db_dir(self):
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/explicit/x.db", "SQLITE_DB_DIR": "/ignored"}):
            assert self.resolve() == "/explicit/x.db"
    def test_db_dir_with_trailing_slash(self):
        env = self._clean()
        env["SQLITE_DB_DIR"] = "/var/log/"
        with patch.dict(os.environ, env, clear=True):
            assert self.resolve() == os.path.join("/var/log/", "agent_logs.db")
    def test_relative_db_dir(self):
        env = self._clean()
        env["SQLITE_DB_DIR"] = "logs"
        with patch.dict(os.environ, env, clear=True):
            assert self.resolve() == os.path.join("logs", "agent_logs.db")
    def test_relative_db_path(self):
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "relative/custom.db"}):
            assert self.resolve() == "relative/custom.db"

class TestBuildWhereClause:
    @pytest.fixture(autouse=True)
    def setup(self, active_db):
        self.db = active_db
    def test_no_filters_returns_empty_string(self):
        where, params = self.db._build_where_clause([])
        assert where == "" and params == ()
    def test_all_none_values_returns_empty(self):
        where, params = self.db._build_where_clause([("agent_name", None), ("user_id", None)])
        assert where == "" and params == ()
    def test_single_filter_postgres(self):
        self.db._backend.PLACEHOLDER = "$"
        where, params = self.db._build_where_clause([("agent_name", "bot")])
        assert "cl.agent_name = $1" in where
        assert where.startswith("WHERE")
        assert params == ("bot",)
    def test_single_filter_sqlite(self):
        self.db._backend.PLACEHOLDER = "?"
        where, params = self.db._build_where_clause([("agent_name", "bot")])
        assert "cl.agent_name = ?" in where
        assert params == ("bot",)
    def test_multiple_filters_postgres(self):
        self.db._backend.PLACEHOLDER = "$"
        where, params = self.db._build_where_clause([("agent_name", "bot"), ("session_id", "s1"), ("user_id", "u1")])
        assert "$1" in where and "$2" in where and "$3" in where
        assert params == ("bot", "s1", "u1")
    def test_none_values_skipped_in_multiple(self):
        self.db._backend.PLACEHOLDER = "$"
        where, params = self.db._build_where_clause([("agent_name", "bot"), ("session_id", None), ("user_id", "u1")])
        assert "$1" in where and "$2" in where and "$3" not in where
        assert params == ("bot", "u1")
    def test_range_operator_preserved(self):
        ts = datetime.now(timezone.utc)
        self.db._backend.PLACEHOLDER = "$"
        where, params = self.db._build_where_clause([("timestamp >=", ts), ("timestamp <=", ts)])
        assert "cl.timestamp >= $1" in where and "cl.timestamp <= $2" in where
        assert params == (ts, ts)
    def test_conditions_joined_with_and(self):
        self.db._backend.PLACEHOLDER = "$"
        where, _ = self.db._build_where_clause([("a", 1), ("b", 2)])
        assert "AND" in where

class TestReadMethods:
    def test_get_logs_returns_empty_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
            db = mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
        assert run(db.get_logs()) == []
    def test_get_logs_calls_fetch(self):
        _, db, backend = make_read_db(fetch_result=[chat_log_row()])
        run(db.get_logs())
        backend.fetch.assert_called_once()
    def test_get_logs_query_targets_chat_logs(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_logs())
        query, _ = backend.fetch.call_args[0]
        assert "chat_logs" in query
    def test_get_logs_returns_list(self):
        _, db, _ = make_read_db(fetch_result=[chat_log_row()])
        result = run(db.get_logs())
        assert isinstance(result, list) and len(result) == 1
    def test_get_logs_deserialises_json_fields(self):
        _, db, _ = make_read_db(fetch_result=[chat_log_row()])
        result = run(db.get_logs())
        assert isinstance(result[0]["input_message"], dict)
        assert isinstance(result[0]["output_response"], dict)
    def test_get_logs_none_json_field_stays_none(self):
        _, db, _ = make_read_db(fetch_result=[chat_log_row(model_info=None, token_usage=None)])
        result = run(db.get_logs())
        assert result[0]["model_info"] is None
        assert result[0]["token_usage"] is None
    def test_get_logs_applies_filter_agent_name(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_logs(agent_name="my-agent"))
        query, params = backend.fetch.call_args[0]
        assert "agent_name" in query and "my-agent" in params
    def test_get_logs_applies_filter_session_id(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_logs(session_id="s-99"))
        query, params = backend.fetch.call_args[0]
        assert "session_id" in query and "s-99" in params
    def test_get_logs_applies_date_range(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end   = datetime(2024, 12, 31, tzinfo=timezone.utc)
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_logs(start_date=start, end_date=end))
        query, params = backend.fetch.call_args[0]
        assert "timestamp" in query and start in params and end in params
    def test_get_logs_limit_offset_in_params(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_logs(limit=25, offset=10))
        _, params = backend.fetch.call_args[0]
        assert 25 in params and 10 in params
    def test_get_logs_redacts_auth_header_in_result(self):
        row = chat_log_row(request_headers='{"Authorization": "Bearer tok", "Content-Type": "application/json"}')
        mod, db, _ = make_read_db(fetch_result=[row])
        result = run(db.get_logs())
        assert result[0]["request_headers"]["Authorization"] == mod._REDACTED_SENTINEL
        assert result[0]["request_headers"]["Content-Type"] == "application/json"
    def test_get_logs_backend_exception_returns_empty(self):
        _, db, backend = make_read_db()
        backend.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        assert run(db.get_logs()) == []
    def test_get_activity_logs_returns_empty_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
            db = mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
        assert run(db.get_activity_logs()) == []
    def test_get_activity_logs_calls_fetch(self):
        _, db, backend = make_read_db(fetch_result=[activity_row()])
        run(db.get_activity_logs())
        backend.fetch.assert_called_once()
    def test_get_activity_logs_query_targets_activity_log(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_activity_logs())
        query, _ = backend.fetch.call_args[0]
        assert "agent_activity_log" in query
    def test_get_activity_logs_returns_list(self):
        _, db, _ = make_read_db(fetch_result=[activity_row()])
        result = run(db.get_activity_logs())
        assert isinstance(result, list) and len(result) == 1
    def test_get_activity_logs_deserialises_chunk_content(self):
        _, db, _ = make_read_db(fetch_result=[activity_row()])
        result = run(db.get_activity_logs())
        assert isinstance(result[0]["chunk_content"], dict)
        assert result[0]["chunk_content"]["text"] == "hello"
    def test_get_activity_logs_applies_filter_session_id(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_activity_logs(session_id="s-42"))
        query, params = backend.fetch.call_args[0]
        assert "session_id" in query and "s-42" in params
    def test_get_activity_logs_redacts_auth_header(self):
        row = activity_row(request_headers='{"Authorization": "Bearer x", "X-Request-Id": "r1"}')
        mod, db, _ = make_read_db(fetch_result=[row])
        result = run(db.get_activity_logs())
        assert result[0]["request_headers"]["Authorization"] == mod._REDACTED_SENTINEL
        assert result[0]["request_headers"]["X-Request-Id"] == "r1"
    def test_get_activity_logs_backend_exception_returns_empty(self):
        _, db, backend = make_read_db()
        backend.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        assert run(db.get_activity_logs()) == []
    def test_get_stats_returns_empty_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
            db = mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
        assert run(db.get_stats()) == {}
    def test_get_stats_calls_fetch_one(self):
        _, db, backend = make_read_db(fetch_one_result=stats_row())
        run(db.get_stats())
        backend.fetch_one.assert_called_once()
    def test_get_stats_query_targets_chat_logs(self):
        _, db, backend = make_read_db(fetch_one_result=stats_row())
        run(db.get_stats())
        query, _ = backend.fetch_one.call_args[0]
        assert "chat_logs" in query
    def test_get_stats_returns_dict(self):
        _, db, _ = make_read_db(fetch_one_result=stats_row())
        assert isinstance(run(db.get_stats()), dict)
    def test_get_stats_all_expected_keys_present(self):
        _, db, _ = make_read_db(fetch_one_result=stats_row())
        expected = {"total_interactions", "unique_sessions", "unique_users", "avg_response_time_ms", "max_response_time_ms", "min_response_time_ms", "total_tokens_sum", "successful_interactions", "failed_interactions"}
        assert set(run(db.get_stats()).keys()) == expected
    def test_get_stats_values_coerced_correctly(self):
        _, db, _ = make_read_db(fetch_one_result=stats_row())
        result = run(db.get_stats())
        assert isinstance(result["total_interactions"], int)
        assert isinstance(result["avg_response_time_ms"], float)
        assert isinstance(result["total_tokens_sum"], int)
    def test_get_stats_none_response_time_stays_none(self):
        _, db, _ = make_read_db(fetch_one_result=stats_row(avg_response_time=None, max_response_time=None, min_response_time=None))
        result = run(db.get_stats())
        assert result["avg_response_time_ms"] is None
        assert result["max_response_time_ms"] is None
        assert result["min_response_time_ms"] is None
    def test_get_stats_filters_agent_name(self):
        _, db, backend = make_read_db(fetch_one_result=stats_row())
        run(db.get_stats(agent_name="bot"))
        query, params = backend.fetch_one.call_args[0]
        assert "agent_name" in query and "bot" in params
    def test_get_stats_filters_user_id(self):
        _, db, backend = make_read_db(fetch_one_result=stats_row())
        run(db.get_stats(user_id="u-99"))
        query, params = backend.fetch_one.call_args[0]
        assert "user_id" in query and "u-99" in params
    def test_get_stats_no_row_returns_empty_dict(self):
        _, db, _ = make_read_db(fetch_one_result=None)
        assert run(db.get_stats()) == {}
    def test_get_stats_backend_exception_returns_empty(self):
        _, db, backend = make_read_db()
        backend.fetch_one = AsyncMock(side_effect=RuntimeError("db down"))
        assert run(db.get_stats()) == {}
    def test_get_user_stats_returns_empty_when_disabled(self):
        with patch.dict(os.environ, {"DB_LOGGING_ENABLED": "false"}):
            mod = load_module(fake_aiosqlite=_FakeAiosqlite(), fake_asyncpg=_FakeAsyncpg())
            db = mod.DatabaseLogger(backends=[], logger=silent_logger())
            run(db.initialize())
        assert run(db.get_user_stats()) == []
    def test_get_user_stats_calls_fetch(self):
        _, db, backend = make_read_db(fetch_result=[{**stats_row(), "user_id": "u1"}])
        run(db.get_user_stats())
        backend.fetch.assert_called_once()
    def test_get_user_stats_query_groups_by_user(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_user_stats())
        query, _ = backend.fetch.call_args[0]
        assert "GROUP BY user_id" in query
    def test_get_user_stats_returns_list(self):
        rows = [{**stats_row(), "user_id": "u1"}, {**stats_row(), "user_id": "u2"}]
        _, db, _ = make_read_db(fetch_result=rows)
        result = run(db.get_user_stats())
        assert isinstance(result, list) and len(result) == 2
    def test_get_user_stats_all_expected_keys_present(self):
        _, db, _ = make_read_db(fetch_result=[{**stats_row(), "user_id": "u1"}])
        expected = {"user_id", "total_interactions", "unique_sessions", "avg_response_time_ms", "max_response_time_ms", "min_response_time_ms", "total_tokens_sum", "successful_interactions", "failed_interactions"}
        assert set(run(db.get_user_stats())[0].keys()) == expected
    def test_get_user_stats_user_id_preserved(self):
        _, db, _ = make_read_db(fetch_result=[{**stats_row(), "user_id": "alice"}])
        assert run(db.get_user_stats())[0]["user_id"] == "alice"
    def test_get_user_stats_filters_agent_name(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_user_stats(agent_name="my-bot"))
        query, params = backend.fetch.call_args[0]
        assert "agent_name" in query and "my-bot" in params
    def test_get_user_stats_no_filter_has_no_where(self):
        _, db, backend = make_read_db(fetch_result=[])
        run(db.get_user_stats())
        query, params = backend.fetch.call_args[0]
        assert "WHERE" not in query
        assert params == ()
    def test_get_user_stats_none_response_time_stays_none(self):
        row = {**stats_row(avg_response_time=None, max_response_time=None, min_response_time=None), "user_id": "u1"}
        _, db, _ = make_read_db(fetch_result=[row])
        assert run(db.get_user_stats())[0]["avg_response_time_ms"] is None
    def test_get_user_stats_backend_exception_returns_empty(self):
        _, db, backend = make_read_db()
        backend.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        assert run(db.get_user_stats()) == []
