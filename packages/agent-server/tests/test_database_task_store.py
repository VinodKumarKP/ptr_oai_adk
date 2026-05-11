"""Tests for DatabaseTaskStore (a2a/database_task_store.py).

Uses an in-memory SQLite-style fake backend to keep tests fast and isolated.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oai_agent_server.a2a.database_task_store import (
    DatabaseTaskStore,
    _LazyTaskStoreProxy,
    build_task_store_from_env,
)
from oai_agent_server.utils.database_logger import (
    PostgresBackend,
    SQLiteBackend,
)

try:
    from a2a.types import a2a_pb2
    from a2a.types.a2a_pb2 import Task
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False

pytestmark = pytest.mark.skipif(not A2A_AVAILABLE, reason="a2a-sdk not installed")


# ----------------- Fakes -----------------

class FakeSQLiteBackend(SQLiteBackend):
    """In-memory SQLite-style backend backed by a Python dict.

    We avoid using real aiosqlite to keep tests pure-Python and fast. Stores
    rows in a dict keyed by task_id; supports the small subset of SQL the
    DatabaseTaskStore actually emits.
    """

    name = "sqlite"
    PLACEHOLDER = "?"

    def __init__(self):
        # Skip parent __init__ (which sets up an asyncio.Lock at import time).
        self._rows: dict = {}
        self._db_path = ":memory:"
        self._initialized = True
        self.executed_queries = []

    async def execute(self, query, params=()):
        self.executed_queries.append((query, params))
        q = query.strip().upper()
        if q.startswith("CREATE") or q.startswith("ALTER"):
            return
        if q.startswith("INSERT OR REPLACE"):
            (task_id, owner, context_id, status, status_ts,
             task_data, created_at, updated_at, expires_at) = params
            self._rows[task_id] = {
                "task_id": task_id, "owner": owner, "context_id": context_id,
                "status": status, "status_ts": status_ts, "task_data": task_data,
                "created_at": created_at, "updated_at": updated_at,
                "expires_at": expires_at,
            }
            return
        if q.startswith("DELETE FROM A2A_TASKS WHERE TASK_ID"):
            task_id, owner = params
            row = self._rows.get(task_id)
            if row and row["owner"] == owner:
                del self._rows[task_id]
            return
        if q.startswith("DELETE FROM A2A_TASKS WHERE EXPIRES_AT"):
            now_iso = params[0]
            for tid in list(self._rows):
                exp = self._rows[tid]["expires_at"]
                if exp and exp < now_iso:
                    del self._rows[tid]
            return

    async def execute_many(self, query, params_seq):
        for p in params_seq:
            await self.execute(query, p)

    async def fetch(self, query, params=()):
        rows = self._filter(query, params)
        # Apply LIMIT if present
        q_upper = query.upper()
        if "LIMIT" in q_upper:
            limit = params[-1] if params else None
            if isinstance(limit, int):
                rows = rows[:limit]
        return rows

    async def fetch_one(self, query, params=()):
        rows = self._filter(query, params)
        if query.strip().upper().startswith("SELECT COUNT"):
            return {"c": len(self._all_matching_count(query, params))}
        return rows[0] if rows else None

    def _all_matching_count(self, query, params):
        return [r for r in self._rows.values()
                if self._row_matches(r, query, params)]

    def _filter(self, query, params):
        q = query.upper()
        if q.strip().startswith("SELECT COUNT"):
            return [{"c": len(self._all_matching_count(query, params))}]
        rows = [r for r in self._rows.values()
                if self._row_matches(r, query, params)]
        return rows

    def _row_matches(self, row, query, params):
        # Crude param-by-param match for the queries we emit.
        # Look for "task_id = ?" / "owner = ?" / "status = ?" / etc.
        # Simpler: check sequentially for known patterns.
        i = 0
        if "TASK_ID = ?" in query.upper():
            if row["task_id"] != params[i]:
                return False
            i += 1
        if "OWNER = ?" in query.upper():
            if row["owner"] != params[i]:
                return False
            i += 1
        if "EXPIRES_AT >=" in query.upper() and "EXPIRES_AT IS NULL" in query.upper():
            now_iso = params[i]
            if row["expires_at"] is not None and row["expires_at"] < now_iso:
                return False
            i += 1
        if "CONTEXT_ID = ?" in query.upper():
            if row["context_id"] != params[i]:
                return False
            i += 1
        if "STATUS = ?" in query.upper():
            if row["status"] != params[i]:
                return False
            i += 1
        return True

    async def close(self):
        self._rows.clear()


def make_task(task_id="t1", context_id="ctx1"):
    """Build a minimal protobuf Task."""
    t = Task()
    t.id = task_id
    t.context_id = context_id
    return t


def make_context(user_name="alice"):
    ctx = MagicMock()
    ctx.user = MagicMock()
    ctx.user.user_name = user_name
    return ctx


# ----------------- Tests -----------------

class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_creates_table_sqlite(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False  # force init
        await store.initialize()
        assert any("CREATE TABLE" in q.upper() and "A2A_TASKS" in q.upper()
                   for q, _ in backend.executed_queries)

    @pytest.mark.asyncio
    async def test_initialize_creates_indexes(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        idx_queries = [q for q, _ in backend.executed_queries if "INDEX" in q.upper()]
        assert len(idx_queries) >= 3  # context, owner, expires

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        first = len(backend.executed_queries)
        await store.initialize()  # second call should be a no-op
        assert len(backend.executed_queries) == first

    @pytest.mark.asyncio
    async def test_initialize_postgres_uses_jsonb(self):
        backend = MagicMock(spec=PostgresBackend)
        backend.execute = AsyncMock()
        backend.name = "postgres"
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        ddl_calls = [c.args[0] for c in backend.execute.call_args_list]
        assert any("JSONB" in q for q in ddl_calls)


class TestSaveAndGet:
    async def _make_store(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        return store, backend

    @pytest.mark.asyncio
    async def test_save_inserts_task(self):
        store, backend = await self._make_store()
        await store.save(make_task("t1"), make_context("alice"))
        assert "t1" in backend._rows

    @pytest.mark.asyncio
    async def test_save_serializes_task_data(self):
        store, backend = await self._make_store()
        await store.save(make_task("t1", "ctxA"), make_context("alice"))
        td = backend._rows["t1"]["task_data"]
        # Stored as JSON string
        assert isinstance(td, str)
        decoded = json.loads(td)
        assert decoded["id"] == "t1"

    @pytest.mark.asyncio
    async def test_save_upsert_replaces_existing(self):
        store, backend = await self._make_store()
        await store.save(make_task("t1", "ctx1"), make_context("alice"))
        await store.save(make_task("t1", "ctx2"), make_context("alice"))
        assert backend._rows["t1"]["context_id"] == "ctx2"

    @pytest.mark.asyncio
    async def test_get_returns_task(self):
        store, _ = await self._make_store()
        await store.save(make_task("t1"), make_context("alice"))
        result = await store.get("t1", make_context("alice"))
        assert result is not None
        assert result.id == "t1"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        store, _ = await self._make_store()
        result = await store.get("nope", make_context("alice"))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_owner_scoped(self):
        store, _ = await self._make_store()
        await store.save(make_task("t1"), make_context("alice"))
        # bob cannot see alice's task
        result = await store.get("t1", make_context("bob"))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_expired_returns_none(self):
        store, backend = await self._make_store()
        await store.save(make_task("t1"), make_context("alice"))
        # Manually expire it
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        backend._rows["t1"]["expires_at"] = past
        result = await store.get("t1", make_context("alice"))
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_honored(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend, ttl_seconds=1)
        store._initialized = False
        await store.initialize()
        await store.save(make_task("t1"), make_context("alice"))
        # Immediately fetch — works
        assert await store.get("t1", make_context("alice")) is not None
        # Force-expire by rewinding
        past = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        backend._rows["t1"]["expires_at"] = past
        assert await store.get("t1", make_context("alice")) is None


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_removes_task(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        await store.save(make_task("t1"), make_context("alice"))
        await store.delete("t1", make_context("alice"))
        assert await store.get("t1", make_context("alice")) is None

    @pytest.mark.asyncio
    async def test_delete_owner_scoped(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        await store.save(make_task("t1"), make_context("alice"))
        # bob's delete is a no-op
        await store.delete("t1", make_context("bob"))
        assert "t1" in backend._rows


class TestCleanup:
    @pytest.mark.asyncio
    async def test_delete_expired_removes_old_rows(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend)
        store._initialized = False
        await store.initialize()
        await store.save(make_task("t1"), make_context("alice"))
        await store.save(make_task("t2"), make_context("alice"))
        past = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        backend._rows["t1"]["expires_at"] = past
        await store._delete_expired()
        assert "t1" not in backend._rows
        assert "t2" in backend._rows

    @pytest.mark.asyncio
    async def test_shutdown_cancels_cleanup_task(self):
        backend = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend, cleanup_interval_seconds=60)
        store._initialized = False
        await store.initialize()
        # initialize should have started a cleanup task
        assert store._cleanup_task is not None
        await store.shutdown()
        assert store._cleanup_task is None


class TestSerialization:
    def test_is_expired_handles_str(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert DatabaseTaskStore._is_expired(past) is True

    def test_is_expired_handles_future(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        assert DatabaseTaskStore._is_expired(future) is False

    def test_is_expired_handles_none(self):
        assert DatabaseTaskStore._is_expired(None) is False

    def test_is_expired_handles_invalid(self):
        assert DatabaseTaskStore._is_expired("not-a-date") is False

    def test_resolve_owner_no_user(self):
        ctx = MagicMock()
        ctx.user = None
        assert DatabaseTaskStore._resolve_owner(ctx) == ""

    def test_resolve_owner_with_user(self):
        ctx = MagicMock()
        ctx.user = MagicMock()
        ctx.user.user_name = "alice"
        assert DatabaseTaskStore._resolve_owner(ctx) == "alice"


class TestLazyTaskStoreProxy:
    @pytest.mark.asyncio
    async def test_proxy_blocks_until_bind(self):
        proxy = _LazyTaskStoreProxy()
        # Calling save before bind should block; use a short timeout.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                proxy.save(make_task("t1"), make_context("alice")),
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_proxy_delegates_after_bind(self):
        proxy = _LazyTaskStoreProxy()
        delegate = MagicMock()
        delegate.save = AsyncMock(return_value=None)
        delegate.get = AsyncMock(return_value="task")
        delegate.delete = AsyncMock(return_value=None)
        delegate.list = AsyncMock(return_value="list")
        proxy.bind(delegate)
        await proxy.save("t", "c")
        delegate.save.assert_called_once_with("t", "c")
        assert await proxy.get("t", "c") == "task"
        assert await proxy.list("p", "c") == "list"
        await proxy.delete("t", "c")
        delegate.delete.assert_called_once()


class TestBuildTaskStoreFromEnv:
    @pytest.mark.asyncio
    async def test_memory_mode(self, monkeypatch):
        monkeypatch.setenv("A2A_TASK_STORE", "memory")
        store = await build_task_store_from_env(db_logger=None)
        from a2a.server.tasks import InMemoryTaskStore
        assert isinstance(store, InMemoryTaskStore)

    @pytest.mark.asyncio
    async def test_falls_back_to_memory_when_no_backend(self, monkeypatch):
        monkeypatch.setenv("A2A_TASK_STORE", "database")
        # Force candidate backends to fail
        with patch.object(PostgresBackend, "initialize", AsyncMock(return_value=False)), \
             patch.object(SQLiteBackend, "initialize", AsyncMock(return_value=False)):
            store = await build_task_store_from_env(db_logger=None)
        from a2a.server.tasks import InMemoryTaskStore
        assert isinstance(store, InMemoryTaskStore)

    @pytest.mark.asyncio
    async def test_reuses_db_logger_backend(self, monkeypatch):
        monkeypatch.setenv("A2A_TASK_STORE", "database")
        backend = FakeSQLiteBackend()
        db_logger = MagicMock()
        db_logger._backend = backend
        store = await build_task_store_from_env(db_logger=db_logger)
        assert isinstance(store, DatabaseTaskStore)
        assert store._backend is backend


class TestSetBackend:
    def test_set_backend_resets_initialized(self):
        backend1 = FakeSQLiteBackend()
        store = DatabaseTaskStore(backend=backend1)
        store._initialized = True
        backend2 = FakeSQLiteBackend()
        store.set_backend(backend2)
        assert store._backend is backend2
        assert store._initialized is False
