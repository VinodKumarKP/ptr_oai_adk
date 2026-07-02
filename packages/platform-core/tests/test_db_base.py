"""
Tests for oai_platform_core.db.base

Covers:
- _resolve_db_path
- BaseSQLiteBackend (per-request connection)
- PersistentSQLiteBackend (persistent connection)
- BasePostgresBackend (asyncpg-backed, mocked)
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite

from oai_platform_core.db.base import (
    _resolve_db_path,
    BaseSQLiteBackend,
    PersistentSQLiteBackend,
    BasePostgresBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


def _silent_logger():
    log = logging.getLogger("test_silent")
    log.handlers = []
    log.addHandler(logging.NullHandler())
    return log


# ---------------------------------------------------------------------------
# Concrete SQLite subclasses (minimal schema)
# ---------------------------------------------------------------------------

class _SimpleSQLiteBackend(BaseSQLiteBackend):
    DEFAULT_DB_NAME = "test_simple.db"

    async def _create_schema(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS items "
                "(id INTEGER PRIMARY KEY, name TEXT)"
            )
            await db.commit()


class _SimplePersistentBackend(PersistentSQLiteBackend):
    DEFAULT_DB_NAME = "test_persistent.db"
    USE_WAL_MODE = True

    async def _create_schema(self):
        conn = await self._get_conn()
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS items "
            "(id INTEGER PRIMARY KEY, name TEXT)"
        )
        await conn.commit()


# ---------------------------------------------------------------------------
# _resolve_db_path
# ---------------------------------------------------------------------------

class TestResolveDbPath:
    def test_sqlite_db_path_env_takes_priority(self, tmp_path, monkeypatch):
        target = str(tmp_path / "override.db")
        monkeypatch.setenv("SQLITE_DB_PATH", target)
        monkeypatch.delenv("SQLITE_DB_DIR", raising=False)
        assert _resolve_db_path("default.db") == target

    def test_sqlite_db_dir_appends_name(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
        monkeypatch.setenv("SQLITE_DB_DIR", str(tmp_path))
        result = _resolve_db_path("my.db")
        assert result == os.path.join(str(tmp_path), "my.db")

    def test_default_name_returned_when_no_env(self, monkeypatch):
        monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
        monkeypatch.delenv("SQLITE_DB_DIR", raising=False)
        assert _resolve_db_path("fallback.db") == "fallback.db"

    def test_no_args_uses_built_in_default(self, monkeypatch):
        monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
        monkeypatch.delenv("SQLITE_DB_DIR", raising=False)
        assert _resolve_db_path() == "data.db"


# ---------------------------------------------------------------------------
# BaseSQLiteBackend
# ---------------------------------------------------------------------------

class TestBaseSQLiteBackend:
    @pytest.fixture
    def db_path(self, tmp_path, monkeypatch):
        path = str(tmp_path / "test.db")
        monkeypatch.setenv("SQLITE_DB_PATH", path)
        return path

    def test_initialize_returns_true(self, db_path):
        backend = _SimpleSQLiteBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is True
        assert backend._db_path == db_path

    def test_initialize_sets_db_path(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        assert backend._db_path == db_path

    def test_execute_and_fetch(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute("INSERT INTO items (name) VALUES (?)", ("alice",)))
        rows = run(backend.fetch("SELECT name FROM items", ()))
        assert rows == [{"name": "alice"}]

    def test_execute_many(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute_many(
            "INSERT INTO items (name) VALUES (?)",
            [("bob",), ("carol",)],
        ))
        rows = run(backend.fetch("SELECT name FROM items ORDER BY name", ()))
        assert [r["name"] for r in rows] == ["bob", "carol"]

    def test_execute_many_empty_is_noop(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        # Should not raise
        run(backend.execute_many("INSERT INTO items (name) VALUES (?)", []))

    def test_fetch_one(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute("INSERT INTO items (name) VALUES (?)", ("dave",)))
        row = run(backend.fetch_one("SELECT name FROM items WHERE name=?", ("dave",)))
        assert row == {"name": "dave"}

    def test_fetch_one_missing_returns_none(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        row = run(backend.fetch_one("SELECT name FROM items WHERE name=?", ("ghost",)))
        assert row is None

    def test_execute_raises_when_not_initialized(self):
        backend = _SimpleSQLiteBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.execute("SELECT 1", ()))

    def test_execute_many_raises_when_not_initialized(self):
        backend = _SimpleSQLiteBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.execute_many("SELECT 1", [(1,)]))

    def test_fetch_raises_when_not_initialized(self):
        backend = _SimpleSQLiteBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.fetch("SELECT 1", ()))

    def test_fetch_one_raises_when_not_initialized(self):
        backend = _SimpleSQLiteBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.fetch_one("SELECT 1", ()))

    def test_close_clears_db_path(self, db_path):
        backend = _SimpleSQLiteBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.close())
        assert backend._db_path is None

    def test_name_attribute(self):
        assert _SimpleSQLiteBackend.name == "sqlite"

    def test_placeholder_attribute(self):
        assert _SimpleSQLiteBackend.PLACEHOLDER == "?"

    def test_initialize_without_logger(self, db_path):
        backend = _SimpleSQLiteBackend()
        result = run(backend.initialize(None))
        assert result is True

    def test_initialize_fails_when_schema_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "bad.db"))

        class _BrokenBackend(BaseSQLiteBackend):
            DEFAULT_DB_NAME = "broken.db"
            async def _create_schema(self):
                raise RuntimeError("schema failure")

        backend = _BrokenBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is False
        assert backend._db_path is None


# ---------------------------------------------------------------------------
# PersistentSQLiteBackend
# ---------------------------------------------------------------------------

class TestPersistentSQLiteBackend:
    @pytest.fixture
    def db_path(self, tmp_path, monkeypatch):
        path = str(tmp_path / "persistent.db")
        monkeypatch.setenv("SQLITE_DB_PATH", path)
        return path

    def test_initialize_returns_true(self, db_path):
        backend = _SimplePersistentBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is True

    def test_execute_and_fetch(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute("INSERT INTO items (name) VALUES (?)", ("alice",)))
        rows = run(backend.fetch("SELECT name FROM items", ()))
        run(backend.close())
        assert rows == [{"name": "alice"}]

    def test_execute_many(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute_many(
            "INSERT INTO items (name) VALUES (?)",
            [("x",), ("y",)],
        ))
        rows = run(backend.fetch("SELECT name FROM items ORDER BY name", ()))
        run(backend.close())
        assert len(rows) == 2

    def test_execute_many_empty_is_noop(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute_many("INSERT INTO items (name) VALUES (?)", []))
        run(backend.close())

    def test_fetch_one(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute("INSERT INTO items (name) VALUES (?)", ("bob",)))
        row = run(backend.fetch_one("SELECT name FROM items WHERE name=?", ("bob",)))
        run(backend.close())
        assert row == {"name": "bob"}

    def test_fetch_one_missing_returns_none(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        row = run(backend.fetch_one("SELECT name FROM items WHERE name=?", ("ghost",)))
        run(backend.close())
        assert row is None

    def test_execute_raises_when_not_initialized(self):
        backend = _SimplePersistentBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.execute("SELECT 1", ()))

    def test_fetch_raises_when_not_initialized(self):
        backend = _SimplePersistentBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.fetch("SELECT 1", ()))

    def test_close_closes_connection(self, db_path):
        backend = _SimplePersistentBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.close())
        assert backend._conn is None
        assert backend._db_path is None

    def test_close_idempotent_when_not_open(self):
        backend = _SimplePersistentBackend()
        # Should not raise even without initialization
        run(backend.close())

    def test_get_conn_raises_before_initialize(self):
        backend = _SimplePersistentBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend._get_conn())

    def test_initialize_fails_when_schema_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "bad.db"))

        class _BrokenBackend(PersistentSQLiteBackend):
            DEFAULT_DB_NAME = "broken.db"
            async def _create_schema(self):
                raise RuntimeError("schema failure")

        backend = _BrokenBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is False


# ---------------------------------------------------------------------------
# BasePostgresBackend (asyncpg mocked)
# ---------------------------------------------------------------------------

def _make_fake_asyncpg():
    """Build a mock asyncpg module with pool/connection stubs."""
    fake_conn = AsyncMock()
    fake_conn.fetchval = AsyncMock(return_value=1)
    fake_conn.execute = AsyncMock()
    fake_conn.close = AsyncMock()

    fake_pool_conn = AsyncMock()
    fake_pool_conn.execute = AsyncMock()
    fake_pool_conn.executemany = AsyncMock()
    fake_pool_conn.fetch = AsyncMock(return_value=[{"col": "val"}])
    fake_pool_conn.fetchrow = AsyncMock(return_value={"col": "val"})

    # pool.acquire() returns an async context manager
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=fake_pool_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    fake_pool = AsyncMock()
    fake_pool.acquire = MagicMock(return_value=acquire_cm)
    fake_pool.close = AsyncMock()

    fake_asyncpg = MagicMock()
    fake_asyncpg.connect = AsyncMock(return_value=fake_conn)
    fake_asyncpg.create_pool = AsyncMock(return_value=fake_pool)

    return fake_asyncpg, fake_pool, fake_pool_conn, fake_conn


class _ConcretePostgresBackend(BasePostgresBackend):
    DEFAULT_PORT = "5432"
    DEFAULT_DB_NAME = "test_logs"

    async def _create_schema(self, logger=None):
        pass  # no-op schema for tests


class TestBasePostgresBackend:
    def test_initialize_returns_true_when_asyncpg_available(self, monkeypatch):
        fake_asyncpg, fake_pool, fake_pool_conn, _ = _make_fake_asyncpg()
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)
        monkeypatch.delenv("LOGGING_DB_HOST", raising=False)

        backend = _ConcretePostgresBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is True

    def test_initialize_returns_false_when_asyncpg_unavailable(self, monkeypatch):
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", False)

        backend = _ConcretePostgresBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is False

    def test_initialize_returns_false_when_pool_creation_fails(self, monkeypatch):
        fake_asyncpg, _, _, _ = _make_fake_asyncpg()
        fake_asyncpg.create_pool = AsyncMock(side_effect=OSError("refused"))
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        result = run(backend.initialize(_silent_logger()))
        assert result is False

    def test_execute_raises_when_not_initialized(self):
        backend = _ConcretePostgresBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.execute("SELECT 1", ()))

    def test_execute_many_raises_when_not_initialized(self):
        backend = _ConcretePostgresBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.execute_many("SELECT 1", [(1,)]))

    def test_fetch_raises_when_not_initialized(self):
        backend = _ConcretePostgresBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.fetch("SELECT 1", ()))

    def test_fetch_one_raises_when_not_initialized(self):
        backend = _ConcretePostgresBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            run(backend.fetch_one("SELECT 1", ()))

    def test_close_before_initialize_is_safe(self):
        backend = _ConcretePostgresBackend()
        run(backend.close())  # should not raise

    def test_name_attribute(self):
        assert _ConcretePostgresBackend.name == "postgres"

    def test_placeholder_attribute(self):
        assert _ConcretePostgresBackend.PLACEHOLDER == "$"

    def test_execute_calls_pool_connection(self, monkeypatch):
        fake_asyncpg, fake_pool, fake_pool_conn, _ = _make_fake_asyncpg()
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))
        # Reset mock after initialize (which calls execute("SELECT 1"))
        fake_pool_conn.execute.reset_mock()
        run(backend.execute("INSERT INTO t VALUES ($1)", ("v",)))
        fake_pool_conn.execute.assert_called_once_with("INSERT INTO t VALUES ($1)", "v")

    def test_execute_many_noop_on_empty(self, monkeypatch):
        fake_asyncpg, _, fake_pool_conn, _ = _make_fake_asyncpg()
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))
        run(backend.execute_many("INSERT INTO t VALUES ($1)", []))
        fake_pool_conn.executemany.assert_not_called()

    def test_fetch_returns_list_of_dicts(self, monkeypatch):
        fake_asyncpg, _, fake_pool_conn, _ = _make_fake_asyncpg()
        fake_pool_conn.fetch = AsyncMock(return_value=[{"col": "val"}])
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))
        result = run(backend.fetch("SELECT col FROM t", ()))
        assert result == [{"col": "val"}]

    def test_fetch_one_returns_dict(self, monkeypatch):
        fake_asyncpg, _, fake_pool_conn, _ = _make_fake_asyncpg()
        fake_pool_conn.fetchrow = AsyncMock(return_value={"col": "val"})
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))
        result = run(backend.fetch_one("SELECT col FROM t", ()))
        assert result == {"col": "val"}

    def test_fetch_one_returns_none_when_no_row(self, monkeypatch):
        fake_asyncpg, _, fake_pool_conn, _ = _make_fake_asyncpg()
        fake_pool_conn.fetchrow = AsyncMock(return_value=None)
        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))
        result = run(backend.fetch_one("SELECT col FROM t WHERE id=1", ()))
        assert result is None

    def test_env_vars_used_for_connection(self, monkeypatch):
        captured = {}

        async def _capture_create_pool(dsn, **kwargs):
            captured["dsn"] = dsn
            captured["kwargs"] = kwargs
            fake_asyncpg, pool, conn, _ = _make_fake_asyncpg()
            return pool

        fake_asyncpg, _, _, _ = _make_fake_asyncpg()
        fake_asyncpg.create_pool = _capture_create_pool

        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)
        monkeypatch.setenv("LOGGING_DB_HOST", "db.example.com")
        monkeypatch.setenv("LOGGING_DB_PORT", "5433")
        monkeypatch.setenv("LOGGING_DB_NAME", "mydb")
        monkeypatch.setenv("LOGGING_DB_USER", "admin")
        monkeypatch.setenv("LOGGING_DB_PASSWORD", "secret")
        monkeypatch.setenv("DB_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("DB_POOL_MAX_SIZE", "8")
        monkeypatch.setenv("DB_POOL_TIMEOUT", "45")

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))

        assert "db.example.com" in captured["dsn"]
        assert "5433" in captured["dsn"]
        assert "mydb" in captured["dsn"]
        assert captured["kwargs"]["min_size"] == 3
        assert captured["kwargs"]["max_size"] == 8
        assert captured["kwargs"]["command_timeout"] == 45

    def test_default_env_vars_applied(self, monkeypatch):
        captured = {}

        async def _capture_create_pool(dsn, **kwargs):
            captured["dsn"] = dsn
            captured["kwargs"] = kwargs
            _, pool, _, _ = _make_fake_asyncpg()
            return pool

        fake_asyncpg, _, _, _ = _make_fake_asyncpg()
        fake_asyncpg.create_pool = _capture_create_pool

        import oai_platform_core.db.base as _db_base
        monkeypatch.setattr(_db_base, "asyncpg", fake_asyncpg)
        monkeypatch.setattr(_db_base, "_ASYNCPG_AVAILABLE", True)
        for k in ["LOGGING_DB_HOST", "LOGGING_DB_PORT", "LOGGING_DB_NAME",
                  "LOGGING_DB_USER", "LOGGING_DB_PASSWORD",
                  "DB_POOL_MIN_SIZE", "DB_POOL_MAX_SIZE", "DB_POOL_TIMEOUT"]:
            monkeypatch.delenv(k, raising=False)

        backend = _ConcretePostgresBackend()
        run(backend.initialize(_silent_logger()))

        assert "localhost" in captured["dsn"]
        assert "5432" in captured["dsn"]
        assert "test_logs" in captured["dsn"]
        assert captured["kwargs"]["min_size"] == 2
        assert captured["kwargs"]["max_size"] == 10
