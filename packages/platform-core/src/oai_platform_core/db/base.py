"""
Shared async database backend infrastructure for OAI platform services.

Provides three concrete base classes that handle all connection-management
boilerplate.  Domain-specific packages subclass the appropriate backend,
declare a handful of class-level configuration variables, and implement
only the ``_create_schema()`` method that creates their tables.

Usage example
-------------
::

    from oai_platform_core.db.base import BasePostgresBackend, BaseSQLiteBackend

    class PostgresBackend(BasePostgresBackend):
        DEFAULT_PORT    = "5433"
        DEFAULT_DB_NAME = "my_service_logs"

        ITEM_INSERT = "INSERT INTO items (name) VALUES ($1)"

        async def _create_schema(self, logger=None) -> None:
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE TABLE IF NOT EXISTS items (...)")

    class SQLiteBackend(BaseSQLiteBackend):
        DEFAULT_DB_NAME = "my_service.db"

        ITEM_INSERT = "INSERT INTO items (name) VALUES (?)"

        async def _create_schema(self) -> None:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("CREATE TABLE IF NOT EXISTS items (...)")
                await db.commit()
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional driver imports — all backends gracefully degrade if unavailable
# ---------------------------------------------------------------------------

try:
    import asyncpg
    from asyncpg.pool import Pool as AsyncpgPool

    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False
    AsyncpgPool = None  # type: ignore[assignment, misc]

try:
    import aiosqlite

    _AIOSQLITE_AVAILABLE = True
except ImportError:
    _AIOSQLITE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resolve_db_path(default_name: str = "data.db") -> str:
    """
    Resolve the SQLite database file path from environment variables.

    Resolution order:

    1. ``SQLITE_DB_PATH`` — full path, used as-is.
    2. ``SQLITE_DB_DIR``  — directory; appends ``default_name``.
    3. ``default_name``   — relative path in the current working directory.
    """
    full_path = os.environ.get("SQLITE_DB_PATH")
    if full_path:
        return full_path
    db_dir = os.environ.get("SQLITE_DB_DIR")
    if db_dir:
        return os.path.join(db_dir, default_name)
    return default_name


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DatabaseBackend(ABC):
    """Abstract interface satisfied by all backend implementations."""

    name: str = "unnamed"

    @abstractmethod
    async def initialize(self, logger: Optional[logging.Logger]) -> bool: ...

    @abstractmethod
    async def execute(self, query: str, params: tuple) -> None: ...

    @abstractmethod
    async def execute_many(self, query: str, params_seq: List[tuple]) -> None: ...

    @abstractmethod
    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# PostgreSQL base backend
# ---------------------------------------------------------------------------

class BasePostgresBackend(DatabaseBackend):
    """
    Full connection-pool lifecycle management for PostgreSQL.

    Subclasses MUST declare:
    - ``DEFAULT_PORT``    (str)  — fallback for ``LOGGING_DB_PORT``
    - ``DEFAULT_DB_NAME`` (str)  — fallback for ``LOGGING_DB_NAME``

    Subclasses MUST implement:
    - ``_create_schema(self, logger)`` — async DDL for domain-specific tables

    Connection parameters are read from environment variables so the same
    Docker/Kubernetes config pattern works across all services:

    +----------------------+--------------------------------------+
    | Env var              | Default in base                      |
    +======================+======================================+
    | LOGGING_DB_HOST      | localhost                            |
    | LOGGING_DB_PORT      | ``DEFAULT_PORT``                     |
    | LOGGING_DB_NAME      | ``DEFAULT_DB_NAME``                  |
    | LOGGING_DB_USER      | postgres                             |
    | LOGGING_DB_PASSWORD  | postgres                             |
    | DB_POOL_MIN_SIZE     | 2                                    |
    | DB_POOL_MAX_SIZE     | 10                                   |
    | DB_POOL_TIMEOUT      | 120                                  |
    +----------------------+--------------------------------------+
    """

    name = "postgres"
    PLACEHOLDER = "$"

    DEFAULT_PORT: str = "5432"
    DEFAULT_DB_NAME: str = "app_logs"

    def __init__(self) -> None:
        self._pool: Optional[AsyncpgPool] = None

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _ASYNCPG_AVAILABLE:
            if logger:
                logger.debug("asyncpg not installed — PostgreSQL backend unavailable.")
            return False
        if logger:
            logger.info("PostgreSQL backend available.")

        host     = os.environ.get("LOGGING_DB_HOST", "localhost")
        port     = os.environ.get("LOGGING_DB_PORT", self.DEFAULT_PORT)
        name     = os.environ.get("LOGGING_DB_NAME", self.DEFAULT_DB_NAME)
        user     = os.environ.get("LOGGING_DB_USER", "postgres")
        password = os.environ.get("LOGGING_DB_PASSWORD", "postgres")
        min_size = int(os.environ.get("DB_POOL_MIN_SIZE", "2"))
        max_size = int(os.environ.get("DB_POOL_MAX_SIZE", "10"))
        timeout  = int(os.environ.get("DB_POOL_TIMEOUT", "120"))

        await self._ensure_database(host, int(port), name, user, password, logger)

        dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        try:
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=min_size,
                max_size=max_size,
                command_timeout=timeout,
                timeout=5,
            )
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            if logger:
                logger.info(f"PostgreSQL backend ready: {host}:{port}/{name}")
            await self._create_schema(logger)
            return True
        except Exception as exc:
            if logger:
                logger.warning(f"PostgreSQL backend unavailable at {dsn}: {exc}")
            await self._cleanup()
            return False

    @staticmethod
    async def _ensure_database(
        host: str,
        port: int,
        name: str,
        user: str,
        password: str,
        logger: Optional[logging.Logger],
    ) -> None:
        """
        Create the target database if it does not already exist.

        Connects to the ``postgres`` maintenance database, checks
        ``pg_database``, and issues ``CREATE DATABASE`` when absent.
        ``CREATE DATABASE`` cannot run inside a transaction block; asyncpg
        auto-commits statements run outside an explicit transaction, so this
        is safe.  Any error is logged as a warning and swallowed — the
        subsequent ``create_pool`` call will surface a clear error if the
        database is still missing.
        """
        try:
            conn = await asyncpg.connect(
                host=host, port=port, database="postgres",
                user=user, password=password,
            )
            try:
                exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1", name
                )
                if not exists:
                    await conn.execute(f'CREATE DATABASE "{name}"')
                    if logger:
                        logger.info("Created PostgreSQL database: %s", name)
                else:
                    if logger:
                        logger.debug("PostgreSQL database already exists: %s", name)
            finally:
                await conn.close()
        except Exception as exc:
            if logger:
                logger.warning(
                    "Could not ensure database '%s' exists (will attempt connection anyway): %s",
                    name, exc,
                )

    async def execute(self, query: str, params: tuple) -> None:
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized")
        if not params_seq:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(query, params_seq)

    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]:
        if self._pool is None:
            raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        return dict(row) if row else None

    async def close(self) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @abstractmethod
    async def _create_schema(self, logger: Optional[logging.Logger]) -> None:
        """Subclasses implement domain-specific DDL here."""


# ---------------------------------------------------------------------------
# SQLite base backend — per-request open/use/close pattern
# ---------------------------------------------------------------------------

class BaseSQLiteBackend(DatabaseBackend):
    """
    SQLite backend using the per-request open/use/close connection pattern.

    Every method opens a new ``aiosqlite.connect()`` context, executes its
    query, and closes the connection.  This is the simplest safe pattern for
    low-to-medium write throughput.

    Subclasses MUST declare:
    - ``DEFAULT_DB_NAME`` (str) — fallback filename when env vars are unset

    Subclasses MUST implement:
    - ``_create_schema(self)`` — async DDL for domain-specific tables

    DB path resolution (see ``_resolve_db_path``):

    1. ``SQLITE_DB_PATH`` env var — full path, used as-is.
    2. ``SQLITE_DB_DIR`` env var  — directory; appends ``DEFAULT_DB_NAME``.
    3. ``DEFAULT_DB_NAME``        — relative path in the CWD.
    """

    name = "sqlite"
    PLACEHOLDER = "?"

    DEFAULT_DB_NAME: str = "data.db"

    def __init__(self) -> None:
        self._db_path: Optional[str] = None

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _AIOSQLITE_AVAILABLE:
            if logger:
                logger.warning("aiosqlite not installed — SQLite backend unavailable.")
            return False
        self._db_path = _resolve_db_path(self.DEFAULT_DB_NAME)
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        try:
            await self._create_schema()
            if logger:
                logger.info(f"SQLite backend ready: {self._db_path}")
            return True
        except Exception as exc:
            if logger:
                logger.error(f"SQLite backend failed to initialise: {exc}")
            self._db_path = None
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(query, params)
            await db.commit()

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        if not params_seq:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(query, params_seq)
            await db.commit()

    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
        return dict(row) if row else None

    async def close(self) -> None:
        self._db_path = None

    @abstractmethod
    async def _create_schema(self) -> None:
        """Subclasses implement domain-specific DDL here."""


# ---------------------------------------------------------------------------
# SQLite persistent-connection backend (write-lock, optional WAL mode)
# ---------------------------------------------------------------------------

class PersistentSQLiteBackend(DatabaseBackend):
    """
    SQLite backend that keeps a single persistent connection for the process
    lifetime.

    All writes are serialised through an ``asyncio.Lock`` to prevent
    "database is locked" errors under concurrent coroutines.

    Set ``USE_WAL_MODE = True`` on the subclass to enable WAL + NORMAL
    synchronous mode — recommended for high write throughput (e.g. activity
    logging, chat log streaming).

    Subclasses MUST declare:
    - ``DEFAULT_DB_NAME`` (str)  — fallback filename when env vars are unset
    - ``USE_WAL_MODE``    (bool) — enable WAL journal mode (default False)

    Subclasses MUST implement:
    - ``_create_schema(self)`` — async DDL; call ``await self._get_conn()``
      to obtain the connection.
    """

    name = "sqlite"
    PLACEHOLDER = "?"

    DEFAULT_DB_NAME: str = "data.db"
    USE_WAL_MODE: bool = False

    def __init__(self) -> None:
        self._db_path: Optional[str] = None
        self._conn: Optional["aiosqlite.Connection"] = None  # type: ignore[name-defined]
        self._write_lock: asyncio.Lock = asyncio.Lock()

    async def _get_conn(self) -> "aiosqlite.Connection":  # type: ignore[name-defined]
        """Return the persistent connection, creating it lazily on first call."""
        if self._conn is None:
            if self._db_path is None:
                raise RuntimeError("SQLiteBackend not initialized")
            conn = await aiosqlite.connect(self._db_path)
            if self.USE_WAL_MODE:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
            await conn.execute("PRAGMA foreign_keys = ON;")
            await conn.commit()
            conn.row_factory = aiosqlite.Row
            self._conn = conn
        return self._conn

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _AIOSQLITE_AVAILABLE:
            if logger:
                logger.warning("aiosqlite not installed — SQLite backend unavailable.")
            return False
        self._db_path = _resolve_db_path(self.DEFAULT_DB_NAME)
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        try:
            await self._create_schema()
            if logger:
                logger.info(f"SQLite backend ready: {self._db_path}")
            return True
        except Exception as exc:
            if logger:
                logger.error(f"SQLite backend failed to initialise: {exc}")
            self._db_path = None
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        async with self._write_lock:
            db = await self._get_conn()
            await db.execute(query, params)
            await db.commit()

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        if not params_seq:
            return
        async with self._write_lock:
            db = await self._get_conn()
            await db.executemany(query, params_seq)
            await db.commit()

    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        db = await self._get_conn()
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend not initialized")
        db = await self._get_conn()
        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def close(self) -> None:
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._db_path = None

    @abstractmethod
    async def _create_schema(self) -> None:
        """Subclasses implement domain-specific DDL here."""
