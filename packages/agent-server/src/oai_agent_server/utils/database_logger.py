"""
Database logger for tracking agent chat interactions.

Architecture
------------
DatabaseLogger is backend-agnostic. It holds a DatabaseBackend instance and
delegates all SQL work to it. Two backends are bundled:

  PostgresBackend  – asyncpg connection pool (requires asyncpg)
  SQLiteBackend    – aiosqlite, one connection per operation (requires aiosqlite)

Adding a new backend (e.g. MySQL, ClickHouse) requires only:
  1. Subclass DatabaseBackend and implement its abstract methods.
  2. Pass an instance to DatabaseLogger (or add it to the fallback list).

Usage
-----
    logger = DatabaseLogger(
        backends=[PostgresBackend(), SQLiteBackend()],   # tried in order
        app_logger=my_logger,
    )
    await logger.initialize()
    await logger.log_interaction(...)
    await logger.close()

Environment variables (all optional, read by the bundled backends):
  DB_LOGGING_ENABLED       – 'true' to enable (default: false)
  LOGGING_DB_HOST/PORT/NAME/USER/PASSWORD – PostgreSQL connection details
  DB_POOL_MIN_SIZE / MAX_SIZE / TIMEOUT   – asyncpg pool tuning
  SQLITE_DB_PATH           – full path for the SQLite file; takes precedence over SQLITE_DB_DIR
  SQLITE_DB_DIR            – directory for the SQLite file; filename defaults to agent_logs.db
                             (ignored when SQLITE_DB_PATH is set)

Sensitive header fields stripped before storage:
  Authorization, Cookie, Set-Cookie, X-Api-Key, X-Auth-Token, Proxy-Authorization
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

try:
    import asyncpg
    from asyncpg.pool import Pool as AsyncpgPool

    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False
    AsyncpgPool = None  # type: ignore[assignment,misc]

try:
    import aiosqlite

    _AIOSQLITE_AVAILABLE = True
except ImportError:
    _AIOSQLITE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Sensitive header keys that must never be persisted
# ---------------------------------------------------------------------------

_REDACTED_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)

_REDACTED_SENTINEL = "***REDACTED***"

# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------


class DatabaseBackend(ABC):
    """
    Contract that every storage backend must satisfy.

    Implementations are responsible for:
      * creating / migrating their own schema
      * providing the correct paramstyle placeholders
      * managing their own connections / pools
    """

    # Human-readable name used in log messages.
    name: str = "unnamed"

    @abstractmethod
    async def initialize(self, app_logger: Optional[logging.Logger]) -> bool:
        """
        Set up the backend (connect, create tables, run migrations).

        Returns True on success, False if this backend is unavailable.
        Should never raise — failures must be caught and logged internally.
        """

    @abstractmethod
    async def execute(self, query: str, params: tuple) -> None:
        """Execute a single write statement."""

    @abstractmethod
    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        """Execute the same write statement for each params tuple in params_seq.

        Implementations should use a native batch API where available
        (e.g. asyncpg executemany, aiosqlite executemany) so the whole
        batch lands in one round-trip / transaction.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release all held resources."""


# ---------------------------------------------------------------------------
# PostgreSQL backend
# ---------------------------------------------------------------------------


class PostgresBackend(DatabaseBackend):
    """
    asyncpg-based backend.

    Reads connection details from environment variables:
      LOGGING_DB_HOST, LOGGING_DB_PORT, LOGGING_DB_NAME,
      LOGGING_DB_USER, LOGGING_DB_PASSWORD
      DB_POOL_MIN_SIZE, DB_POOL_MAX_SIZE, DB_POOL_TIMEOUT
    """

    name = "postgres"

    # asyncpg uses $1, $2, … placeholders.
    CHAT_LOGS_INSERT = """
        INSERT INTO chat_logs (
            timestamp, agent_name, session_id, user_id, endpoint,
            input_message, output_response, request_headers, model_info,
            token_usage, total_tokens, response_time_ms, status, error_message
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
    """
    ACTIVITY_LOG_INSERT = """
        INSERT INTO agent_activity_log (
            timestamp, agent_name, session_id, user_id, endpoint,
            chunk_sequence, chunk_content, chunk_text,
            serialization_warning, request_headers
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
    """

    def __init__(self) -> None:
        self._pool: Optional[AsyncpgPool] = None

    async def initialize(self, app_logger: Optional[logging.Logger]) -> bool:
        if not _ASYNCPG_AVAILABLE:
            if app_logger:
                app_logger.debug("asyncpg not installed — PostgreSQL backend unavailable.")
            return False

        host = os.environ.get("LOGGING_DB_HOST", "localhost")
        port = os.environ.get("LOGGING_DB_PORT", "5432")
        name = os.environ.get("LOGGING_DB_NAME", "agent_logs")
        user = os.environ.get("LOGGING_DB_USER", "postgres")
        password = os.environ.get("LOGGING_DB_PASSWORD", "postgres")
        min_size = int(os.environ.get("DB_POOL_MIN_SIZE", "2"))
        max_size = int(os.environ.get("DB_POOL_MAX_SIZE", "4"))
        timeout = int(os.environ.get("DB_POOL_TIMEOUT", "120"))

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

            await self._create_schema()

            if app_logger:
                app_logger.info(f"PostgreSQL backend ready: {host}:{port}/{name}")
            return True

        except (
            asyncpg.exceptions.InvalidPasswordError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.PostgresError,
            OSError,
        ) as exc:
            if app_logger:
                app_logger.warning(f"PostgreSQL backend unavailable: {exc}")
            await self._cleanup()
            return False
        except Exception as exc:  # pylint: disable=broad-except
            if app_logger:
                app_logger.error(f"Unexpected error initialising PostgreSQL backend: {exc}")
            await self._cleanup()
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._pool is None:
            raise RuntimeError("PostgresBackend.execute called before successful initialize()")
        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._pool is None:
            raise RuntimeError("PostgresBackend.execute_many called before successful initialize()")
        if not params_seq:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(query, params_seq)

    async def close(self) -> None:
        await self._cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _create_schema(self) -> None:
        chat_logs_ddl = """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id                SERIAL PRIMARY KEY,
                timestamp         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                agent_name        VARCHAR(255),
                session_id        VARCHAR(255),
                user_id           VARCHAR(255),
                endpoint          VARCHAR(50),
                input_message     JSONB,
                output_response   JSONB,
                request_headers   JSONB,
                model_info        JSONB,
                token_usage       JSONB,
                total_tokens      INT,
                response_time_ms  FLOAT,
                status            VARCHAR(50) DEFAULT 'success',
                error_message     TEXT,
                created_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        activity_log_ddl = """
            CREATE TABLE IF NOT EXISTS agent_activity_log (
                id                    SERIAL PRIMARY KEY,
                timestamp             TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                agent_name            VARCHAR(255),
                session_id            VARCHAR(255),
                user_id               VARCHAR(255),
                endpoint              VARCHAR(50),
                chunk_sequence        INT,
                chunk_content         JSONB,
                chunk_text            TEXT,
                serialization_warning TEXT,
                request_headers       JSONB,
                created_at            TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        migration_ddl = "ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS total_tokens INT"

        async with self._pool.acquire() as conn:
            await conn.execute(chat_logs_ddl)
            await conn.execute(activity_log_ddl)
            try:
                await conn.execute(migration_ddl)
            except Exception:  # pylint: disable=broad-except
                pass  # Column already exists on older Postgres without IF NOT EXISTS


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SQLiteBackend(DatabaseBackend):
    """
    aiosqlite-based backend. Opens a new connection per write (lightweight for
    low-volume fallback use). For high-throughput scenarios consider wrapping
    with a persistent connection or a write queue.

    Reads:
      SQLITE_DB_PATH – path to the database file (default: agent_logs.db)
    """

    name = "sqlite"

    # SQLite uses ? placeholders.
    CHAT_LOGS_INSERT = """
        INSERT INTO chat_logs (
            timestamp, agent_name, session_id, user_id, endpoint,
            input_message, output_response, request_headers, model_info,
            token_usage, total_tokens, response_time_ms, status, error_message
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    ACTIVITY_LOG_INSERT = """
        INSERT INTO agent_activity_log (
            timestamp, agent_name, session_id, user_id, endpoint,
            chunk_sequence, chunk_content, chunk_text,
            serialization_warning, request_headers
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """

    def __init__(self) -> None:
        self._db_path: Optional[str] = None

    async def initialize(self, app_logger: Optional[logging.Logger]) -> bool:
        if not _AIOSQLITE_AVAILABLE:
            if app_logger:
                app_logger.warning("aiosqlite not installed — SQLite backend unavailable.")
            return False

        self._db_path = self._resolve_db_path()
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        try:
            await self._create_schema()
            if app_logger:
                app_logger.info(f"SQLite backend ready: {self._db_path}")
            return True
        except Exception as exc:  # pylint: disable=broad-except
            if app_logger:
                app_logger.error(f"SQLite backend failed to initialise: {exc}")
            self._db_path = None
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend.execute called before successful initialize()")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(query, params)
            await db.commit()

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLiteBackend.execute_many called before successful initialize()")
        if not params_seq:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(query, params_seq)
            await db.commit()

    async def close(self) -> None:
        # aiosqlite connections are context-managed per operation; nothing to tear down.
        self._db_path = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_db_path() -> str:
        """Determine the SQLite file path from environment variables.

        Resolution order (first match wins):
          1. SQLITE_DB_PATH  – caller supplies the full path explicitly.
          2. SQLITE_DB_DIR   – caller supplies only the directory;
                               the filename defaults to ``agent_logs.db``.
          3. Fallback        – ``agent_logs.db`` in the current working directory.
        """
        full_path = os.environ.get("SQLITE_DB_PATH")
        if full_path:
            return full_path

        db_dir = os.environ.get("SQLITE_DB_DIR")
        if db_dir:
            return os.path.join(db_dir, "agent_logs.db")

        return "agent_logs.db"

    async def _create_schema(self) -> None:
        chat_logs_ddl = """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name       TEXT,
                session_id       TEXT,
                user_id          TEXT,
                endpoint         TEXT,
                input_message    TEXT,
                output_response  TEXT,
                request_headers  TEXT,
                model_info       TEXT,
                token_usage      TEXT,
                total_tokens     INTEGER,
                response_time_ms REAL,
                status           TEXT DEFAULT 'success',
                error_message    TEXT,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        activity_log_ddl = """
            CREATE TABLE IF NOT EXISTS agent_activity_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name            TEXT,
                session_id            TEXT,
                user_id               TEXT,
                endpoint              TEXT,
                chunk_sequence        INTEGER,
                chunk_content         TEXT,
                chunk_text            TEXT,
                serialization_warning TEXT,
                request_headers       TEXT,
                created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(chat_logs_ddl)
            await db.execute(activity_log_ddl)
            await db.commit()


# ---------------------------------------------------------------------------
# DatabaseLogger — backend-agnostic coordinator
# ---------------------------------------------------------------------------


class DatabaseLogger:
    """
    Asynchronous logger for agent chat interactions.

    Tries each backend in the supplied list until one initialises
    successfully, then uses it exclusively for all writes.

    Parameters
    ----------
    backends:
        Ordered list of DatabaseBackend instances to try. Defaults to
        [PostgresBackend(), SQLiteBackend()].
    logger:
        Standard library Logger instance; optional.
    """

    def __init__(
        self,
        backends: Optional[List[DatabaseBackend]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._backends: List[DatabaseBackend] = (
            [PostgresBackend(), SQLiteBackend()] if backends is None else backends
        )
        self.logger = logger
        self._backend: Optional[DatabaseBackend] = None
        self._db_logging_enabled = False
        self.is_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Try each backend in order; use the first that succeeds."""
        self._db_logging_enabled = (
            os.environ.get("DB_LOGGING_ENABLED", "false").lower() == "true"
        )

        if not self._db_logging_enabled:
            if self.logger:
                self.logger.info("Database logging is disabled (DB_LOGGING_ENABLED=false).")
            return

        for backend in self._backends:
            ok = await backend.initialize(self.logger)
            if ok:
                self._backend = backend
                self.is_active = True
                return

        if self.logger:
            self.logger.warning(
                "All database backends failed to initialise. Continuing without DB logging."
            )

    async def log_interaction(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        input_message: Any,
        output_response: Any,
        request_headers: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        token_usage: Optional[Dict] = None,
        response_time_ms: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """
        Persist a complete chat interaction.

        Silently skips when logging is disabled or no backend is active.
        """
        if not self._ready():
            return

        try:
            now = datetime.now(timezone.utc)
            input_json = self._to_json(input_message)
            output_json = self._to_json(output_response)
            headers_json = self._to_json(self._redact_headers(request_headers))
            model_json = self._to_json(model_info)
            usage_json = self._to_json(token_usage)
            total_tokens = self._extract_total_tokens(token_usage)

            query = self._backend.CHAT_LOGS_INSERT  # type: ignore[union-attr]
            params = (
                now, agent_name, session_id, user_id, endpoint,
                input_json, output_json, headers_json, model_json,
                usage_json, total_tokens, response_time_ms, status, error_message,
            )
            await self._backend.execute(query, params)  # type: ignore[union-attr]

            if self.logger:
                self.logger.debug(f"Logged interaction: {endpoint} — {session_id}")

        except Exception as exc:  # pylint: disable=broad-except
            if self.logger:
                self.logger.warning(f"Failed to log interaction: {exc}")

    async def log_stream_chunk(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        chunk_sequence: int,
        chunk_content: Any,
        chunk_text: Optional[str] = None,
        serialization_warning: Optional[str] = None,
        request_headers: Optional[Dict] = None,
    ) -> None:
        """Persist a single streaming chunk to agent_activity_log."""
        if not self._ready():
            return

        try:
            now = datetime.now(timezone.utc)
            content_json = self._to_json(chunk_content)
            headers_json = self._to_json(self._redact_headers(request_headers))

            query = self._backend.ACTIVITY_LOG_INSERT  # type: ignore[union-attr]
            params = (
                now, agent_name, session_id, user_id, endpoint,
                chunk_sequence, content_json, chunk_text,
                serialization_warning, headers_json,
            )
            await self._backend.execute(query, params)  # type: ignore[union-attr]

            if self.logger:
                self.logger.debug(
                    f"Logged stream chunk {chunk_sequence} for session {session_id}"
                )

        except Exception as exc:  # pylint: disable=broad-except
            if self.logger:
                self.logger.warning(f"Failed to log stream chunk: {exc}")

    async def log_stream_chunks_batch(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        endpoint: str,
        chunks: List[Dict[str, Any]],
        request_headers: Optional[Dict] = None,
    ) -> None:
        """Persist multiple streaming chunks in a single database round-trip.

        Each item in *chunks* is a dict with the following keys:

            chunk_sequence      int           (required)
            chunk_content       Any           (required)
            chunk_text          str | None    (optional)
            serialization_warning str | None  (optional)

        *agent_name*, *session_id*, *user_id*, *endpoint*, and
        *request_headers* are shared across all chunks in the batch and
        only need to be supplied once.

        Silently skips when logging is disabled or no backend is active.
        An empty *chunks* list is a no-op.
        """
        if not self._ready() or not chunks:
            return

        try:
            now = datetime.now(timezone.utc)
            headers_json = self._to_json(self._redact_headers(request_headers))
            query = self._backend.ACTIVITY_LOG_INSERT  # type: ignore[union-attr]

            params_seq = []
            for chunk in chunks:
                params_seq.append((
                    now,
                    agent_name,
                    session_id,
                    user_id,
                    endpoint,
                    chunk["chunk_sequence"],
                    self._to_json(chunk["chunk_content"]),
                    chunk.get("chunk_text"),
                    chunk.get("serialization_warning"),
                    headers_json,
                ))

            await self._backend.execute_many(query, params_seq)  # type: ignore[union-attr]

            if self.logger:
                self.logger.debug(
                    f"Logged batch of {len(chunks)} stream chunks for session {session_id}"
                )

        except Exception as exc:  # pylint: disable=broad-except
            if self.logger:
                self.logger.warning(f"Failed to log stream chunk batch: {exc}")

    async def close(self) -> None:
        """Release backend resources."""
        if self._backend is not None:
            await self._backend.close()
            self._backend = None
        self.is_active = False
        if self.logger:
            self.logger.info("Database connection closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ready(self) -> bool:
        return self._db_logging_enabled and self.is_active and self._backend is not None

    @staticmethod
    def _redact_headers(headers: Optional[Dict]) -> Optional[Dict]:
        """Return a copy of *headers* with sensitive keys replaced."""
        if not headers:
            return headers
        return {
            k: (_REDACTED_SENTINEL if k.lower() in _REDACTED_HEADERS else v)
            for k, v in headers.items()
        }

    @staticmethod
    def _extract_total_tokens(token_usage: Optional[Dict]) -> Optional[int]:
        if token_usage and isinstance(token_usage, dict):
            return token_usage.get("total_tokens") or token_usage.get("totalTokens")
        return None

    @staticmethod
    def _serialize_for_json(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
        """Recursively convert *obj* to a JSON-serialisable structure."""
        if depth > max_depth:
            return str(obj)
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [
                DatabaseLogger._serialize_for_json(item, depth + 1, max_depth)
                for item in obj
            ]
        if isinstance(obj, dict):
            return {
                str(k): DatabaseLogger._serialize_for_json(v, depth + 1, max_depth)
                for k, v in obj.items()
            }
        # Prefer Pydantic serialisation over raw __dict__ to respect validators/aliases.
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            try:
                return obj.model_dump()
            except Exception:  # pylint: disable=broad-except
                pass
        if hasattr(obj, "dict") and callable(obj.dict):
            try:
                return obj.dict()
            except Exception:  # pylint: disable=broad-except
                pass
        if hasattr(obj, "__dict__") and obj.__dict__:
            return {
                str(k): DatabaseLogger._serialize_for_json(v, depth + 1, max_depth)
                for k, v in obj.__dict__.items()
            }
        return str(obj)

    @classmethod
    def _to_json(cls, obj: Any) -> Optional[str]:
        """Serialise *obj* to a JSON string, or None if *obj* is None."""
        serialised = cls._serialize_for_json(obj)
        return json.dumps(serialised) if serialised is not None else None