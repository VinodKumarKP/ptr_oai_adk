"""Persistent A2A TaskStore backed by Postgres or SQLite.

Implements the a2a-sdk ``TaskStore`` ABC by serialising the SDK's protobuf
``Task`` messages to JSON and persisting them in a single ``a2a_tasks`` table.
The store reuses the existing :class:`oai_agent_server.utils.database_logger.DatabaseBackend`
connection pools (``PostgresBackend`` / ``SQLiteBackend``) so the A2A task store
and chat-log writer share one underlying pool / connection.

Behaviour:

* Tasks are upserted by ID on ``save``.
* ``get`` returns ``None`` for missing or expired (``expires_at`` in the past)
  tasks.
* A background asyncio cleanup task purges expired rows on a configurable
  interval (default every 60s).
* TTL is configurable via the ``A2A_TASK_TTL_SECONDS`` environment variable
  (default 86400 seconds = 24 hours).

The SDK ships its own SQLAlchemy-based ``DatabaseTaskStore``; this module
deliberately avoids that dependency so the agent server keeps a single
DB-driver stack (``asyncpg`` + ``aiosqlite``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from a2a.server.context import ServerCallContext
    from a2a.server.tasks.task_store import TaskStore
    from a2a.types import a2a_pb2
    from a2a.types.a2a_pb2 import Task
    from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
    from a2a.utils.errors import InvalidParamsError
    from a2a.utils.task import decode_page_token, encode_page_token
    from google.protobuf.json_format import MessageToDict, ParseDict
    _A2A_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    TaskStore = object  # type: ignore[assignment,misc]
    ServerCallContext = object  # type: ignore[assignment,misc]
    Task = object  # type: ignore[assignment,misc]
    a2a_pb2 = None  # type: ignore[assignment]
    DEFAULT_LIST_TASKS_PAGE_SIZE = 50
    InvalidParamsError = Exception  # type: ignore[misc,assignment]
    MessageToDict = ParseDict = None  # type: ignore[assignment]
    decode_page_token = encode_page_token = None  # type: ignore[assignment]
    _A2A_SDK_AVAILABLE = False

from oai_agent_server.utils.database_logger import (
    DatabaseBackend,
    PostgresBackend,
    SQLiteBackend,
)


logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 86_400  # 24h
DEFAULT_CLEANUP_INTERVAL_SECONDS = 60


def _utcnow() -> datetime:
    """Return current UTC time as a tz-aware datetime."""
    return datetime.now(timezone.utc)


class DatabaseTaskStore(TaskStore):  # type: ignore[misc]
    """Persistent ``TaskStore`` backed by Postgres or SQLite.

    Args:
        backend: An initialised :class:`DatabaseBackend` (Postgres or SQLite).
            The backend's connection pool is reused; this class does not open
            its own pool.
        ttl_seconds: How long a task should persist before being considered
            expired. Defaults to ``A2A_TASK_TTL_SECONDS`` env (or 24h).
        cleanup_interval_seconds: How often the background cleanup task runs.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        ttl_seconds: Optional[int] = None,
        cleanup_interval_seconds: int = DEFAULT_CLEANUP_INTERVAL_SECONDS,
    ) -> None:
        if not _A2A_SDK_AVAILABLE:
            raise ImportError(
                "DatabaseTaskStore requires 'a2a-sdk[http-server]' to be installed."
            )
        self._backend: DatabaseBackend = backend
        env_ttl = os.environ.get("A2A_TASK_TTL_SECONDS")
        self._ttl = int(env_ttl) if env_ttl else (ttl_seconds or DEFAULT_TTL_SECONDS)
        self._cleanup_interval = cleanup_interval_seconds
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    def set_backend(self, backend: DatabaseBackend) -> None:
        """Replace the underlying backend (used when the backend is initialised lazily)."""
        self._backend = backend
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_postgres(self) -> bool:
        """True when the underlying backend is Postgres (uses ``$N`` placeholders)."""
        return isinstance(self._backend, PostgresBackend)

    def _ph(self, n: int) -> str:
        """Return the placeholder for parameter index *n* (1-based) for the backend."""
        return f"${n}" if self.is_postgres else "?"

    async def initialize(self) -> None:
        """Create the ``a2a_tasks`` table and start the cleanup task.

        Idempotent — safe to call multiple times.
        """
        async with self._init_lock:
            if self._initialized:
                return
            await self._create_schema()
            self._initialized = True
            if self._cleanup_task is None or self._cleanup_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self._cleanup_task = loop.create_task(self._cleanup_loop())
                except RuntimeError:
                    # No running loop (e.g. unit tests); cleanup will be skipped.
                    self._cleanup_task = None
            logger.info(
                "DatabaseTaskStore initialised (backend=%s, ttl=%ss).",
                self._backend.name, self._ttl,
            )

    async def shutdown(self) -> None:
        """Cancel the background cleanup task. Safe to call multiple times."""
        task = self._cleanup_task
        self._cleanup_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _create_schema(self) -> None:
        """Create the ``a2a_tasks`` table for the active backend."""
        if self.is_postgres:
            ddl_table = (
                "CREATE TABLE IF NOT EXISTS a2a_tasks ("
                "  task_id     TEXT PRIMARY KEY,"
                "  owner       TEXT NOT NULL DEFAULT '',"
                "  context_id  TEXT,"
                "  status      TEXT,"
                "  status_ts   TIMESTAMPTZ,"
                "  task_data   JSONB NOT NULL,"
                "  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
                "  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
                "  expires_at  TIMESTAMPTZ"
                ")"
            )
            ddl_idx_ctx = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_context "
                "ON a2a_tasks(context_id)"
            )
            ddl_idx_owner = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_owner "
                "ON a2a_tasks(owner)"
            )
            ddl_idx_exp = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_expires "
                "ON a2a_tasks(expires_at) WHERE expires_at IS NOT NULL"
            )
        else:
            ddl_table = (
                "CREATE TABLE IF NOT EXISTS a2a_tasks ("
                "  task_id     TEXT PRIMARY KEY,"
                "  owner       TEXT NOT NULL DEFAULT '',"
                "  context_id  TEXT,"
                "  status      TEXT,"
                "  status_ts   TEXT,"
                "  task_data   TEXT NOT NULL,"
                "  created_at  TEXT NOT NULL DEFAULT (datetime('now')),"
                "  updated_at  TEXT NOT NULL DEFAULT (datetime('now')),"
                "  expires_at  TEXT"
                ")"
            )
            ddl_idx_ctx = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_context "
                "ON a2a_tasks(context_id)"
            )
            ddl_idx_owner = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_owner "
                "ON a2a_tasks(owner)"
            )
            ddl_idx_exp = (
                "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_expires "
                "ON a2a_tasks(expires_at)"
            )
        await self._backend.execute(ddl_table, ())
        await self._backend.execute(ddl_idx_ctx, ())
        await self._backend.execute(ddl_idx_owner, ())
        await self._backend.execute(ddl_idx_exp, ())

    # ------------------------------------------------------------------
    # Cleanup loop
    # ------------------------------------------------------------------
    async def _cleanup_loop(self) -> None:
        """Periodically delete expired rows. Runs until cancelled."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._delete_expired()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("a2a_tasks cleanup iteration failed: %s", exc)

    async def _delete_expired(self) -> None:
        """Hard-delete any rows whose ``expires_at`` is in the past."""
        if self.is_postgres:
            query = "DELETE FROM a2a_tasks WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            await self._backend.execute(query, ())
        else:
            query = (
                "DELETE FROM a2a_tasks WHERE expires_at IS NOT NULL "
                "AND expires_at < ?"
            )
            await self._backend.execute(query, (_utcnow().isoformat(),))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_owner(context: "ServerCallContext") -> str:
        """Return the owner string for the given call context, defaulting to ''."""
        try:
            user = getattr(context, "user", None)
            name = getattr(user, "user_name", "") if user is not None else ""
            return name or ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _serialize_task(task: "Task") -> str:
        """Serialise a protobuf ``Task`` to a JSON string."""
        return json.dumps(MessageToDict(task, preserving_proto_field_name=True))

    @staticmethod
    def _deserialize_task(payload: object) -> "Task":
        """Deserialise a stored row's ``task_data`` (str or dict) back into a Task."""
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
        task = Task()
        ParseDict(payload, task, ignore_unknown_fields=True)
        return task

    def _expires_at(self) -> datetime:
        return _utcnow() + timedelta(seconds=self._ttl)

    # ------------------------------------------------------------------
    # TaskStore ABC
    # ------------------------------------------------------------------
    async def save(self, task: "Task", context: "ServerCallContext") -> None:
        """Upsert *task* in the store, refreshing its TTL."""
        await self._ensure_initialized()
        owner = self._resolve_owner(context)
        task_data = self._serialize_task(task)
        context_id = getattr(task, "context_id", "") or None
        status_state = ""
        status_ts: Optional[datetime] = None
        try:
            if task.HasField("status"):
                status_state = a2a_pb2.TaskState.Name(task.status.state)
                if task.status.HasField("timestamp"):
                    status_ts = task.status.timestamp.ToDatetime().replace(
                        tzinfo=timezone.utc
                    )
        except Exception:  # noqa: BLE001
            pass

        now = _utcnow()
        expires = self._expires_at()

        if self.is_postgres:
            query = (
                "INSERT INTO a2a_tasks (task_id, owner, context_id, status, "
                "status_ts, task_data, created_at, updated_at, expires_at) "
                "VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9) "
                "ON CONFLICT (task_id) DO UPDATE SET "
                "owner = EXCLUDED.owner, "
                "context_id = EXCLUDED.context_id, "
                "status = EXCLUDED.status, "
                "status_ts = EXCLUDED.status_ts, "
                "task_data = EXCLUDED.task_data, "
                "updated_at = EXCLUDED.updated_at, "
                "expires_at = EXCLUDED.expires_at"
            )
            params = (
                task.id, owner, context_id, status_state, status_ts,
                task_data, now, now, expires,
            )
        else:
            query = (
                "INSERT OR REPLACE INTO a2a_tasks (task_id, owner, context_id, "
                "status, status_ts, task_data, created_at, updated_at, expires_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)"
            )
            params = (
                task.id, owner, context_id, status_state,
                status_ts.isoformat() if status_ts else None,
                task_data, now.isoformat(), now.isoformat(), expires.isoformat(),
            )
        await self._backend.execute(query, params)

    async def get(
        self, task_id: str, context: "ServerCallContext"
    ) -> Optional["Task"]:
        """Return the stored task or ``None`` if missing/expired/not owned."""
        await self._ensure_initialized()
        owner = self._resolve_owner(context)
        if self.is_postgres:
            query = (
                "SELECT task_data, expires_at FROM a2a_tasks "
                "WHERE task_id = $1 AND owner = $2"
            )
        else:
            query = (
                "SELECT task_data, expires_at FROM a2a_tasks "
                "WHERE task_id = ? AND owner = ?"
            )
        row = await self._backend.fetch_one(query, (task_id, owner))
        if not row:
            return None
        if self._is_expired(row.get("expires_at")):
            return None
        return self._deserialize_task(row["task_data"])

    async def delete(self, task_id: str, context: "ServerCallContext") -> None:
        """Delete *task_id* for the resolved owner. No-op if missing."""
        await self._ensure_initialized()
        owner = self._resolve_owner(context)
        if self.is_postgres:
            query = "DELETE FROM a2a_tasks WHERE task_id = $1 AND owner = $2"
        else:
            query = "DELETE FROM a2a_tasks WHERE task_id = ? AND owner = ?"
        await self._backend.execute(query, (task_id, owner))

    async def list(
        self,
        params: "a2a_pb2.ListTasksRequest",
        context: "ServerCallContext",
    ) -> "a2a_pb2.ListTasksResponse":
        """Return tasks for the resolved owner, filtered by ``params``.

        Supports filtering by ``context_id`` and ``status`` and basic
        ``page_token`` / ``page_size`` pagination ordered by status timestamp
        descending then ``task_id`` descending.
        """
        await self._ensure_initialized()
        owner = self._resolve_owner(context)

        conditions = []
        values: list = []
        idx = 1

        def ph() -> str:
            nonlocal idx
            p = self._ph(idx)
            idx += 1
            return p

        conditions.append(f"owner = {ph()}")
        values.append(owner)
        # Filter out expired rows
        if self.is_postgres:
            conditions.append("(expires_at IS NULL OR expires_at >= NOW())")
        else:
            conditions.append(f"(expires_at IS NULL OR expires_at >= {ph()})")
            values.append(_utcnow().isoformat())

        if params.context_id:
            conditions.append(f"context_id = {ph()}")
            values.append(params.context_id)
        if params.status:
            conditions.append(f"status = {ph()}")
            values.append(a2a_pb2.TaskState.Name(params.status))
        if params.HasField("status_timestamp_after"):
            after = params.status_timestamp_after.ToDatetime().replace(
                tzinfo=timezone.utc
            )
            conditions.append(f"status_ts >= {ph()}")
            values.append(after if self.is_postgres else after.isoformat())

        where = " WHERE " + " AND ".join(conditions)

        # Total count
        count_query = f"SELECT COUNT(*) AS c FROM a2a_tasks{where}"
        count_row = await self._backend.fetch_one(count_query, tuple(values))
        total_count = int(count_row["c"]) if count_row else 0

        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        # Pagination via page_token (a task_id from a previous page boundary).
        order_clause = " ORDER BY status_ts DESC NULLS LAST, task_id DESC" if self.is_postgres \
            else " ORDER BY status_ts DESC, task_id DESC"
        if params.page_token:
            start_id = decode_page_token(params.page_token)
            conditions_pag = list(conditions)
            values_pag = list(values)
            conditions_pag.append(f"task_id <= {ph()}")
            values_pag.append(start_id)
            where_pag = " WHERE " + " AND ".join(conditions_pag)
            list_query = (
                f"SELECT task_data FROM a2a_tasks{where_pag}{order_clause} "
                f"LIMIT {ph()}"
            )
            values_pag.append(page_size + 1)
            rows = await self._backend.fetch(list_query, tuple(values_pag))
        else:
            list_query = (
                f"SELECT task_data FROM a2a_tasks{where}{order_clause} "
                f"LIMIT {ph()}"
            )
            values.append(page_size + 1)
            rows = await self._backend.fetch(list_query, tuple(values))

        tasks = [self._deserialize_task(r["task_data"]) for r in rows]
        next_token = (
            encode_page_token(tasks[-1].id) if len(tasks) == page_size + 1 else None
        )
        return a2a_pb2.ListTasksResponse(
            tasks=tasks[:page_size],
            total_size=total_count,
            next_page_token=next_token or "",
            page_size=page_size,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _is_expired(expires_at: object) -> bool:
        """Return True if *expires_at* (datetime or ISO str) is in the past."""
        if expires_at is None:
            return False
        if isinstance(expires_at, str):
            try:
                expires_at = datetime.fromisoformat(expires_at)
            except ValueError:
                return False
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at < _utcnow()
        return False


class _LazyTaskStoreProxy(TaskStore):  # type: ignore[misc]
    """A ``TaskStore`` that defers construction of its delegate until first use.

    Useful when the A2A router must be wired up before the database backend is
    initialised. Call :meth:`bind` once the real store is ready; until then,
    method calls block on an asyncio Event.
    """

    def __init__(self) -> None:
        if not _A2A_SDK_AVAILABLE:
            raise ImportError(
                "_LazyTaskStoreProxy requires 'a2a-sdk[http-server]' to be installed."
            )
        self._delegate: Optional[TaskStore] = None
        self._ready = asyncio.Event()

    def bind(self, delegate: "TaskStore") -> None:
        """Set the underlying delegate. Subsequent calls proxy to it."""
        self._delegate = delegate
        self._ready.set()

    async def _wait(self) -> "TaskStore":
        if self._delegate is None:
            await self._ready.wait()
        assert self._delegate is not None
        return self._delegate

    async def save(self, task, context):  # type: ignore[override]
        """Delegate ``save`` to the bound store."""
        store = await self._wait()
        return await store.save(task, context)

    async def get(self, task_id, context):  # type: ignore[override]
        """Delegate ``get`` to the bound store."""
        store = await self._wait()
        return await store.get(task_id, context)

    async def list(self, params, context):  # type: ignore[override]
        """Delegate ``list`` to the bound store."""
        store = await self._wait()
        return await store.list(params, context)

    async def delete(self, task_id, context):  # type: ignore[override]
        """Delegate ``delete`` to the bound store."""
        store = await self._wait()
        return await store.delete(task_id, context)


async def build_task_store_from_env(
    db_logger,
    logger_: Optional[logging.Logger] = None,
):
    """Construct an A2A task store based on environment configuration.

    Returns either a :class:`DatabaseTaskStore` (default) or the SDK's
    ``InMemoryTaskStore`` when ``A2A_TASK_STORE=memory``. When the database
    backend cannot be initialised we fall back to in-memory and log a warning,
    so the server still starts.

    Args:
        db_logger: An initialised :class:`DatabaseLogger`. Its active backend
            is reused. May be ``None`` or inactive.
        logger_: Optional logger for diagnostics.
    """
    log = logger_ or logger
    mode = os.environ.get("A2A_TASK_STORE", "database").strip().lower()
    if mode == "memory":
        from a2a.server.tasks import InMemoryTaskStore  # local import
        log.info("A2A task store: using InMemoryTaskStore (A2A_TASK_STORE=memory).")
        return InMemoryTaskStore()

    backend: Optional[DatabaseBackend] = None
    if db_logger is not None and getattr(db_logger, "_backend", None) is not None:
        backend = db_logger._backend  # noqa: SLF001 - intentional reuse
    else:
        # Stand-alone backend selection (mirrors DatabaseLogger logic).
        db_type = os.environ.get("DB_LOGGING_BACKEND", "").strip().lower()
        candidates: list[DatabaseBackend] = []
        if db_type == "postgres":
            candidates = [PostgresBackend(), SQLiteBackend()]
        elif db_type == "sqlite":
            candidates = [SQLiteBackend()]
        else:
            candidates = [PostgresBackend(), SQLiteBackend()]
        for cand in candidates:
            if await cand.initialize(log):
                backend = cand
                break

    if backend is None:
        from a2a.server.tasks import InMemoryTaskStore  # local import
        log.warning(
            "A2A task store: no database backend available; "
            "falling back to InMemoryTaskStore."
        )
        return InMemoryTaskStore()

    store = DatabaseTaskStore(backend=backend)
    await store.initialize()
    return store
