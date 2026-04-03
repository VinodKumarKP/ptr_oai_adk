"""
Database logger for tracking agent chat interactions.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Optional dependency flags
try:
    import asyncpg
    from asyncpg.pool import Pool as AsyncpgPool
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False
    AsyncpgPool = None

try:
    import aiosqlite
    _AIOSQLITE_AVAILABLE = True
except ImportError:
    _AIOSQLITE_AVAILABLE = False

# Sensitive header keys
_REDACTED_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "x-api-key", "x-auth-token", "proxy-authorization"})
_REDACTED_SENTINEL = "***REDACTED***"


class DatabaseBackend(ABC):
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


class PostgresBackend(DatabaseBackend):
    name = "postgres"
    CHAT_LOGS_INSERT = "INSERT INTO chat_logs (interaction_id, timestamp, agent_name, session_id, user_id, endpoint, input_message, output_response, request_headers, model_info, token_usage, total_tokens, response_time_ms, status, error_message) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)"
    ACTIVITY_LOG_INSERT = "INSERT INTO agent_activity_log (timestamp, agent_name, session_id, user_id, endpoint, chunk_sequence, chunk_content, chunk_text, serialization_warning, request_headers) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)"
    EVALUATION_LOG_INSERT = "INSERT INTO llm_judge_evaluations (interaction_id, quality_score, hallucination_detected, evaluation_data, timestamp) VALUES ($1, $2, $3, $4, $5)"
    PLACEHOLDER = "$"

    def __init__(self) -> None:
        self._pool: Optional[AsyncpgPool] = None

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _ASYNCPG_AVAILABLE:
            if logger: logger.debug("asyncpg not installed — PostgreSQL backend unavailable.")
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
            self._pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, command_timeout=timeout, timeout=5)
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            await self._create_schema()
            if logger: logger.info(f"PostgreSQL backend ready: {host}:{port}/{name}")
            return True
        except Exception as exc:
            if logger: logger.warning(f"PostgreSQL backend unavailable: {exc}")
            await self._cleanup()
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._pool is None: raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._pool is None: raise RuntimeError("PostgresBackend not initialized")
        if not params_seq: return
        async with self._pool.acquire() as conn:
            await conn.executemany(query, params_seq)

    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if self._pool is None: raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]:
        if self._pool is None: raise RuntimeError("PostgresBackend not initialized")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        return dict(row) if row else None

    async def close(self) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _create_schema(self) -> None:
        chat_logs_ddl = """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id SERIAL PRIMARY KEY, interaction_id VARCHAR(255) UNIQUE, timestamp TIMESTAMP WITH TIME ZONE,
                agent_name VARCHAR(255), session_id VARCHAR(255), user_id VARCHAR(255), endpoint VARCHAR(50),
                input_message JSONB, output_response JSONB, request_headers JSONB, model_info JSONB,
                token_usage JSONB, total_tokens INT, response_time_ms FLOAT, status VARCHAR(50),
                error_message TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        activity_log_ddl = "CREATE TABLE IF NOT EXISTS agent_activity_log (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE, agent_name VARCHAR(255), session_id VARCHAR(255), user_id VARCHAR(255), endpoint VARCHAR(50), chunk_sequence INT, chunk_content JSONB, chunk_text TEXT, serialization_warning TEXT, request_headers JSONB, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);"
        evaluation_log_ddl = """
            CREATE TABLE IF NOT EXISTS llm_judge_evaluations (
                id SERIAL PRIMARY KEY, interaction_id VARCHAR(255) NOT NULL REFERENCES chat_logs(interaction_id) ON DELETE CASCADE,
                quality_score FLOAT, hallucination_detected BOOLEAN, evaluation_data JSONB,
                timestamp TIMESTAMP WITH TIME ZONE, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        async with self._pool.acquire() as conn:
            await conn.execute(chat_logs_ddl)
            await conn.execute(activity_log_ddl)
            await conn.execute(evaluation_log_ddl)
            try:
                await conn.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS total_tokens INT")
                await conn.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS interaction_id VARCHAR(255)")
            except Exception:
                pass


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"
    CHAT_LOGS_INSERT = "INSERT INTO chat_logs (interaction_id, timestamp, agent_name, session_id, user_id, endpoint, input_message, output_response, request_headers, model_info, token_usage, total_tokens, response_time_ms, status, error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    ACTIVITY_LOG_INSERT = "INSERT INTO agent_activity_log (timestamp, agent_name, session_id, user_id, endpoint, chunk_sequence, chunk_content, chunk_text, serialization_warning, request_headers) VALUES (?,?,?,?,?,?,?,?,?,?)"
    EVALUATION_LOG_INSERT = "INSERT INTO llm_judge_evaluations (interaction_id, quality_score, hallucination_detected, evaluation_data, timestamp) VALUES (?, ?, ?, ?, ?)"
    PLACEHOLDER = "?"

    def __init__(self) -> None:
        self._db_path: Optional[str] = None

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _AIOSQLITE_AVAILABLE:
            if logger: logger.warning("aiosqlite not installed — SQLite backend unavailable.")
            return False
        self._db_path = self._resolve_db_path()
        db_dir = os.path.dirname(self._db_path)
        if db_dir: os.makedirs(db_dir, exist_ok=True)
        try:
            await self._create_schema()
            if logger: logger.info(f"SQLite backend ready: {self._db_path}")
            return True
        except Exception as exc:
            if logger: logger.error(f"SQLite backend failed to initialise: {exc}")
            self._db_path = None
            return False

    async def execute(self, query: str, params: tuple) -> None:
        if self._db_path is None: raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(query, params)
            await db.commit()

    async def execute_many(self, query: str, params_seq: List[tuple]) -> None:
        if self._db_path is None: raise RuntimeError("SQLiteBackend not initialized")
        if not params_seq: return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executemany(query, params_seq)
            await db.commit()

    async def fetch(self, query: str, params: tuple) -> List[Dict[str, Any]]:
        if self._db_path is None: raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple) -> Optional[Dict[str, Any]]:
        if self._db_path is None: raise RuntimeError("SQLiteBackend not initialized")
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
        return dict(row) if row else None

    async def close(self) -> None:
        self._db_path = None

    @staticmethod
    def _resolve_db_path() -> str:
        full_path = os.environ.get("SQLITE_DB_PATH")
        if full_path: return full_path
        db_dir = os.environ.get("SQLITE_DB_DIR")
        if db_dir: return os.path.join(db_dir, "agent_logs.db")
        return "agent_logs.db"

    async def _create_schema(self) -> None:
        chat_logs_ddl = """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY, interaction_id TEXT UNIQUE, timestamp DATETIME, agent_name TEXT,
                session_id TEXT, user_id TEXT, endpoint TEXT, input_message TEXT, output_response TEXT,
                request_headers TEXT, model_info TEXT, token_usage TEXT, total_tokens INTEGER,
                response_time_ms REAL, status TEXT, error_message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        activity_log_ddl = "CREATE TABLE IF NOT EXISTS agent_activity_log (id INTEGER PRIMARY KEY, timestamp DATETIME, agent_name TEXT, session_id TEXT, user_id TEXT, endpoint TEXT, chunk_sequence INTEGER, chunk_content TEXT, chunk_text TEXT, serialization_warning TEXT, request_headers TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
        evaluation_log_ddl = """
            CREATE TABLE IF NOT EXISTS llm_judge_evaluations (
                id INTEGER PRIMARY KEY, interaction_id TEXT NOT NULL, quality_score REAL,
                hallucination_detected INTEGER, evaluation_data TEXT, timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES chat_logs (interaction_id) ON DELETE CASCADE
            );
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(chat_logs_ddl)
            await db.execute(activity_log_ddl)
            await db.execute(evaluation_log_ddl)
            try:
                await db.execute("ALTER TABLE chat_logs ADD COLUMN interaction_id TEXT")
            except aiosqlite.OperationalError as e:
                if "duplicate column name" not in str(e): raise


class DatabaseLogger:
    def __init__(self, backends: Optional[List[DatabaseBackend]] = None, logger: Optional[logging.Logger] = None) -> None:
        self._backends: List[DatabaseBackend] = backends or [PostgresBackend(), SQLiteBackend()]
        self.logger = logger
        self._backend: Optional[DatabaseBackend] = None
        self._db_logging_enabled = False
        self.is_active = False

    async def initialize(self) -> None:
        self._db_logging_enabled = os.environ.get("DB_LOGGING_ENABLED", "false").lower() == "true"
        if not self._db_logging_enabled:
            if self.logger: self.logger.info("Database logging is disabled.")
            return
        for backend in self._backends:
            if await backend.initialize(self.logger):
                self._backend = backend
                self.is_active = True
                return
        if self.logger: self.logger.warning("All database backends failed to initialise.")

    async def log_interaction(self, **kwargs) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (
                kwargs["interaction_id"], now, kwargs["agent_name"], kwargs["session_id"], kwargs["user_id"],
                kwargs["endpoint"], self._to_json(kwargs["input_message"]), self._to_json(kwargs["output_response"]),
                self._to_json(self._redact_headers(kwargs.get("request_headers"))), self._to_json(kwargs.get("model_info")),
                self._to_json(kwargs.get("token_usage")), self._extract_total_tokens(kwargs.get("token_usage")),
                kwargs.get("response_time_ms"), kwargs.get("status", "success"), kwargs.get("error_message"),
            )
            await self._backend.execute(self._backend.CHAT_LOGS_INSERT, params)
            if self.logger: self.logger.debug(f"Logged interaction: {kwargs['interaction_id']}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log interaction {kwargs['interaction_id']}: {exc}")
            raise

    async def log_llm_judge_evaluation(self, interaction_id: str, evaluation_data: Dict) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            quality_score = evaluation_data.get("quality_score")
            hallucination_detected = str(evaluation_data.get("hallucination_detected", "false")).lower() == "true"
            params = (
                interaction_id, float(quality_score) if quality_score is not None else None,
                hallucination_detected, self._to_json(evaluation_data), now,
            )
            await self._backend.execute(self._backend.EVALUATION_LOG_INSERT, params)
            if self.logger: self.logger.debug(f"Logged LLM evaluation for interaction: {interaction_id}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log LLM evaluation for {interaction_id}: {exc}")
            raise

    async def log_stream_chunks_batch(self, **kwargs) -> None:
        if not self._ready() or not kwargs.get("chunks"): return
        try:
            now = datetime.now(timezone.utc)
            headers_json = self._to_json(self._redact_headers(kwargs.get("request_headers")))
            params_seq = [
                (
                    now, kwargs["agent_name"], kwargs["session_id"], kwargs["user_id"], kwargs["endpoint"],
                    chunk["chunk_sequence"], self._to_json(chunk["chunk_content"]),
                    chunk.get("chunk_text"), chunk.get("serialization_warning"), headers_json,
                ) for chunk in kwargs["chunks"]
            ]
            await self._backend.execute_many(self._backend.ACTIVITY_LOG_INSERT, params_seq)
            if self.logger: self.logger.debug(f"Logged batch of {len(kwargs['chunks'])} stream chunks for session {kwargs['session_id']}")
        except Exception as exc:
            if self.logger: self.logger.warning(f"Failed to log stream chunk batch: {exc}")
            raise

    async def get_logs(self, **kwargs) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            filters = [
                ("agent_name", kwargs.get("agent_name")), ("session_id", kwargs.get("session_id")),
                ("user_id", kwargs.get("user_id")), ("endpoint", kwargs.get("endpoint")),
                ("timestamp >=", kwargs.get("start_date")), ("timestamp <=", kwargs.get("end_date")),
                ("status", kwargs.get("status")),
            ]
            where, params = self._build_where_clause(filters)
            ph = self._backend.PLACEHOLDER
            n = len(params)
            limit_ph, offset_ph = (f"${n + 1}", f"${n + 2}") if ph == "$" else ("?", "?")
            params = (*params, kwargs.get("limit", 100), kwargs.get("offset", 0))
            query = f"SELECT cl.*, lje.evaluation_data FROM chat_logs cl LEFT JOIN llm_judge_evaluations lje ON cl.interaction_id = lje.interaction_id {where} ORDER BY cl.timestamp DESC LIMIT {limit_ph} OFFSET {offset_ph}"
            rows = await self._backend.fetch(query, params)
            return [self._deserialize_chat_log_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve logs: {exc}")
            return []

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
            self._backend = None
        self.is_active = False
        if self.logger: self.logger.info("Database connection closed.")

    def _ready(self) -> bool:
        return self._db_logging_enabled and self.is_active and self._backend is not None

    @staticmethod
    def _redact_headers(headers: Optional[Dict]) -> Optional[Dict]:
        if not headers: return None
        return {k: (_REDACTED_SENTINEL if k.lower() in _REDACTED_HEADERS else v) for k, v in headers.items()}

    @staticmethod
    def _extract_total_tokens(token_usage: Optional[Dict]) -> Optional[int]:
        if token_usage and isinstance(token_usage, dict):
            return token_usage.get("total_tokens") or token_usage.get("totalTokens")
        return None

    @staticmethod
    def _to_json(obj: Any) -> Optional[str]:
        if obj is None: return None
        return json.dumps(obj)

    def _build_where_clause(self, filters: List[tuple]) -> tuple:
        conditions, params = [], []
        placeholder = self._backend.PLACEHOLDER
        for i, (col_expr, value) in enumerate(filters):
            if value is not None:
                op = "="
                if " " in col_expr:
                    col_expr, op = col_expr.split(None, 1)
                ph = f"${i + 1}" if placeholder == "$" else "?"
                conditions.append(f"cl.{col_expr} {op} {ph}")
                params.append(value)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where, tuple(params)

    def _deserialize_chat_log_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["input_message", "output_response", "request_headers", "model_info", "token_usage", "evaluation_data"]:
            if key in row: row[key] = self._parse_json_field(row[key])
        for key in ["timestamp", "created_at"]:
            if key in row: row[key] = self._isoformat(row[key])
        return row

    @staticmethod
    def _parse_json_field(value: Any) -> Any:
        if isinstance(value, str):
            try: return json.loads(value)
            except (json.JSONDecodeError, TypeError): return value
        return value

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None
