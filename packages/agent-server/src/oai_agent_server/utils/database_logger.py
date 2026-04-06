"""
Database logger for tracking agent chat interactions.
"""

from __future__ import annotations

import json
import logging
import os
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
_REDACTED_HEADERS = frozenset(
    {"authorization", "cookie", "set-cookie", "x-api-key", "x-auth-token", "proxy-authorization"})
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
    ACTIVITY_LOG_INSERT = "INSERT INTO agent_activity_log (interaction_id, timestamp, agent_name, session_id, user_id, endpoint, chunk_sequence, chunk_content, chunk_text, serialization_warning, request_headers) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)"
    EVALUATION_LOG_INSERT = "INSERT INTO llm_judge_evaluations (interaction_id, quality_score, hallucination_detected, evaluation_data, timestamp) VALUES ($1, $2, $3, $4, $5)"
    PLACEHOLDER = "$"

    def __init__(self) -> None:
        self._pool: Optional[AsyncpgPool] = None

    async def initialize(self, logger: Optional[logging.Logger]) -> bool:
        if not _ASYNCPG_AVAILABLE:
            if logger: logger.debug("asyncpg not installed — PostgreSQL backend unavailable.")
            return False
        else:
            if logger: logger.info("PostgreSQL backend available.")

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
            self._pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, command_timeout=timeout,
                                                   timeout=5)
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            if logger: logger.info(f"PostgreSQL backend available: {host}:{port}/{name}")
            await self._create_schema(logger)
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

    async def _create_schema(self, logger: Optional[logging.Logger]) -> None:
        if logger: logger.info("Creating database schema...")
        chat_logs_ddl = "CREATE TABLE IF NOT EXISTS chat_logs (id SERIAL PRIMARY KEY, interaction_id VARCHAR(255) UNIQUE, timestamp TIMESTAMP WITH TIME ZONE, agent_name VARCHAR(255), session_id VARCHAR(255), user_id VARCHAR(255), endpoint VARCHAR(50), input_message JSONB, output_response JSONB, request_headers JSONB, model_info JSONB, token_usage JSONB, total_tokens INT, response_time_ms FLOAT, status VARCHAR(50), error_message TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);"
        activity_log_ddl = "CREATE TABLE IF NOT EXISTS agent_activity_log (id SERIAL PRIMARY KEY, interaction_id VARCHAR(255), timestamp TIMESTAMP WITH TIME ZONE, agent_name VARCHAR(255), session_id VARCHAR(255), user_id VARCHAR(255), endpoint VARCHAR(50), chunk_sequence INT, chunk_content JSONB, chunk_text TEXT, serialization_warning TEXT, request_headers JSONB, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);"
        evaluation_log_ddl = "CREATE TABLE IF NOT EXISTS llm_judge_evaluations (id SERIAL PRIMARY KEY, interaction_id VARCHAR(255) NOT NULL REFERENCES chat_logs(interaction_id) ON DELETE CASCADE, quality_score FLOAT, hallucination_detected BOOLEAN, evaluation_data JSONB, timestamp TIMESTAMP WITH TIME ZONE, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);"
        activity_log_index_ddl = "CREATE INDEX IF NOT EXISTS idx_agent_activity_log_interaction_id ON agent_activity_log(interaction_id);"

        async with self._pool.acquire() as conn:
            await conn.execute(chat_logs_ddl)
            await conn.execute(activity_log_ddl)
            await conn.execute(evaluation_log_ddl)
            await conn.execute(activity_log_index_ddl)
            try:
                await conn.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS total_tokens INT")
                await conn.execute("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS interaction_id VARCHAR(255) UNIQUE DEFAULT gen_random_uuid()")
                await conn.execute(
                    "ALTER TABLE agent_activity_log ADD COLUMN IF NOT EXISTS interaction_id VARCHAR(255)")
            except Exception as ex:
                if logger: logger.error(f"Schema creation failed {str(ex)}")


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"
    CHAT_LOGS_INSERT = "INSERT INTO chat_logs (interaction_id, timestamp, agent_name, session_id, user_id, endpoint, input_message, output_response, request_headers, model_info, token_usage, total_tokens, response_time_ms, status, error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    ACTIVITY_LOG_INSERT = "INSERT INTO agent_activity_log (interaction_id, timestamp, agent_name, session_id, user_id, endpoint, chunk_sequence, chunk_content, chunk_text, serialization_warning, request_headers) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
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
        chat_logs_ddl = "CREATE TABLE IF NOT EXISTS chat_logs (id INTEGER PRIMARY KEY, interaction_id TEXT UNIQUE, timestamp DATETIME, agent_name TEXT, session_id TEXT, user_id TEXT, endpoint TEXT, input_message TEXT, output_response TEXT, request_headers TEXT, model_info TEXT, token_usage TEXT, total_tokens INTEGER, response_time_ms REAL, status TEXT, error_message TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
        activity_log_ddl = "CREATE TABLE IF NOT EXISTS agent_activity_log (id INTEGER PRIMARY KEY, interaction_id TEXT, timestamp DATETIME, agent_name TEXT, session_id TEXT, user_id TEXT, endpoint TEXT, chunk_sequence INTEGER, chunk_content TEXT, chunk_text TEXT, serialization_warning TEXT, request_headers TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
        evaluation_log_ddl = "CREATE TABLE IF NOT EXISTS llm_judge_evaluations (id INTEGER PRIMARY KEY, interaction_id TEXT NOT NULL, quality_score REAL, hallucination_detected INTEGER, evaluation_data TEXT, timestamp DATETIME, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);"
        activity_log_index_ddl = "CREATE INDEX IF NOT EXISTS idx_agent_activity_log_interaction_id ON agent_activity_log(interaction_id);"

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(chat_logs_ddl)
            await db.execute(activity_log_ddl)
            await db.execute(evaluation_log_ddl)
            await db.execute(activity_log_index_ddl)
            try:
                await db.execute("ALTER TABLE chat_logs ADD COLUMN interaction_id TEXT")
                await db.execute("ALTER TABLE agent_activity_log ADD COLUMN interaction_id TEXT")
            except aiosqlite.OperationalError as e:
                if "duplicate column name" not in str(e): raise


class DatabaseLogger:
    def __init__(self, backends: Optional[List[DatabaseBackend]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
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
        self.is_active = False

    async def log_interaction(
            self,
            interaction_id: str,
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
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (
                interaction_id, now, agent_name, session_id, user_id,
                endpoint, self._to_json(input_message), self._to_json(output_response),
                self._to_json(self._redact_headers(request_headers)), self._to_json(model_info),
                self._to_json(token_usage), self._extract_total_tokens(token_usage),
                response_time_ms, status, error_message,
            )
            await self._backend.execute(self._backend.CHAT_LOGS_INSERT, params)
            if self.logger: self.logger.debug(f"Logged interaction: {interaction_id}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log interaction {interaction_id}: {exc}")
            raise

    async def log_llm_judge_evaluation(self, interaction_id: str, evaluation_data: Dict) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            quality_score = evaluation_data.get("quality_score", 0)
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

    async def get_chat_log_by_interaction_id(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        if not self._ready(): return None
        try:
            query = "SELECT * FROM chat_logs WHERE interaction_id = ?"
            if isinstance(self._backend, PostgresBackend):
                query = "SELECT * FROM chat_logs WHERE interaction_id = $1"
            row = await self._backend.fetch_one(query, (interaction_id,))
            return self._deserialize_chat_log_row(row) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve chat log for interaction {interaction_id}: {exc}")
            return None

    async def get_evaluation_by_interaction_id(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        if not self._ready(): return None
        try:
            query = "SELECT evaluation_data FROM llm_judge_evaluations WHERE interaction_id = ?"
            if isinstance(self._backend, PostgresBackend):
                query = "SELECT evaluation_data FROM llm_judge_evaluations WHERE interaction_id = $1"
            row = await self._backend.fetch_one(query, (interaction_id,))
            return self._parse_json_field(row.get("evaluation_data")) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve evaluation for interaction {interaction_id}: {exc}")
            return None

    async def log_stream_chunks_batch(self, **kwargs) -> None:
        if not self._ready() or not kwargs.get("chunks"): return
        try:
            now = datetime.now(timezone.utc)
            headers_json = self._to_json(self._redact_headers(kwargs.get("request_headers")))
            params_seq = [
                (
                    kwargs["interaction_id"], now, kwargs["agent_name"], kwargs["session_id"], kwargs["user_id"],
                    kwargs["endpoint"],
                    chunk["chunk_sequence"], self._to_json(chunk["chunk_content"]),
                    chunk.get("chunk_text"), chunk.get("serialization_warning"), headers_json,
                ) for chunk in kwargs["chunks"]
            ]
            await self._backend.execute_many(self._backend.ACTIVITY_LOG_INSERT, params_seq)
            if self.logger: self.logger.debug(
                f"Logged batch of {len(kwargs['chunks'])} stream chunks for session {kwargs['session_id']}")
        except Exception as exc:
            if self.logger: self.logger.warning(f"Failed to log stream chunk batch: {exc}")
            raise

    async def get_activity_logs(self, **kwargs) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            filters = [
                ("agent_name", kwargs.get("agent_name")), ("session_id", kwargs.get("session_id")),
                ("user_id", kwargs.get("user_id")), ("interaction_id", kwargs.get("interaction_id")),
            ]
            where, params = self._build_where_clause(filters, is_activity=True)
            ph = self._backend.PLACEHOLDER
            n = len(params)
            limit_ph, offset_ph = (f"${n + 1}", f"${n + 2}") if ph == "$" else ("?", "?")
            params = (*params, kwargs.get("limit", 100), kwargs.get("offset", 0))
            query = f"SELECT * FROM agent_activity_log al {where} ORDER BY timestamp DESC, chunk_sequence ASC LIMIT {limit_ph} OFFSET {offset_ph}"
            rows = await self._backend.fetch(query, params)
            return [self._deserialize_activity_log_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve activity logs: {exc}")
            return []

    async def get_stats(self, agent_name: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        if not self._ready(): return {}
        try:
            filters = [("agent_name", agent_name), ("user_id", user_id)]
            where, params = self._build_where_clause(filters)
            query = f"SELECT COUNT(*) AS total_interactions, COUNT(DISTINCT session_id) AS unique_sessions, COUNT(DISTINCT user_id) AS unique_users, AVG(response_time_ms) AS avg_response_time, MAX(response_time_ms) AS max_response_time, MIN(response_time_ms) AS min_response_time, SUM(total_tokens) AS total_tokens_sum, SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_interactions, SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_interactions FROM chat_logs cl {where}"
            row = await self._backend.fetch_one(query, params)
            if not row: return {}
            return {
                "total_interactions": int(row["total_interactions"] or 0),
                "unique_sessions": int(row["unique_sessions"] or 0),
                "unique_users": int(row["unique_users"] or 0),
                "avg_response_time_ms": float(row["avg_response_time"]) if row[
                                                                               "avg_response_time"] is not None else None,
                "max_response_time_ms": float(row["max_response_time"]) if row[
                                                                               "max_response_time"] is not None else None,
                "min_response_time_ms": float(row["min_response_time"]) if row[
                                                                               "min_response_time"] is not None else None,
                "total_tokens_sum": int(row["total_tokens_sum"] or 0),
                "successful_interactions": int(row["successful_interactions"] or 0),
                "failed_interactions": int(row["failed_interactions"] or 0),
            }
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to get stats: {exc}")
            return {}

    async def get_user_stats(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            filters = [("agent_name", agent_name)]
            where, params = self._build_where_clause(filters)
            query = f"SELECT user_id, COUNT(*) AS total_interactions, COUNT(DISTINCT session_id) AS unique_sessions, AVG(response_time_ms) AS avg_response_time, MAX(response_time_ms) AS max_response_time, MIN(response_time_ms) AS min_response_time, SUM(total_tokens) AS total_tokens_sum, SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_interactions, SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed_interactions FROM chat_logs cl {where} GROUP BY user_id ORDER BY total_interactions DESC"
            rows = await self._backend.fetch(query, params)
            return [
                {
                    "user_id": row["user_id"],
                    "total_interactions": int(row["total_interactions"] or 0),
                    "unique_sessions": int(row["unique_sessions"] or 0),
                    "avg_response_time_ms": float(row["avg_response_time"]) if row[
                                                                                   "avg_response_time"] is not None else None,
                    "max_response_time_ms": float(row["max_response_time"]) if row[
                                                                                   "max_response_time"] is not None else None,
                    "min_response_time_ms": float(row["min_response_time"]) if row[
                                                                                   "min_response_time"] is not None else None,
                    "total_tokens_sum": int(row["total_tokens_sum"] or 0),
                    "successful_interactions": int(row["successful_interactions"] or 0),
                    "failed_interactions": int(row["failed_interactions"] or 0),
                }
                for row in rows
            ]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to get user stats: {exc}")
            return []

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
        if headers is None: return None
        return {k: (_REDACTED_SENTINEL if k.lower() in _REDACTED_HEADERS else v) for k, v in headers.items()}

    @staticmethod
    def _extract_total_tokens(token_usage: Optional[Dict]) -> Optional[int]:
        if token_usage and isinstance(token_usage, dict):
            return token_usage.get("total_tokens") or token_usage.get("totalTokens")
        return None

    @staticmethod
    def _to_json(obj: Any) -> Optional[str]:
        """Serialise *obj* to a JSON string, or None if *obj* is None."""
        serialised = DatabaseLogger._serialize_for_json(obj)
        return json.dumps(serialised) if serialised is not None else None

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

    def _build_where_clause(self, filters: List[tuple], is_activity: bool = False) -> tuple:
        conditions, params = [], []
        prefix = "al" if is_activity else "cl"
        placeholder = self._backend.PLACEHOLDER
        param_idx = 1
        for col_expr, value in filters:
            if value is not None:
                op = "="
                if " " in col_expr:
                    col_expr, op = col_expr.split(None, 1)
                ph = f"${param_idx}" if placeholder == "$" else "?"
                param_idx += 1
                conditions.append(f"{prefix}.{col_expr} {op} {ph}")
                params.append(value)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        return where, tuple(params)

    def _deserialize_chat_log_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._parse_json_field(row.get("request_headers"))
        row["request_headers"] = self._redact_headers(headers) if isinstance(headers, dict) else headers
        for key in ["input_message", "output_response", "model_info", "token_usage", "evaluation_data"]:
            if key in row: row[key] = self._parse_json_field(row[key])
        for key in ["timestamp", "created_at"]:
            if key in row: row[key] = self._isoformat(row[key])
        return row

    def _deserialize_activity_log_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._parse_json_field(row.get("request_headers"))
        row["request_headers"] = self._redact_headers(headers) if isinstance(headers, dict) else headers
        row["chunk_content"] = self._parse_json_field(row.get("chunk_content"))
        for key in ["timestamp", "created_at"]:
            if key in row: row[key] = self._isoformat(row[key])
        return row

    @staticmethod
    def _parse_json_field(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None
