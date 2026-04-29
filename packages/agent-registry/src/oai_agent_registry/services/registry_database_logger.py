"""
Database logger for tracking agent registry events.
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

    AGENT_REGISTRY_UPSERT = "INSERT INTO agent_registry (agent_name, endpoint_url, port, git_source_url, active, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (agent_name) DO UPDATE SET endpoint_url = EXCLUDED.endpoint_url, port = EXCLUDED.port, git_source_url = EXCLUDED.git_source_url, active = EXCLUDED.active, updated_at = EXCLUDED.updated_at"
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = FALSE, updated_at = $2 WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"

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
        if logger: logger.info("Creating agent registry database schema...")
        agent_registry_ddl = """
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_name VARCHAR(255) PRIMARY KEY,
                endpoint_url VARCHAR(255),
                port INTEGER,
                git_source_url VARCHAR(255),
                active BOOLEAN,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        async with self._pool.acquire() as conn:
            await conn.execute(agent_registry_ddl)
            if logger: logger.info("Agent registry database schema created/updated.")


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"

    AGENT_REGISTRY_UPSERT = "INSERT INTO agent_registry (agent_name, endpoint_url, port, git_source_url, active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(agent_name) DO UPDATE SET endpoint_url = EXCLUDED.endpoint_url, port = EXCLUDED.port, git_source_url = EXCLUDED.git_source_url, active = EXCLUDED.active, updated_at = EXCLUDED.updated_at"
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = 0, updated_at = ? WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"

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
        if db_dir: return os.path.join(db_dir, "agent_registry.db")
        return "agent_registry.db"

    async def _create_schema(self) -> None:
        agent_registry_ddl = """
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_name TEXT PRIMARY KEY,
                endpoint_url TEXT,
                port INTEGER,
                git_source_url TEXT,
                active BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(agent_registry_ddl)


class RegistryDatabaseLogger:
    def __init__(self, backends: Optional[List[DatabaseBackend]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self._backends: List[DatabaseBackend] = backends or [PostgresBackend(), SQLiteBackend()]
        self.logger = logger
        self._backend: Optional[DatabaseBackend] = None
        self._db_logging_enabled = False
        self.is_active = False

    async def initialize(self) -> None:
        self._db_logging_enabled = os.environ.get("REGISTRY_DB_LOGGING_ENABLED", "true").lower() == "true"
        if not self._db_logging_enabled:
            if self.logger: self.logger.info("Registry database logging is disabled.")
            return
        for backend in self._backends:
            if await backend.initialize(self.logger):
                self._backend = backend
                self.is_active = True
                return
        if self.logger: self.logger.warning("All registry database backends failed to initialise.")
        self.is_active = False

    async def log_agent_registration(
            self,
            agent_name: str,
            endpoint_url: str,
            port: int,
            git_source_url: str,
            active: bool = True,
    ) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (
                agent_name, endpoint_url, port, git_source_url, active, now, now,
            )
            await self._backend.execute(self._backend.AGENT_REGISTRY_UPSERT, params)
            if self.logger: self.logger.debug(f"Logged agent registration/update for: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log agent registration for {agent_name}: {exc}")
            raise

    async def deregister_agent(self, agent_name: str) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            # SQLite parameters are position based. $2 is for Postgres.
            # AGENT_REGISTRY_DEACTIVATE for Postgres is "UPDATE agent_registry SET active = FALSE, updated_at = $2 WHERE agent_name = $1"
            # AGENT_REGISTRY_DEACTIVATE for SQLite is "UPDATE agent_registry SET active = 0, updated_at = ? WHERE agent_name = ?"
            # Therefore, we need to pass updated_at (now) first, then agent_name.
            params = (now, agent_name)
            if isinstance(self._backend, PostgresBackend):
                params = (agent_name, now)
            await self._backend.execute(self._backend.AGENT_REGISTRY_DEACTIVATE, params)
            if self.logger: self.logger.debug(f"Deregistered agent: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to deregister agent {agent_name}: {exc}")
            raise

    async def get_agent_details(self, agent_name: str) -> Optional[Dict[str, Any]]:
        if not self._ready(): return None
        try:
            row = await self._backend.fetch_one(self._backend.AGENT_REGISTRY_SELECT_ONE, (agent_name,))
            return self._deserialize_agent_registry_row(row) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve agent details for {agent_name}: {exc}")
            return None

    async def get_all_registered_agents(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.AGENT_REGISTRY_SELECT_ALL, ())
            return [self._deserialize_agent_registry_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve all registered agents: {exc}")
            return []

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
            self._backend = None
        self.is_active = False
        if self.logger: self.logger.info("Registry database connection closed.")

    def _ready(self) -> bool:
        return self._db_logging_enabled and self.is_active and self._backend is not None

    def _deserialize_agent_registry_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row: return {}
        for key in ["created_at", "updated_at"]:
            if key in row: row[key] = self._isoformat(row[key])
        # SQLite returns 0/1 for boolean, convert to actual boolean
        if self._backend and self._backend.name == "sqlite" and "active" in row:
            row["active"] = bool(row["active"])
        return row

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None