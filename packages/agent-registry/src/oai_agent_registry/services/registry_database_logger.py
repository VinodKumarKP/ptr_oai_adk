"""
Database logger for tracking agent registry events.
"""

from __future__ import annotations

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

    # registered_via and framework are excluded from the upsert ON CONFLICT update — we never want to overwrite
    # 'config' with 'dynamic' if an agent name collides, and a re-registering dynamic agent
    # should keep its original source value.
    AGENT_REGISTRY_UPSERT = """
        INSERT INTO agent_registry
            (agent_name, endpoint_url, port, source_url, active, registered_via, framework, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (agent_name) DO UPDATE SET
            endpoint_url   = EXCLUDED.endpoint_url,
            port           = EXCLUDED.port,
            source_url = EXCLUDED.source_url,
            active         = EXCLUDED.active,
            updated_at     = EXCLUDED.updated_at
    """
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = FALSE, updated_at = $2 WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"
    AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM agent_registry WHERE active = TRUE AND registered_via = 'dynamic'"

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
                agent_name     VARCHAR(255) PRIMARY KEY,
                endpoint_url   VARCHAR(255),
                port           INTEGER,
                source_url VARCHAR(255),
                active         BOOLEAN,
                registered_via VARCHAR(50) NOT NULL DEFAULT 'dynamic',
                framework      VARCHAR(255),
                created_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        # ADD COLUMN is idempotent via the IF NOT EXISTS guard — safe to run on every startup
        # against an existing DB that pre-dates the registered_via column.
        migrate_ddl_registered_via = """
            ALTER TABLE agent_registry
                ADD COLUMN IF NOT EXISTS registered_via VARCHAR(50) NOT NULL DEFAULT 'dynamic';
        """
        migrate_ddl_framework = """
            ALTER TABLE agent_registry
                ADD COLUMN IF NOT EXISTS framework VARCHAR(255);
        """
        async with self._pool.acquire() as conn:
            await conn.execute(agent_registry_ddl)
            await conn.execute(migrate_ddl_registered_via)
            await conn.execute(migrate_ddl_framework)
            if logger: logger.info("Agent registry database schema created/updated.")


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"

    # Same registered_via exclusion from ON CONFLICT update as Postgres — see comment above.
    AGENT_REGISTRY_UPSERT = """
        INSERT INTO agent_registry
            (agent_name, endpoint_url, port, source_url, active, registered_via, framework, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_name) DO UPDATE SET
            endpoint_url   = EXCLUDED.endpoint_url,
            port           = EXCLUDED.port,
            source_url = EXCLUDED.source_url,
            active         = EXCLUDED.active,
            updated_at     = EXCLUDED.updated_at
    """
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = 0, updated_at = ? WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"
    AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM agent_registry WHERE active = 1 AND registered_via = 'dynamic'"

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
                agent_name     TEXT PRIMARY KEY,
                endpoint_url   TEXT,
                port           INTEGER,
                source_url TEXT,
                active         BOOLEAN,
                registered_via TEXT NOT NULL DEFAULT 'dynamic',
                framework      TEXT,
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        # SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we attempt the migration
        # and swallow the "duplicate column" error — safe for existing DBs.
        migrate_ddl_registered_via = "ALTER TABLE agent_registry ADD COLUMN registered_via TEXT NOT NULL DEFAULT 'dynamic'"
        migrate_ddl_framework = "ALTER TABLE agent_registry ADD COLUMN framework TEXT"
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(agent_registry_ddl)
            try:
                await db.execute(migrate_ddl_registered_via)
                await db.commit()
            except Exception:
                pass  # Column already exists — expected on all runs after the first
            try:
                await db.execute(migrate_ddl_framework)
                await db.commit()
            except Exception:
                pass  # Column already exists — expected on all runs after the first


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
            source_url: str,
            active: bool = True,
            registered_via: str = "dynamic",
            framework: Optional[str] = None,
    ) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (agent_name, endpoint_url, port, source_url, active, registered_via, framework, now, now)
            await self._backend.execute(self._backend.AGENT_REGISTRY_UPSERT, params)
            if self.logger: self.logger.debug(f"Logged agent registration/update for: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log agent registration for {agent_name}: {exc}")
            raise

    async def deregister_agent(self, agent_name: str) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            # Parameter order differs between backends due to positional vs named placeholders.
            # Postgres: $1=agent_name, $2=updated_at  |  SQLite: ?=updated_at, ?=agent_name
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
            return self._deserialize_row(row) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve agent details for {agent_name}: {exc}")
            return None

    async def get_all_registered_agents(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.AGENT_REGISTRY_SELECT_ALL, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve all registered agents: {exc}")
            return []

    async def get_active_dynamic_agents(self) -> List[Dict[str, Any]]:
        """
        Returns all agents that were dynamically registered and are still marked active.
        Used at registry startup to restore dynamic agents that survived a registry restart.
        Only agents the registry itself didn't register via config are returned — config
        agents are always reloaded from the config file, never from the DB.
        """
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve active dynamic agents: {exc}")
            return []

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
            self._backend = None
        self.is_active = False
        if self.logger: self.logger.info("Registry database connection closed.")

    def _ready(self) -> bool:
        return self._db_logging_enabled and self.is_active and self._backend is not None

    def _deserialize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row: return {}
        for key in ["created_at", "updated_at"]:
            if key in row: row[key] = self._isoformat(row[key])
        # SQLite stores booleans as 0/1 integers
        if self._backend and self._backend.name == "sqlite" and "active" in row:
            row["active"] = bool(row["active"])
        return row

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None