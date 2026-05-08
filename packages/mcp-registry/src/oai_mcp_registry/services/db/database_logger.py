"""
Database logger for tracking MCP registry events.
"""

from __future__ import annotations

import logging
import os
import json
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

    MCP_REGISTRY_UPSERT = """
        INSERT INTO mcp_registry
            (server_name, endpoint_url, port, description, active, registered_via, source, tags, current_version, available_versions, deployment_mode, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (server_name) DO UPDATE SET
            endpoint_url       = EXCLUDED.endpoint_url,
            port               = EXCLUDED.port,
            description        = EXCLUDED.description,
            active             = EXCLUDED.active,
            source             = EXCLUDED.source,
            tags               = EXCLUDED.tags,
            current_version    = EXCLUDED.current_version,
            available_versions = EXCLUDED.available_versions,
            deployment_mode    = EXCLUDED.deployment_mode,
            updated_at         = EXCLUDED.updated_at
    """
    MCP_REGISTRY_DEACTIVATE = "UPDATE mcp_registry SET active = FALSE, updated_at = $2 WHERE server_name = $1"
    MCP_REGISTRY_DELETE = "DELETE FROM mcp_registry WHERE server_name = $1"
    MCP_REGISTRY_SELECT_ONE = "SELECT * FROM mcp_registry WHERE server_name = $1"
    MCP_REGISTRY_SELECT_ALL = "SELECT * FROM mcp_registry"
    MCP_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM mcp_registry WHERE active = TRUE AND registered_via = 'dynamic'"

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
        name = os.environ.get("LOGGING_DB_NAME", "mcp_logs")
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
        if logger: logger.info("Creating MCP registry database schema...")
        mcp_registry_ddl = """
            CREATE TABLE IF NOT EXISTS mcp_registry (
                server_name        VARCHAR(255) PRIMARY KEY,
                endpoint_url       VARCHAR(255),
                port               INTEGER,
                description        TEXT,
                active             BOOLEAN,
                registered_via     VARCHAR(50) NOT NULL DEFAULT 'dynamic',
                source             VARCHAR(255),
                tags               TEXT,
                current_version    VARCHAR(255),
                available_versions TEXT,
                deployment_mode    VARCHAR(50) DEFAULT 'docker',
                created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        migrations = [
            "ALTER TABLE mcp_registry ADD COLUMN IF NOT EXISTS source VARCHAR(255);",
            "ALTER TABLE mcp_registry ADD COLUMN IF NOT EXISTS tags TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN IF NOT EXISTS current_version VARCHAR(255);",
            "ALTER TABLE mcp_registry ADD COLUMN IF NOT EXISTS available_versions TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN IF NOT EXISTS deployment_mode VARCHAR(50) DEFAULT 'docker';",
        ]
        async with self._pool.acquire() as conn:
            await conn.execute(mcp_registry_ddl)
            for migration in migrations:
                try:
                    await conn.execute(migration)
                except Exception:
                    pass
            if logger: logger.info("MCP registry database schema created/updated.")


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"

    MCP_REGISTRY_UPSERT = """
        INSERT INTO mcp_registry
            (server_name, endpoint_url, port, description, active, registered_via, source, tags, current_version, available_versions, deployment_mode, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(server_name) DO UPDATE SET
            endpoint_url       = EXCLUDED.endpoint_url,
            port               = EXCLUDED.port,
            description        = EXCLUDED.description,
            active             = EXCLUDED.active,
            source             = EXCLUDED.source,
            tags               = EXCLUDED.tags,
            current_version    = EXCLUDED.current_version,
            available_versions = EXCLUDED.available_versions,
            deployment_mode    = EXCLUDED.deployment_mode,
            updated_at         = EXCLUDED.updated_at
    """
    MCP_REGISTRY_DEACTIVATE = "UPDATE mcp_registry SET active = 0, updated_at = ? WHERE server_name = ?"
    MCP_REGISTRY_DELETE = "DELETE FROM mcp_registry WHERE server_name = ?"
    MCP_REGISTRY_SELECT_ONE = "SELECT * FROM mcp_registry WHERE server_name = ?"
    MCP_REGISTRY_SELECT_ALL = "SELECT * FROM mcp_registry"
    MCP_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM mcp_registry WHERE active = 1 AND registered_via = 'dynamic'"

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
        if db_dir: return os.path.join(db_dir, "mcp_registry.db")
        return "mcp_registry.db"

    async def _create_schema(self) -> None:
        mcp_registry_ddl = """
            CREATE TABLE IF NOT EXISTS mcp_registry (
                server_name        TEXT PRIMARY KEY,
                endpoint_url       TEXT,
                port               INTEGER,
                description        TEXT,
                active             BOOLEAN,
                registered_via     TEXT NOT NULL DEFAULT 'dynamic',
                source             TEXT,
                tags               TEXT,
                current_version    TEXT,
                available_versions TEXT,
                deployment_mode    TEXT DEFAULT 'docker',
                created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at         DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        migrations = [
            "ALTER TABLE mcp_registry ADD COLUMN source TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN tags TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN current_version TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN available_versions TEXT;",
            "ALTER TABLE mcp_registry ADD COLUMN deployment_mode TEXT DEFAULT 'docker';",
        ]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(mcp_registry_ddl)
            for migration in migrations:
                try:
                    await db.execute(migration)
                    await db.commit()
                except Exception:
                    pass


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

    async def log_server_registration(
            self,
            server_name: str,
            endpoint_url: str,
            port: int,
            description: Optional[str] = None,
            active: bool = True,
            registered_via: str = "dynamic",
            source: Optional[str] = None,
            tags: Optional[List[str]] = None,
            current_version: Optional[str] = None,
            available_versions: Optional[List[str]] = None,
            deployment_mode: str = "docker"
    ) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            tags_json = json.dumps(tags) if tags is not None else "[]"
            available_versions_json = json.dumps(available_versions) if available_versions is not None else "[]"
            params = (server_name, endpoint_url, port, description, active, registered_via, source, tags_json, current_version, available_versions_json, deployment_mode, now, now)
            await self._backend.execute(self._backend.MCP_REGISTRY_UPSERT, params)
            if self.logger: self.logger.debug(f"Logged MCP server registration/update for: {server_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log MCP server registration for {server_name}: {exc}")
            raise

    async def deregister_server(self, server_name: str) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (now, server_name)
            if isinstance(self._backend, PostgresBackend):
                params = (server_name, now)
            await self._backend.execute(self._backend.MCP_REGISTRY_DEACTIVATE, params)
            if self.logger: self.logger.debug(f"Deregistered MCP server: {server_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to deregister MCP server {server_name}: {exc}")
            raise

    async def delete_server(self, server_name: str) -> None:
        if not self._ready(): return
        try:
            params = (server_name,)
            await self._backend.execute(self._backend.MCP_REGISTRY_DELETE, params)
            if self.logger: self.logger.debug(f"Deleted MCP server entry: {server_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to delete MCP server {server_name}: {exc}")
            raise

    async def get_server_details(self, server_name: str) -> Optional[Dict[str, Any]]:
        if not self._ready(): return None
        try:
            row = await self._backend.fetch_one(self._backend.MCP_REGISTRY_SELECT_ONE, (server_name,))
            return self._deserialize_row(row) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve MCP server details for {server_name}: {exc}")
            return None

    async def get_all_servers(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.MCP_REGISTRY_SELECT_ALL, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve all registered MCP servers: {exc}")
            return []

    async def get_active_dynamic_servers(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.MCP_REGISTRY_SELECT_ACTIVE_DYNAMIC, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve active dynamic MCP servers: {exc}")
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
        if self._backend and self._backend.name == "sqlite" and "active" in row:
            row["active"] = bool(row["active"])
        for key in ["tags", "available_versions"]:
            if key in row and row[key]:
                try:
                    row[key] = json.loads(row[key])
                except Exception:
                    row[key] = []
        return row

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None
