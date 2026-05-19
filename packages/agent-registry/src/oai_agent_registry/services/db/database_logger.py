"""
Database logger for tracking agent registry events.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oai_platform_core.db.base import DatabaseBackend, BasePostgresBackend, BaseSQLiteBackend

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


class PostgresBackend(BasePostgresBackend):
    DEFAULT_PORT    = "5432"
    DEFAULT_DB_NAME = "agent_logs"

    AGENT_REGISTRY_UPSERT = """
        INSERT INTO agent_registry
            (agent_name, endpoint_url, port, source, active, registered_via, framework, prompts, tags, description, current_version, available_versions, deployment_mode, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        ON CONFLICT (agent_name) DO UPDATE SET
            endpoint_url       = EXCLUDED.endpoint_url,
            port               = EXCLUDED.port,
            source             = EXCLUDED.source,
            active             = EXCLUDED.active,
            prompts            = EXCLUDED.prompts,
            tags               = EXCLUDED.tags,
            description        = EXCLUDED.description,
            current_version    = EXCLUDED.current_version,
            available_versions = EXCLUDED.available_versions,
            deployment_mode    = EXCLUDED.deployment_mode,
            updated_at         = EXCLUDED.updated_at
    """
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = FALSE, updated_at = $2 WHERE agent_name = $1"
    AGENT_REGISTRY_DELETE = "DELETE FROM agent_registry WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = $1"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"
    AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM agent_registry WHERE active = TRUE AND registered_via = 'dynamic'"

    AGENT_ACTION_INSERT = "INSERT INTO agent_actions (agent_name, action, version, created_at) VALUES ($1, $2, $3, $4)"
    AGENT_ACTION_SELECT_ALL = "SELECT id, agent_name, action, version, created_at FROM agent_actions WHERE agent_name = $1 ORDER BY created_at DESC"
    AGENT_ACTION_SELECT_FILTERED = "SELECT id, agent_name, action, version, created_at FROM agent_actions WHERE agent_name = $1 AND action = $2 ORDER BY created_at DESC"
    AGENT_ACTION_COUNT = "SELECT COUNT(*) as count FROM agent_actions WHERE agent_name = $1"

    async def _create_schema(self, logger: Optional[logging.Logger] = None) -> None:
        if logger: logger.info("Creating agent registry database schema...")
        agent_registry_ddl = """
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_name         VARCHAR(255) PRIMARY KEY,
                endpoint_url       VARCHAR(255),
                port               INTEGER,
                source             VARCHAR(255),
                active             BOOLEAN,
                registered_via     VARCHAR(50) NOT NULL DEFAULT 'dynamic',
                framework          VARCHAR(255),
                prompts            TEXT,
                tags               TEXT,
                description        TEXT,
                current_version    VARCHAR(255),
                available_versions TEXT,
                deployment_mode    VARCHAR(50) DEFAULT 'docker',
                created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        agent_actions_ddl = """
            CREATE TABLE IF NOT EXISTS agent_actions (
                id                 SERIAL PRIMARY KEY,
                agent_name         VARCHAR(255),
                action             VARCHAR(255),
                version            VARCHAR(255),
                created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """
        migrations = [
            "ALTER TABLE agent_registry RENAME COLUMN source_url TO source;",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS registered_via VARCHAR(50) NOT NULL DEFAULT 'dynamic';",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS framework VARCHAR(255);",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS prompts TEXT;",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS tags TEXT;",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS description TEXT;",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS current_version VARCHAR(255);",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS available_versions TEXT;",
            "ALTER TABLE agent_registry ADD COLUMN IF NOT EXISTS deployment_mode VARCHAR(50) DEFAULT 'docker';",
        ]
        async with self._pool.acquire() as conn:
            await conn.execute(agent_registry_ddl)
            await conn.execute(agent_actions_ddl)
            for migration in migrations:
                try:
                    await conn.execute(migration)
                except Exception:
                    pass
            if logger: logger.info("Agent registry database schema created/updated.")


class SQLiteBackend(BaseSQLiteBackend):
    DEFAULT_DB_NAME = "agent_registry.db"

    AGENT_REGISTRY_UPSERT = """
        INSERT INTO agent_registry
            (agent_name, endpoint_url, port, source, active, registered_via, framework, prompts, tags, description, current_version, available_versions, deployment_mode, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_name) DO UPDATE SET
            endpoint_url       = EXCLUDED.endpoint_url,
            port               = EXCLUDED.port,
            source             = EXCLUDED.source,
            active             = EXCLUDED.active,
            prompts            = EXCLUDED.prompts,
            tags               = EXCLUDED.tags,
            description        = EXCLUDED.description,
            current_version    = EXCLUDED.current_version,
            available_versions = EXCLUDED.available_versions,
            deployment_mode    = EXCLUDED.deployment_mode,
            updated_at         = EXCLUDED.updated_at
    """
    AGENT_REGISTRY_DEACTIVATE = "UPDATE agent_registry SET active = 0, updated_at = ? WHERE agent_name = ?"
    AGENT_REGISTRY_DELETE = "DELETE FROM agent_registry WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ONE = "SELECT * FROM agent_registry WHERE agent_name = ?"
    AGENT_REGISTRY_SELECT_ALL = "SELECT * FROM agent_registry"
    AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC = "SELECT * FROM agent_registry WHERE active = 1 AND registered_via = 'dynamic'"

    AGENT_ACTION_INSERT = "INSERT INTO agent_actions (agent_name, action, version, created_at) VALUES (?, ?, ?, ?)"
    AGENT_ACTION_SELECT_ALL = "SELECT id, agent_name, action, version, created_at FROM agent_actions WHERE agent_name = ? ORDER BY created_at DESC"
    AGENT_ACTION_SELECT_FILTERED = "SELECT id, agent_name, action, version, created_at FROM agent_actions WHERE agent_name = ? AND action = ? ORDER BY created_at DESC"
    AGENT_ACTION_COUNT = "SELECT COUNT(*) as count FROM agent_actions WHERE agent_name = ?"

    async def _create_schema(self) -> None:
        agent_registry_ddl = """
            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_name         TEXT PRIMARY KEY,
                endpoint_url       TEXT,
                port               INTEGER,
                source             TEXT,
                active             BOOLEAN,
                registered_via     TEXT NOT NULL DEFAULT 'dynamic',
                framework          TEXT,
                prompts            TEXT,
                tags               TEXT,
                description        TEXT,
                current_version    TEXT,
                available_versions TEXT,
                deployment_mode    TEXT DEFAULT 'docker',
                created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at         DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        agent_actions_ddl = """
            CREATE TABLE IF NOT EXISTS agent_actions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name         TEXT,
                action             TEXT,
                version            TEXT,
                created_at         DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """
        migrations = [
            "ALTER TABLE agent_registry RENAME COLUMN source_url TO source",
            "ALTER TABLE agent_registry ADD COLUMN registered_via TEXT NOT NULL DEFAULT 'dynamic'",
            "ALTER TABLE agent_registry ADD COLUMN framework TEXT",
            "ALTER TABLE agent_registry ADD COLUMN prompts TEXT",
            "ALTER TABLE agent_registry ADD COLUMN tags TEXT",
            "ALTER TABLE agent_registry ADD COLUMN description TEXT",
            "ALTER TABLE agent_registry ADD COLUMN current_version TEXT",
            "ALTER TABLE agent_registry ADD COLUMN available_versions TEXT",
            "ALTER TABLE agent_registry ADD COLUMN deployment_mode TEXT DEFAULT 'docker'",
        ]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute(agent_registry_ddl)
            await db.execute(agent_actions_ddl)
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

    async def log_agent_action(self, agent_name: str, action: str, version: Optional[str] = None) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (agent_name, action, version, now)
            await self._backend.execute(self._backend.AGENT_ACTION_INSERT, params)
            if self.logger: self.logger.debug(f"Logged agent action: {agent_name} -> {action}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log agent action for {agent_name}: {exc}")
            raise

    async def log_agent_registration(
            self,
            agent_name: str,
            endpoint_url: str,
            port: int,
            source: str,
            active: bool = True,
            registered_via: str = "dynamic",
            framework: Optional[str] = None,
            prompts: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            description: Optional[str] = None,
            current_version: Optional[str] = None,
            available_versions: Optional[List[str]] = None,
            deployment_mode: str = "docker"
    ) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            prompts_json = json.dumps(prompts) if prompts is not None else "[]"
            tags_json = json.dumps(tags) if tags is not None else "[]"
            available_versions_json = json.dumps(available_versions) if available_versions is not None else "[]"
            params = (agent_name, endpoint_url, port, source, active, registered_via, framework, prompts_json, tags_json, description, current_version, available_versions_json, deployment_mode, now, now)
            await self._backend.execute(self._backend.AGENT_REGISTRY_UPSERT, params)
            if self.logger: self.logger.debug(f"Logged agent registration/update for: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to log agent registration for {agent_name}: {exc}")
            raise

    async def deregister_agent(self, agent_name: str) -> None:
        if not self._ready(): return
        try:
            now = datetime.now(timezone.utc)
            params = (now, agent_name)
            if isinstance(self._backend, PostgresBackend):
                params = (agent_name, now)
            await self._backend.execute(self._backend.AGENT_REGISTRY_DEACTIVATE, params)
            if self.logger: self.logger.debug(f"Deregistered agent: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to deregister agent {agent_name}: {exc}")
            raise

    async def delete_agent(self, agent_name: str) -> None:
        if not self._ready(): return
        try:
            params = (agent_name,)
            await self._backend.execute(self._backend.AGENT_REGISTRY_DELETE, params)
            if self.logger: self.logger.debug(f"Deleted agent entry: {agent_name}")
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to delete agent {agent_name}: {exc}")
            raise

    async def get_agent_details(self, agent_name: str) -> Optional[Dict[str, Any]]:
        if not self._ready(): return None
        try:
            row = await self._backend.fetch_one(self._backend.AGENT_REGISTRY_SELECT_ONE, (agent_name,))
            return self._deserialize_row(row) if row else None
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve agent details for {agent_name}: {exc}")
            return None

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.AGENT_REGISTRY_SELECT_ALL, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve all registered agents: {exc}")
            return []

    async def get_active_dynamic_agents(self) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            rows = await self._backend.fetch(self._backend.AGENT_REGISTRY_SELECT_ACTIVE_DYNAMIC, ())
            return [self._deserialize_row(r) for r in rows]
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve active dynamic agents: {exc}")
            return []

    async def get_agent_actions(self, agent_name: str, action_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._ready(): return []
        try:
            if action_type:
                rows = await self._backend.fetch(self._backend.AGENT_ACTION_SELECT_FILTERED, (agent_name, action_type))
            else:
                rows = await self._backend.fetch(self._backend.AGENT_ACTION_SELECT_ALL, (agent_name,))
            result = rows[:limit] if limit else rows
            for row in result:
                if 'created_at' in row:
                    row['created_at'] = self._isoformat(row['created_at'])
            return result
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve agent actions for {agent_name}: {exc}")
            return []

    async def get_agent_action_count(self, agent_name: str) -> int:
        if not self._ready(): return 0
        try:
            row = await self._backend.fetch_one(self._backend.AGENT_ACTION_COUNT, (agent_name,))
            return row['count'] if row and 'count' in row else 0
        except Exception as exc:
            if self.logger: self.logger.error(f"Failed to retrieve action count for {agent_name}: {exc}")
            return 0

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
        for key in ["prompts", "tags", "available_versions"]:
            if key in row and row[key]:
                try:
                    row[key] = json.loads(row[key])
                except Exception:
                    row[key] = []
        return row

    @staticmethod
    def _isoformat(value: Any) -> Optional[str]:
        return value.isoformat() if isinstance(value, datetime) else str(value) if value else None
