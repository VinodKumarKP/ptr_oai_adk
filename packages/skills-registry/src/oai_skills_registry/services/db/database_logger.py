"""
Database logger for skills registry.
Handles PostgreSQL and SQLite backends.
"""

import logging
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oai_platform_core.db.base import DatabaseBackend, BasePostgresBackend, PersistentSQLiteBackend

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


class PostgresBackend(BasePostgresBackend):
    """PostgreSQL backend for skills registry."""
    DEFAULT_PORT    = "5434"
    DEFAULT_DB_NAME = "skills_logs"

    # SQL Queries
    SKILL_UPSERT = """
        INSERT INTO skills (name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (name) DO UPDATE SET
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            tags = EXCLUDED.tags,
            current_version = EXCLUDED.current_version,
            status = EXCLUDED.status,
            git_repository_url = EXCLUDED.git_repository_url,
            updated_at = EXCLUDED.updated_at
    """

    SKILL_SELECT_ONE = "SELECT id, name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at FROM skills WHERE name = $1"
    SKILL_SELECT_ALL = "SELECT id, name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at FROM skills ORDER BY name"
    SKILL_DELETE = "DELETE FROM skills WHERE name = $1"

    SKILL_VERSION_INSERT = """
        INSERT INTO skill_versions (skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    """

    SKILL_VERSION_SELECT = "SELECT id, skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, deprecated_at, created_at FROM skill_versions WHERE skill_id = $1 AND version = $2"
    SKILL_VERSIONS_SELECT_ALL = "SELECT id, skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, deprecated_at, created_at FROM skill_versions WHERE skill_id = $1 ORDER BY created_at DESC"
    # params order: (status, published_by, published_at, status_again, deprecated_at, version_id)
    # $4 = status repeated so CASE WHEN comparison uses a dedicated placeholder like SQLite does.
    SKILL_VERSION_UPDATE_STATUS = "UPDATE skill_versions SET status = $1, published_by = $2, published_at = $3, deprecated_at = CASE WHEN $4 = 'deprecated' THEN $5 ELSE deprecated_at END WHERE id = $6"

    SKILL_ACTION_INSERT = "INSERT INTO skill_actions (skill_id, action, from_version, to_version, performed_by, message, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)"
    SKILL_ACTIONS_SELECT_ALL = "SELECT id, skill_id, action, from_version, to_version, performed_by, message, created_at FROM skill_actions WHERE skill_id = $1 ORDER BY created_at DESC"
    SKILL_ACTION_COUNT = "SELECT COUNT(*) as count FROM skill_actions WHERE skill_id = $1"

    GIT_SOURCE_INSERT = """
        INSERT INTO skill_git_sources (name, git_provider, repository, git_url, branch, auth_type, auth_token_encrypted, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
    """

    GIT_SOURCE_SELECT_ONE = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources WHERE id = $1"
    GIT_SOURCE_SELECT_ALL = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources ORDER BY name"
    GIT_SOURCE_SELECT_BY_NAME = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources WHERE name = $1"

    AGENT_SKILL_MAPPING_INSERT = "INSERT INTO agent_skill_versions (agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade) VALUES ($1, $2, $3, $4, $5, $6)"
    AGENT_SKILL_MAPPING_SELECT = "SELECT id, agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade FROM agent_skill_versions WHERE agent_id = $1 AND skill_id = $2"
    AGENT_SKILL_MAPPINGS_SELECT = "SELECT id, agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade FROM agent_skill_versions WHERE agent_id = $1"
    AGENT_SKILL_MAPPING_UPDATE = "UPDATE agent_skill_versions SET current_resolved_version = $1 WHERE id = $2"

    async def _create_schema(self, logger: Optional[logging.Logger] = None) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS skills (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            category VARCHAR(100),
            tags TEXT[],
            current_version VARCHAR(50),
            status VARCHAR(50) DEFAULT 'active',
            author VARCHAR(255),
            git_repository_url VARCHAR(500),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skill_versions (
            id SERIAL PRIMARY KEY,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            version VARCHAR(50) NOT NULL,
            git_source_id INTEGER,
            git_branch VARCHAR(100),
            git_commit_sha VARCHAR(40),
            git_tag VARCHAR(100),
            content TEXT,
            config JSONB,
            dependencies JSONB,
            breaking_changes JSONB,
            status VARCHAR(50) DEFAULT 'draft',
            published_by VARCHAR(255),
            published_at TIMESTAMP WITH TIME ZONE,
            deprecated_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(skill_id, version)
        );

        CREATE TABLE IF NOT EXISTS skill_actions (
            id SERIAL PRIMARY KEY,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            action VARCHAR(50) NOT NULL,
            from_version VARCHAR(50),
            to_version VARCHAR(50),
            performed_by VARCHAR(255) NOT NULL,
            message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skill_git_sources (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            git_provider VARCHAR(50) NOT NULL,
            repository VARCHAR(255) NOT NULL,
            git_url VARCHAR(500) NOT NULL,
            branch VARCHAR(100) DEFAULT 'main',
            auth_type VARCHAR(50) DEFAULT 'token',
            auth_token_encrypted TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_skill_versions (
            id SERIAL PRIMARY KEY,
            agent_id VARCHAR(255) NOT NULL,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            version_constraint VARCHAR(100),
            current_resolved_version VARCHAR(50),
            installed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            auto_upgrade BOOLEAN DEFAULT false,
            UNIQUE(agent_id, skill_id)
        );

        CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
        CREATE INDEX IF NOT EXISTS idx_skill_versions_skill_id ON skill_versions(skill_id);
        CREATE INDEX IF NOT EXISTS idx_skill_actions_skill_id ON skill_actions(skill_id);
        CREATE INDEX IF NOT EXISTS idx_agent_skill_agent_id ON agent_skill_versions(agent_id);
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(schema)
            if logger:
                logger.info("Skills registry schema initialized")
        except Exception as e:
            if logger:
                logger.error(f"Failed to initialize schema: {e}")


class SQLiteBackend(PersistentSQLiteBackend):
    """SQLite backend for skills registry (persistent connection)."""
    DEFAULT_DB_NAME = "skills_registry.db"
    USE_WAL_MODE    = False

    # SQL Queries (using ? for SQLite parameter binding)
    SKILL_UPSERT = """
        INSERT INTO skills (name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = excluded.description,
            category = excluded.category,
            tags = excluded.tags,
            current_version = excluded.current_version,
            status = excluded.status,
            git_repository_url = excluded.git_repository_url,
            updated_at = excluded.updated_at
    """

    SKILL_SELECT_ONE = "SELECT id, name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at FROM skills WHERE name = ?"
    SKILL_SELECT_ALL = "SELECT id, name, description, category, tags, current_version, status, author, git_repository_url, created_at, updated_at FROM skills ORDER BY name"
    SKILL_DELETE = "DELETE FROM skills WHERE name = ?"

    SKILL_VERSION_INSERT = """
        INSERT INTO skill_versions (skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    SKILL_VERSION_SELECT = "SELECT id, skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, deprecated_at, created_at FROM skill_versions WHERE skill_id = ? AND version = ?"
    SKILL_VERSIONS_SELECT_ALL = "SELECT id, skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag, content, config, dependencies, breaking_changes, status, published_by, published_at, deprecated_at, created_at FROM skill_versions WHERE skill_id = ? ORDER BY created_at DESC"
    SKILL_VERSION_UPDATE_STATUS = "UPDATE skill_versions SET status = ?, published_by = ?, published_at = ?, deprecated_at = CASE WHEN ? = 'deprecated' THEN ? ELSE deprecated_at END WHERE id = ?"

    SKILL_ACTION_INSERT = "INSERT INTO skill_actions (skill_id, action, from_version, to_version, performed_by, message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    SKILL_ACTIONS_SELECT_ALL = "SELECT id, skill_id, action, from_version, to_version, performed_by, message, created_at FROM skill_actions WHERE skill_id = ? ORDER BY created_at DESC"
    SKILL_ACTION_COUNT = "SELECT COUNT(*) as count FROM skill_actions WHERE skill_id = ?"

    GIT_SOURCE_INSERT = """
        INSERT INTO skill_git_sources (name, git_provider, repository, git_url, branch, auth_type, auth_token_encrypted, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    GIT_SOURCE_SELECT_ONE = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources WHERE id = ?"
    GIT_SOURCE_SELECT_ALL = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources ORDER BY name"
    GIT_SOURCE_SELECT_BY_NAME = "SELECT id, name, git_provider, repository, git_url, branch, auth_type, created_at, updated_at FROM skill_git_sources WHERE name = ?"

    AGENT_SKILL_MAPPING_INSERT = "INSERT INTO agent_skill_versions (agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade) VALUES (?, ?, ?, ?, ?, ?)"
    AGENT_SKILL_MAPPING_SELECT = "SELECT id, agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade FROM agent_skill_versions WHERE agent_id = ? AND skill_id = ?"
    AGENT_SKILL_MAPPINGS_SELECT = "SELECT id, agent_id, skill_id, version_constraint, current_resolved_version, installed_at, auto_upgrade FROM agent_skill_versions WHERE agent_id = ?"
    AGENT_SKILL_MAPPING_UPDATE = "UPDATE agent_skill_versions SET current_resolved_version = ? WHERE id = ?"

    async def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            category TEXT,
            tags TEXT,
            current_version TEXT,
            status TEXT DEFAULT 'active',
            author TEXT,
            git_repository_url TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skill_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            version TEXT NOT NULL,
            git_source_id INTEGER,
            git_branch TEXT,
            git_commit_sha TEXT,
            git_tag TEXT,
            content TEXT,
            config TEXT,
            dependencies TEXT,
            breaking_changes TEXT,
            status TEXT DEFAULT 'draft',
            published_by TEXT,
            published_at DATETIME,
            deprecated_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(skill_id, version)
        );

        CREATE TABLE IF NOT EXISTS skill_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            action TEXT NOT NULL,
            from_version TEXT,
            to_version TEXT,
            performed_by TEXT NOT NULL,
            message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skill_git_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            git_provider TEXT NOT NULL,
            repository TEXT NOT NULL,
            git_url TEXT NOT NULL,
            branch TEXT DEFAULT 'main',
            auth_type TEXT DEFAULT 'token',
            auth_token_encrypted TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_skill_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            version_constraint TEXT,
            current_resolved_version TEXT,
            installed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            auto_upgrade BOOLEAN DEFAULT 0,
            UNIQUE(agent_id, skill_id)
        );

        CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
        CREATE INDEX IF NOT EXISTS idx_skill_versions_skill_id ON skill_versions(skill_id);
        CREATE INDEX IF NOT EXISTS idx_skill_actions_skill_id ON skill_actions(skill_id);
        CREATE INDEX IF NOT EXISTS idx_agent_skill_agent_id ON agent_skill_versions(agent_id);
        """
        db = await self._get_conn()
        await db.executescript(schema)
        await db.commit()


class SkillsDatabaseLogger:
    """Main database logger for skills registry."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._backend: Optional[DatabaseBackend] = None
        self._enabled = os.environ.get("REGISTRY_DB_LOGGING_ENABLED", "true").lower() == "true"

    async def initialize(self) -> bool:
        if not self._enabled:
            self.logger.info("Skills registry database logging disabled")
            return False

        # Try PostgreSQL first, then SQLite
        backend = PostgresBackend()
        if await backend.initialize(self.logger):
            self._backend = backend
            return True

        backend = SQLiteBackend()
        if await backend.initialize(self.logger):
            self._backend = backend
            return True

        self.logger.warning("No database backend available")
        return False

    def _ready(self) -> bool:
        return self._enabled and self._backend is not None

    async def close(self) -> None:
        if self._backend:
            await self._backend.close()

    # ==================== SKILL OPERATIONS ====================

    def _process_skill(self, skill: Dict[str, Any]) -> Dict[str, Any]:
        if not skill:
            return skill
        if "tags" in skill and isinstance(skill["tags"], str):
            try:
                skill["tags"] = json.loads(skill["tags"]) if skill["tags"] else []
            except (json.JSONDecodeError, TypeError):
                skill["tags"] = []
        return skill

    async def create_skill(self, name: str, description: str, category: str,
                           tags: List[str], author: str, git_repository_url: Optional[str] = None) -> Dict[str, Any]:
        if not self._ready():
            return {"id": None, "name": name}
        try:
            now = datetime.now(timezone.utc)
            if self._backend.name == "postgres":
                tags_param = list(tags) if tags else []
            else:
                tags_param = json.dumps(tags) if tags else None
            params = (name, description, category, tags_param, None, "active", author, git_repository_url, now, now)
            await self._backend.execute(self._backend.SKILL_UPSERT, params)
            self.logger.debug(f"Created skill: {name}")
            skill = await self._backend.fetch_one(self._backend.SKILL_SELECT_ONE, (name,))
            return self._process_skill(skill) if skill else {"id": None, "name": name}
        except Exception as e:
            self.logger.error(f"Failed to create skill {name}: {e}")
            return {"id": None, "name": name}

    async def get_skill(self, name: str) -> Optional[Dict[str, Any]]:
        if not self._ready():
            return None
        try:
            skill = await self._backend.fetch_one(self._backend.SKILL_SELECT_ONE, (name,))
            return self._process_skill(skill) if skill else None
        except Exception as e:
            self.logger.error(f"Failed to get skill {name}: {e}")
            return None

    async def get_all_skills(self) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            skills = await self._backend.fetch(self._backend.SKILL_SELECT_ALL, ())
            return [self._process_skill(skill) for skill in skills]
        except Exception as e:
            self.logger.error(f"Failed to get all skills: {e}")
            return []

    async def update_skill_current_version(self, skill_name: str, version: str) -> None:
        if not self._ready():
            return
        try:
            now = datetime.now(timezone.utc)
            ph = self._backend.PLACEHOLDER
            if ph == "$":
                query = "UPDATE skills SET current_version = $1, updated_at = $2 WHERE name = $3"
            else:
                query = "UPDATE skills SET current_version = ?, updated_at = ? WHERE name = ?"
            await self._backend.execute(query, (version, now, skill_name))
            self.logger.debug(f"Updated {skill_name} current version to {version}")
        except Exception as e:
            self.logger.error(f"Failed to update skill version: {e}")

    async def delete_skill(self, name: str) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.SKILL_DELETE, (name,))
            self.logger.info(f"Deleted skill: {name}")
        except Exception as e:
            self.logger.error(f"Failed to delete skill {name}: {e}")

    # ==================== SKILL VERSION OPERATIONS ====================

    async def create_skill_version(self, skill_id: int, version: str, git_source_id: Optional[int],
                                   git_branch: Optional[str], git_commit_sha: Optional[str],
                                   git_tag: Optional[str], content: str, config: Optional[Dict],
                                   dependencies: Optional[Dict], breaking_changes: Optional[List[str]],
                                   status: str, published_by: str) -> Dict[str, Any]:
        if not self._ready():
            return {"id": None, "version": version}
        try:
            now = datetime.now(timezone.utc)
            config_param = json.dumps(config) if config else None
            deps_param = json.dumps(dependencies) if dependencies else None
            breaking_param = json.dumps(breaking_changes) if breaking_changes else None
            params = (
                skill_id, version, git_source_id, git_branch, git_commit_sha, git_tag,
                content, config_param, deps_param, breaking_param, status, published_by, None, now
            )
            await self._backend.execute(self._backend.SKILL_VERSION_INSERT, params)
            self.logger.debug(f"Created skill version: {version}")
            return await self._backend.fetch_one(
                self._backend.SKILL_VERSION_SELECT, (skill_id, version)
            ) or {"id": None, "version": version}
        except Exception as e:
            self.logger.error(f"Failed to create skill version {version}: {e}")
            return {"id": None, "version": version}

    async def get_skill_version(self, skill_id: int, version: str) -> Optional[Dict[str, Any]]:
        if not self._ready():
            return None
        try:
            return await self._backend.fetch_one(self._backend.SKILL_VERSION_SELECT, (skill_id, version))
        except Exception as e:
            self.logger.error(f"Failed to get skill version: {e}")
            return None

    async def get_skill_versions(self, skill_id: int) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            return await self._backend.fetch(self._backend.SKILL_VERSIONS_SELECT_ALL, (skill_id,))
        except Exception as e:
            self.logger.error(f"Failed to get skill versions: {e}")
            return []

    async def update_skill_version_status(self, version_id: int, status: str,
                                          published_by: Optional[str] = None,
                                          published_at: Optional[datetime] = None,
                                          deprecated_at: Optional[datetime] = None) -> None:
        if not self._ready():
            return
        try:
            params = (status, published_by, published_at, status, deprecated_at, version_id)
            await self._backend.execute(self._backend.SKILL_VERSION_UPDATE_STATUS, params)
            self.logger.debug(f"Updated skill version status to {status}")
        except Exception as e:
            self.logger.error(f"Failed to update skill version status: {e}")

    # ==================== SKILL ACTION OPERATIONS ====================

    async def log_skill_action(self, skill_id: int, action: str, from_version: Optional[str],
                               to_version: Optional[str], performed_by: str,
                               message: Optional[str] = None) -> Dict[str, Any]:
        if not self._ready():
            return {"id": None}
        try:
            now = datetime.now(timezone.utc)
            params = (skill_id, action, from_version, to_version, performed_by, message, now)
            await self._backend.execute(self._backend.SKILL_ACTION_INSERT, params)
            self.logger.debug(f"Logged skill action: {action}")
            actions = await self._backend.fetch(self._backend.SKILL_ACTIONS_SELECT_ALL, (skill_id,))
            return actions[0] if actions else {"id": None}
        except Exception as e:
            self.logger.error(f"Failed to log skill action: {e}")
            return {"id": None}

    async def get_skill_actions(self, skill_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            actions = await self._backend.fetch(self._backend.SKILL_ACTIONS_SELECT_ALL, (skill_id,))
            for action in actions[:limit]:
                if isinstance(action.get("created_at"), str):
                    continue
                if action.get("created_at"):
                    action["created_at"] = action["created_at"].isoformat()
            return actions[:limit]
        except Exception as e:
            self.logger.error(f"Failed to get skill actions: {e}")
            return []

    async def get_skill_action_count(self, skill_id: int) -> int:
        if not self._ready():
            return 0
        try:
            result = await self._backend.fetch_one(self._backend.SKILL_ACTION_COUNT, (skill_id,))
            return result["count"] if result else 0
        except Exception as e:
            self.logger.error(f"Failed to get skill action count: {e}")
            return 0

    # ==================== GIT SOURCE OPERATIONS ====================

    async def register_git_source(self, config: Dict) -> int:
        if not self._ready():
            return None
        try:
            now = datetime.now(timezone.utc)
            params = (
                config["name"], config["git_provider"], config["repository"],
                config["git_url"], config.get("branch", "main"),
                config.get("auth_type", "token"), config.get("auth_token_encrypted"),
                now, now
            )
            await self._backend.execute(self._backend.GIT_SOURCE_INSERT, params)
            self.logger.debug(f"Registered Git source: {config['name']}")
            source = await self._backend.fetch_one(self._backend.GIT_SOURCE_SELECT_BY_NAME, (config["name"],))
            return source["id"] if source else None
        except Exception as e:
            self.logger.error(f"Failed to register Git source: {e}")
            return None

    async def get_git_source(self, source_id: int) -> Optional[Dict[str, Any]]:
        if not self._ready():
            return None
        try:
            return await self._backend.fetch_one(self._backend.GIT_SOURCE_SELECT_ONE, (source_id,))
        except Exception as e:
            self.logger.error(f"Failed to get Git source: {e}")
            return None

    async def get_all_git_sources(self) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            return await self._backend.fetch(self._backend.GIT_SOURCE_SELECT_ALL, ())
        except Exception as e:
            self.logger.error(f"Failed to get Git sources: {e}")
            return []

    # ==================== AGENT-SKILL MAPPING OPERATIONS ====================

    async def install_skill_on_agent(self, agent_id: str, skill_id: int,
                                     version_constraint: str, resolved_version: Optional[str]) -> None:
        if not self._ready():
            return
        try:
            now = datetime.now(timezone.utc)
            params = (agent_id, skill_id, version_constraint, resolved_version, now, False)
            await self._backend.execute(self._backend.AGENT_SKILL_MAPPING_INSERT, params)
            self.logger.debug(f"Installed skill {skill_id} on agent {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to install skill on agent: {e}")

    async def get_agent_skills(self, agent_id: str) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            return await self._backend.fetch(self._backend.AGENT_SKILL_MAPPINGS_SELECT, (agent_id,))
        except Exception as e:
            self.logger.error(f"Failed to get agent skills: {e}")
            return []

    async def update_agent_skill_version(self, mapping_id: int, resolved_version: str) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.AGENT_SKILL_MAPPING_UPDATE, (resolved_version, mapping_id))
            self.logger.debug(f"Updated agent skill version to {resolved_version}")
        except Exception as e:
            self.logger.error(f"Failed to update agent skill version: {e}")
