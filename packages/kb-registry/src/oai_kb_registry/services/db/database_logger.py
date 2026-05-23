"""
Database logger for Knowledge Base Registry.

Handles PostgreSQL and SQLite backends via oai_platform_core abstractions.

Tables
------
knowledge_bases
    Master registry record for each KB.
kb_configs
    Encrypted key-value store for per-KB connection details and API keys.
kb_documents
    Tracks each document (file upload, S3 reference, URL) linked to a KB.
kb_actions
    Audit log: register, add-document, remove-document, reindex, delete …
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oai_platform_core.db.base import BasePostgresBackend, DatabaseBackend, PersistentSQLiteBackend

try:
    import aiosqlite  # noqa: F401 (presence checked via aiosqlite is None)
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------

class PostgresBackend(BasePostgresBackend):
    """PostgreSQL backend for KB registry."""

    DEFAULT_PORT    = "5436"         # Keep different from other registries
    DEFAULT_DB_NAME = "kb_logs"

    # ------------------------------------------------------------------
    # knowledge_bases
    # ------------------------------------------------------------------
    KB_UPSERT = """
        INSERT INTO knowledge_bases (
            name, description, tags,
            vector_db_type, deployment_mode,
            embedding_model_id, embedding_region,
            chunk_size, chunk_overlap,
            status, created_at, updated_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        ON CONFLICT (name) DO UPDATE SET
            description         = EXCLUDED.description,
            tags                = EXCLUDED.tags,
            embedding_model_id  = EXCLUDED.embedding_model_id,
            embedding_region     = EXCLUDED.embedding_region,
            chunk_size          = EXCLUDED.chunk_size,
            chunk_overlap       = EXCLUDED.chunk_overlap,
            status              = EXCLUDED.status,
            updated_at          = EXCLUDED.updated_at
    """
    KB_SELECT_ONE = """
        SELECT kb.id, kb.name, kb.description, kb.tags,
               kb.vector_db_type, kb.deployment_mode,
               kb.embedding_model_id, kb.embedding_region,
               kb.chunk_size, kb.chunk_overlap,
               kb.status, kb.created_at, kb.updated_at,
               COALESCE(doc_counts.doc_count, 0) AS document_count,
               COALESCE(doc_counts.total_chunks, 0) AS total_chunks
        FROM knowledge_bases kb
        LEFT JOIN (
            SELECT kb_id,
                   COUNT(*) AS doc_count,
                   COALESCE(SUM(chunk_count), 0) AS total_chunks
            FROM kb_documents
            WHERE status = 'indexed'
            GROUP BY kb_id
        ) doc_counts ON doc_counts.kb_id = kb.id
        WHERE kb.name = $1
    """
    KB_SELECT_ALL = """
        SELECT kb.id, kb.name, kb.description, kb.tags,
               kb.vector_db_type, kb.deployment_mode,
               kb.embedding_model_id, kb.embedding_region,
               kb.chunk_size, kb.chunk_overlap,
               kb.status, kb.created_at, kb.updated_at,
               COALESCE(doc_counts.doc_count, 0) AS document_count,
               COALESCE(doc_counts.total_chunks, 0) AS total_chunks
        FROM knowledge_bases kb
        LEFT JOIN (
            SELECT kb_id,
                   COUNT(*) AS doc_count,
                   COALESCE(SUM(chunk_count), 0) AS total_chunks
            FROM kb_documents
            WHERE status = 'indexed'
            GROUP BY kb_id
        ) doc_counts ON doc_counts.kb_id = kb.id
        ORDER BY kb.name
    """
    KB_UPDATE_STATUS = "UPDATE knowledge_bases SET status = $1, updated_at = $2 WHERE name = $3"
    KB_DELETE = "DELETE FROM knowledge_bases WHERE name = $1"

    # ------------------------------------------------------------------
    # kb_configs
    # ------------------------------------------------------------------
    KB_CONFIG_UPSERT = """
        INSERT INTO kb_configs (kb_id, config_key, config_value_encrypted, updated_at)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (kb_id, config_key) DO UPDATE SET
            config_value_encrypted = EXCLUDED.config_value_encrypted,
            updated_at             = EXCLUDED.updated_at
    """
    KB_CONFIGS_SELECT = "SELECT config_key, config_value_encrypted FROM kb_configs WHERE kb_id = $1"
    KB_CONFIG_DELETE  = "DELETE FROM kb_configs WHERE kb_id = $1"

    # ------------------------------------------------------------------
    # kb_documents
    # ------------------------------------------------------------------
    KB_DOC_INSERT = """
        INSERT INTO kb_documents (
            kb_id, name, source_type, source_uri,
            file_size_bytes, chunk_count, status,
            error_message, indexed_at, created_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        RETURNING id
    """
    KB_DOC_SELECT_ONE = """
        SELECT id, kb_id, name, source_type, source_uri,
               file_size_bytes, chunk_count, status,
               error_message, indexed_at, created_at
        FROM kb_documents WHERE id = $1
    """
    KB_DOCS_SELECT_ALL = """
        SELECT id, kb_id, name, source_type, source_uri,
               file_size_bytes, chunk_count, status,
               error_message, indexed_at, created_at
        FROM kb_documents WHERE kb_id = $1 ORDER BY created_at DESC
    """
    KB_DOC_UPDATE_STATUS = """
        UPDATE kb_documents
        SET status = $1, chunk_count = $2, error_message = $3,
            indexed_at = $4
        WHERE id = $5
    """
    KB_DOC_DELETE = "DELETE FROM kb_documents WHERE id = $1"
    KB_DOCS_DELETE_BY_KB = "DELETE FROM kb_documents WHERE kb_id = $1"

    # ------------------------------------------------------------------
    # kb_actions
    # ------------------------------------------------------------------
    KB_ACTION_INSERT = """
        INSERT INTO kb_actions (kb_id, action, performed_by, message, created_at)
        VALUES ($1, $2, $3, $4, $5)
    """
    KB_ACTIONS_SELECT = """
        SELECT id, kb_id, action, performed_by, message, created_at
        FROM kb_actions WHERE kb_id = $1 ORDER BY created_at DESC
    """

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    async def _create_schema(self, logger: Optional[logging.Logger] = None) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id                  SERIAL PRIMARY KEY,
            name                VARCHAR(255) UNIQUE NOT NULL,
            description         TEXT,
            tags                TEXT[],
            vector_db_type      VARCHAR(50) NOT NULL DEFAULT 'chroma',
            deployment_mode     VARCHAR(50) NOT NULL DEFAULT 'builtin',
            embedding_model_id  VARCHAR(255) NOT NULL DEFAULT 'bedrock/amazon.titan-embed-text-v1',
            embedding_region     VARCHAR(100),
            chunk_size          INTEGER NOT NULL DEFAULT 1000,
            chunk_overlap       INTEGER NOT NULL DEFAULT 200,
            status              VARCHAR(50) NOT NULL DEFAULT 'active',
            created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_configs (
            id                      SERIAL PRIMARY KEY,
            kb_id                   INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            config_key              VARCHAR(255) NOT NULL,
            config_value_encrypted  TEXT,
            updated_at              TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(kb_id, config_key)
        );

        CREATE TABLE IF NOT EXISTS kb_documents (
            id               SERIAL PRIMARY KEY,
            kb_id            INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            name             VARCHAR(500) NOT NULL,
            source_type      VARCHAR(50) NOT NULL DEFAULT 'upload',
            source_uri       TEXT,
            file_size_bytes  BIGINT,
            chunk_count      INTEGER DEFAULT 0,
            status           VARCHAR(50) NOT NULL DEFAULT 'pending',
            error_message    TEXT,
            indexed_at       TIMESTAMP WITH TIME ZONE,
            created_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_actions (
            id            SERIAL PRIMARY KEY,
            kb_id         INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            action        VARCHAR(100) NOT NULL,
            performed_by  VARCHAR(255),
            message       TEXT,
            created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_bases_name   ON knowledge_bases(name);
        CREATE INDEX IF NOT EXISTS idx_kb_configs_kb_id       ON kb_configs(kb_id);
        CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_id     ON kb_documents(kb_id);
        CREATE INDEX IF NOT EXISTS idx_kb_documents_status    ON kb_documents(status);
        CREATE INDEX IF NOT EXISTS idx_kb_actions_kb_id       ON kb_actions(kb_id);
        """
        # Migration statements — add columns introduced in the LiteLLM refactor.
        # Safe to run on both fresh and existing databases.
        migrations = [
            "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS "
            "embedding_model_id VARCHAR(255) NOT NULL DEFAULT 'bedrock/amazon.titan-embed-text-v1'",
            "ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS "
            "embedding_region VARCHAR(100)",
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(schema)
                for stmt in migrations:
                    try:
                        await conn.execute(stmt)
                    except Exception as mig_exc:
                        if logger:
                            logger.warning("Migration skipped (%s): %s", stmt[:60], mig_exc)
            if logger:
                logger.info("KB registry schema initialised (Postgres)")
        except Exception as exc:
            if logger:
                logger.error("Failed to initialise KB schema: %s", exc)
            raise


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class SQLiteBackend(PersistentSQLiteBackend):
    """SQLite backend for KB registry (dev / demo with no Postgres)."""

    DEFAULT_DB_NAME = "kb_registry.db"
    USE_WAL_MODE    = False

    # ------------------------------------------------------------------
    # knowledge_bases
    # ------------------------------------------------------------------
    KB_UPSERT = """
        INSERT INTO knowledge_bases (
            name, description, tags,
            vector_db_type, deployment_mode,
            embedding_model_id, embedding_region,
            chunk_size, chunk_overlap,
            status, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            description         = excluded.description,
            tags                = excluded.tags,
            embedding_model_id  = excluded.embedding_model_id,
            embedding_region     = excluded.embedding_region,
            chunk_size          = excluded.chunk_size,
            chunk_overlap       = excluded.chunk_overlap,
            status              = excluded.status,
            updated_at          = excluded.updated_at
    """
    KB_SELECT_ONE = """
        SELECT kb.id, kb.name, kb.description, kb.tags,
               kb.vector_db_type, kb.deployment_mode,
               kb.embedding_model_id, kb.embedding_region,
               kb.chunk_size, kb.chunk_overlap,
               kb.status, kb.created_at, kb.updated_at,
               COALESCE(doc_counts.doc_count, 0) AS document_count,
               COALESCE(doc_counts.total_chunks, 0) AS total_chunks
        FROM knowledge_bases kb
        LEFT JOIN (
            SELECT kb_id,
                   COUNT(*) AS doc_count,
                   COALESCE(SUM(chunk_count), 0) AS total_chunks
            FROM kb_documents
            WHERE status = 'indexed'
            GROUP BY kb_id
        ) doc_counts ON doc_counts.kb_id = kb.id
        WHERE kb.name = ?
    """
    KB_SELECT_ALL = """
        SELECT kb.id, kb.name, kb.description, kb.tags,
               kb.vector_db_type, kb.deployment_mode,
               kb.embedding_model_id, kb.embedding_region,
               kb.chunk_size, kb.chunk_overlap,
               kb.status, kb.created_at, kb.updated_at,
               COALESCE(doc_counts.doc_count, 0) AS document_count,
               COALESCE(doc_counts.total_chunks, 0) AS total_chunks
        FROM knowledge_bases kb
        LEFT JOIN (
            SELECT kb_id,
                   COUNT(*) AS doc_count,
                   COALESCE(SUM(chunk_count), 0) AS total_chunks
            FROM kb_documents
            WHERE status = 'indexed'
            GROUP BY kb_id
        ) doc_counts ON doc_counts.kb_id = kb.id
        ORDER BY kb.name
    """
    KB_UPDATE_STATUS = "UPDATE knowledge_bases SET status = ?, updated_at = ? WHERE name = ?"
    KB_DELETE = "DELETE FROM knowledge_bases WHERE name = ?"

    KB_CONFIG_UPSERT = """
        INSERT INTO kb_configs (kb_id, config_key, config_value_encrypted, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(kb_id, config_key) DO UPDATE SET
            config_value_encrypted = excluded.config_value_encrypted,
            updated_at             = excluded.updated_at
    """
    KB_CONFIGS_SELECT = "SELECT config_key, config_value_encrypted FROM kb_configs WHERE kb_id = ?"
    KB_CONFIG_DELETE  = "DELETE FROM kb_configs WHERE kb_id = ?"

    KB_DOC_INSERT = """
        INSERT INTO kb_documents (
            kb_id, name, source_type, source_uri,
            file_size_bytes, chunk_count, status,
            error_message, indexed_at, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    KB_DOC_SELECT_ONE = """
        SELECT id, kb_id, name, source_type, source_uri,
               file_size_bytes, chunk_count, status,
               error_message, indexed_at, created_at
        FROM kb_documents WHERE id = ?
    """
    KB_DOCS_SELECT_ALL = """
        SELECT id, kb_id, name, source_type, source_uri,
               file_size_bytes, chunk_count, status,
               error_message, indexed_at, created_at
        FROM kb_documents WHERE kb_id = ? ORDER BY created_at DESC
    """
    KB_DOC_UPDATE_STATUS = """
        UPDATE kb_documents
        SET status = ?, chunk_count = ?, error_message = ?,
            indexed_at = ?
        WHERE id = ?
    """
    KB_DOC_DELETE        = "DELETE FROM kb_documents WHERE id = ?"
    KB_DOCS_DELETE_BY_KB = "DELETE FROM kb_documents WHERE kb_id = ?"

    KB_ACTION_INSERT  = """
        INSERT INTO kb_actions (kb_id, action, performed_by, message, created_at)
        VALUES (?,?,?,?,?)
    """
    KB_ACTIONS_SELECT = """
        SELECT id, kb_id, action, performed_by, message, created_at
        FROM kb_actions WHERE kb_id = ? ORDER BY created_at DESC
    """

    async def _create_schema(self) -> None:  # type: ignore[override]
        schema = """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT UNIQUE NOT NULL,
            description         TEXT,
            tags                TEXT,
            vector_db_type      TEXT NOT NULL DEFAULT 'chroma',
            deployment_mode     TEXT NOT NULL DEFAULT 'builtin',
            embedding_model_id  TEXT NOT NULL DEFAULT 'bedrock/amazon.titan-embed-text-v1',
            embedding_region     TEXT,
            chunk_size          INTEGER NOT NULL DEFAULT 1000,
            chunk_overlap       INTEGER NOT NULL DEFAULT 200,
            status              TEXT NOT NULL DEFAULT 'active',
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_configs (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id                   INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            config_key              TEXT NOT NULL,
            config_value_encrypted  TEXT,
            updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(kb_id, config_key)
        );

        CREATE TABLE IF NOT EXISTS kb_documents (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id            INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            name             TEXT NOT NULL,
            source_type      TEXT NOT NULL DEFAULT 'upload',
            source_uri       TEXT,
            file_size_bytes  INTEGER,
            chunk_count      INTEGER DEFAULT 0,
            status           TEXT NOT NULL DEFAULT 'pending',
            error_message    TEXT,
            indexed_at       DATETIME,
            created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kb_actions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_id         INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            action        TEXT NOT NULL,
            performed_by  TEXT,
            message       TEXT,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_bases_name ON knowledge_bases(name);
        CREATE INDEX IF NOT EXISTS idx_kb_configs_kb_id     ON kb_configs(kb_id);
        CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_id   ON kb_documents(kb_id);
        CREATE INDEX IF NOT EXISTS idx_kb_actions_kb_id     ON kb_actions(kb_id);
        """
        db = await self._get_conn()
        await db.executescript(schema)
        await db.commit()

        # Migration: add columns introduced in the LiteLLM refactor.
        # executescript cannot catch individual errors, so run each ALTER
        # separately and swallow the "duplicate column" error on re-runs.
        migrations = [
            "ALTER TABLE knowledge_bases ADD COLUMN "
            "embedding_model_id TEXT NOT NULL DEFAULT 'bedrock/amazon.titan-embed-text-v1'",
            "ALTER TABLE knowledge_bases ADD COLUMN embedding_region TEXT",
        ]
        for stmt in migrations:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Column already exists — safe to ignore
        await db.commit()


# ---------------------------------------------------------------------------
# High-level logger facade
# ---------------------------------------------------------------------------

class KBDatabaseLogger:
    """Facade over the database backend for KB registry operations."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._backend: Optional[DatabaseBackend] = None
        self._enabled = os.environ.get("REGISTRY_DB_LOGGING_ENABLED", "true").lower() == "true"

    async def initialize(self) -> bool:
        if not self._enabled:
            self.logger.info("KB registry database logging disabled")
            return False

        backend: DatabaseBackend = PostgresBackend()
        if await backend.initialize(self.logger):
            self._backend = backend
            return True

        backend = SQLiteBackend()
        if await backend.initialize(self.logger):
            self._backend = backend
            return True

        self.logger.warning("No database backend available — running in no-persistence mode")
        return False

    def _ready(self) -> bool:
        return self._enabled and self._backend is not None

    async def close(self) -> None:
        if self._backend:
            await self._backend.close()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _decode_tags(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if row and "tags" in row and isinstance(row["tags"], str):
            try:
                row["tags"] = json.loads(row["tags"]) if row["tags"] else []
            except (json.JSONDecodeError, TypeError):
                row["tags"] = []
        return row

    def _encode_tags(self, tags: List[str]) -> Any:
        if self._backend and self._backend.name == "postgres":
            return list(tags)
        return json.dumps(tags) if tags else "[]"

    # ------------------------------------------------------------------ #
    #  Knowledge Base CRUD                                                 #
    # ------------------------------------------------------------------ #

    async def create_knowledge_base(
        self,
        name: str,
        description: Optional[str],
        tags: List[str],
        vector_db_type: str,
        deployment_mode: str,
        embedding_model_id: str,
        embedding_region: Optional[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> Dict[str, Any]:
        if not self._ready():
            return {"id": None, "name": name}
        try:
            now = datetime.now(timezone.utc)
            params = (
                name, description, self._encode_tags(tags),
                vector_db_type, deployment_mode,
                embedding_model_id, embedding_region,
                chunk_size, chunk_overlap,
                "active", now, now,
            )
            await self._backend.execute(self._backend.KB_UPSERT, params)
            self.logger.debug("Created/updated KB: %s", name)
            row = await self._backend.fetch_one(self._backend.KB_SELECT_ONE, (name,))
            return self._decode_tags(dict(row)) if row else {"id": None, "name": name}
        except Exception as exc:
            self.logger.error("Failed to create KB %s: %s", name, exc)
            return {"id": None, "name": name}

    async def get_knowledge_base(self, name: str) -> Optional[Dict[str, Any]]:
        if not self._ready():
            return None
        try:
            row = await self._backend.fetch_one(self._backend.KB_SELECT_ONE, (name,))
            return self._decode_tags(dict(row)) if row else None
        except Exception as exc:
            self.logger.error("Failed to get KB %s: %s", name, exc)
            return None

    async def get_all_knowledge_bases(self) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            rows = await self._backend.fetch(self._backend.KB_SELECT_ALL, ())
            return [self._decode_tags(dict(r)) for r in rows]
        except Exception as exc:
            self.logger.error("Failed to list KBs: %s", exc)
            return []

    async def update_kb_status(self, name: str, status: str) -> None:
        if not self._ready():
            return
        try:
            now = datetime.now(timezone.utc)
            await self._backend.execute(self._backend.KB_UPDATE_STATUS, (status, now, name))
        except Exception as exc:
            self.logger.error("Failed to update status for KB %s: %s", name, exc)

    async def delete_knowledge_base(self, name: str) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.KB_DELETE, (name,))
            self.logger.info("Deleted KB: %s", name)
        except Exception as exc:
            self.logger.error("Failed to delete KB %s: %s", name, exc)

    # ------------------------------------------------------------------ #
    #  KB Config (encrypted connection details)                           #
    # ------------------------------------------------------------------ #

    async def upsert_kb_config(self, kb_id: int, key: str, value_encrypted: str) -> None:
        if not self._ready():
            return
        try:
            now = datetime.now(timezone.utc)
            await self._backend.execute(
                self._backend.KB_CONFIG_UPSERT,
                (kb_id, key, value_encrypted, now),
            )
        except Exception as exc:
            self.logger.error("Failed to upsert config key=%s for kb_id=%s: %s", key, kb_id, exc)

    async def get_kb_configs(self, kb_id: int) -> Dict[str, str]:
        if not self._ready():
            return {}
        try:
            rows = await self._backend.fetch(self._backend.KB_CONFIGS_SELECT, (kb_id,))
            return {r["config_key"]: r["config_value_encrypted"] for r in rows}
        except Exception as exc:
            self.logger.error("Failed to get configs for kb_id=%s: %s", kb_id, exc)
            return {}

    async def delete_kb_configs(self, kb_id: int) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.KB_CONFIG_DELETE, (kb_id,))
        except Exception as exc:
            self.logger.error("Failed to delete configs for kb_id=%s: %s", kb_id, exc)

    # ------------------------------------------------------------------ #
    #  Documents                                                           #
    # ------------------------------------------------------------------ #

    async def create_document(
        self,
        kb_id: int,
        name: str,
        source_type: str,
        source_uri: Optional[str],
        file_size_bytes: Optional[int],
    ) -> Dict[str, Any]:
        if not self._ready():
            return {"id": None, "name": name}
        try:
            now = datetime.now(timezone.utc)
            params = (kb_id, name, source_type, source_uri, file_size_bytes, 0, "pending", None, None, now)
            if self._backend.name == "postgres":
                row = await self._backend.fetch_one(
                    self._backend.KB_DOC_INSERT, params
                )
                doc_id = row["id"] if row else None
            else:
                await self._backend.execute(self._backend.KB_DOC_INSERT, params)
                # For SQLite, get the last inserted row
                row = await self._backend.fetch_one(
                    "SELECT id FROM kb_documents WHERE kb_id=? AND name=? ORDER BY created_at DESC LIMIT 1",
                    (kb_id, name),
                )
                doc_id = row["id"] if row else None
            doc = await self._backend.fetch_one(self._backend.KB_DOC_SELECT_ONE, (doc_id,))
            return dict(doc) if doc else {"id": doc_id, "name": name}
        except Exception as exc:
            self.logger.error("Failed to create document %s for kb_id=%s: %s", name, kb_id, exc)
            return {"id": None, "name": name}

    async def update_document_status(
        self,
        doc_id: int,
        status: str,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        if not self._ready():
            return
        try:
            indexed_at = datetime.now(timezone.utc) if status == "indexed" else None
            await self._backend.execute(
                self._backend.KB_DOC_UPDATE_STATUS,
                (status, chunk_count, error_message, indexed_at, doc_id),
            )
        except Exception as exc:
            self.logger.error("Failed to update status for doc_id=%s: %s", doc_id, exc)

    async def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        if not self._ready():
            return None
        try:
            row = await self._backend.fetch_one(self._backend.KB_DOC_SELECT_ONE, (doc_id,))
            return dict(row) if row else None
        except Exception as exc:
            self.logger.error("Failed to get document id=%s: %s", doc_id, exc)
            return None

    async def get_kb_documents(self, kb_id: int) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            rows = await self._backend.fetch(self._backend.KB_DOCS_SELECT_ALL, (kb_id,))
            return [dict(r) for r in rows]
        except Exception as exc:
            self.logger.error("Failed to list documents for kb_id=%s: %s", kb_id, exc)
            return []

    async def delete_document(self, doc_id: int) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.KB_DOC_DELETE, (doc_id,))
        except Exception as exc:
            self.logger.error("Failed to delete document id=%s: %s", doc_id, exc)

    async def delete_kb_documents(self, kb_id: int) -> None:
        if not self._ready():
            return
        try:
            await self._backend.execute(self._backend.KB_DOCS_DELETE_BY_KB, (kb_id,))
        except Exception as exc:
            self.logger.error("Failed to delete documents for kb_id=%s: %s", kb_id, exc)

    # ------------------------------------------------------------------ #
    #  Action audit log                                                    #
    # ------------------------------------------------------------------ #

    async def log_action(
        self,
        kb_id: int,
        action: str,
        performed_by: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if not self._ready():
            return
        try:
            now = datetime.now(timezone.utc)
            await self._backend.execute(
                self._backend.KB_ACTION_INSERT,
                (kb_id, action, performed_by, message, now),
            )
        except Exception as exc:
            self.logger.error("Failed to log action %s for kb_id=%s: %s", action, kb_id, exc)

    async def get_kb_actions(self, kb_id: int) -> List[Dict[str, Any]]:
        if not self._ready():
            return []
        try:
            rows = await self._backend.fetch(self._backend.KB_ACTIONS_SELECT, (kb_id,))
            return [dict(r) for r in rows]
        except Exception as exc:
            self.logger.error("Failed to get actions for kb_id=%s: %s", kb_id, exc)
            return []
