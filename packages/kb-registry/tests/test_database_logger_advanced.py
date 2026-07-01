"""
Advanced unit tests for KBDatabaseLogger.

Uses SQLite in-process so no real PostgreSQL is needed.
All tests patch REGISTRY_DB_LOGGING_ENABLED=true so state from
other test modules that set it to false does not bleed in.
"""

import os
import pytest
import pytest_asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from oai_kb_registry.services.db.database_logger import KBDatabaseLogger


# ---------------------------------------------------------------------------
# Shared fixture – a fully initialized SQLite-backed logger
# ---------------------------------------------------------------------------

class TestKBDatabaseLoggerAdvanced:
    """Tests for KBDatabaseLogger backed by an in-memory SQLite DB."""

    @pytest_asyncio.fixture
    async def db_logger(self, tmp_path):
        db_file = tmp_path / "test_kb_registry.db"
        with patch.dict(os.environ, {
            "REGISTRY_DB_LOGGING_ENABLED": "true",
            "SQLITE_DB_PATH": str(db_file),
        }):
            # Patch Postgres so we fall through to SQLite
            with patch(
                "oai_kb_registry.services.db.database_logger.PostgresBackend.initialize",
                new_callable=AsyncMock,
                return_value=False,
            ):
                logger = KBDatabaseLogger(logger=logging.getLogger("test"))
                success = await logger.initialize()
                assert success is True, "SQLite backend failed to init"
                assert logger._ready() is True

        yield logger
        await logger.close()

    # -----------------------------------------------------------------------
    # Knowledge base CRUD
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_kb_crud_operations(self, db_logger):
        # Create
        kb = await db_logger.create_knowledge_base(
            name="mykb",
            description="desc",
            tags=["tag1", "tag2"],
            vector_db_type="postgres",
            deployment_mode="builtin",
            embedding_model_id="model-id",
            embedding_region="us-east-1",
            chunk_size=100,
            chunk_overlap=10,
            retrieval_config={"top_k": 3},
        )
        assert kb["id"] is not None
        assert kb["name"] == "mykb"

        # Get
        res = await db_logger.get_knowledge_base("mykb")
        assert res["name"] == "mykb"
        assert res["description"] == "desc"
        assert "tag1" in res["tags"]

        # List all
        all_kbs = await db_logger.get_all_knowledge_bases()
        assert len(all_kbs) >= 1
        assert any(kb["name"] == "mykb" for kb in all_kbs)

        # Update status
        await db_logger.update_kb_status("mykb", "ready")
        res = await db_logger.get_knowledge_base("mykb")
        assert res["status"] == "ready"

        # Upsert config
        await db_logger.upsert_kb_config(kb["id"], "api_key", "secret-enc")
        configs = await db_logger.get_kb_configs(kb["id"])
        assert configs["api_key"] == "secret-enc"

        # Delete config
        await db_logger.delete_kb_configs(kb["id"])
        configs = await db_logger.get_kb_configs(kb["id"])
        assert configs == {}

        # Delete KB
        await db_logger.delete_knowledge_base("mykb")
        res = await db_logger.get_knowledge_base("mykb")
        assert res is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_kb(self, db_logger):
        result = await db_logger.get_knowledge_base("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_kbs_empty(self, db_logger):
        result = await db_logger.get_all_knowledge_bases()
        assert isinstance(result, list)

    # -----------------------------------------------------------------------
    # Document CRUD
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_document_operations(self, db_logger):
        kb = await db_logger.create_knowledge_base(
            name="mykb2", description="d", tags=[], vector_db_type="chroma",
            deployment_mode="builtin", embedding_model_id="m", embedding_region=None,
            chunk_size=100, chunk_overlap=10,
        )
        kb_id = kb["id"]

        # Create document
        doc = await db_logger.create_document(
            kb_id=kb_id,
            name="file.txt",
            source_type="upload",
            source_uri=None,
            file_size_bytes=500,
        )
        doc_id = doc["id"]
        assert doc_id is not None

        # Get document
        res = await db_logger.get_document(doc_id)
        assert res["name"] == "file.txt"
        assert res["status"] == "pending"

        # Update status → indexed
        await db_logger.update_document_status(doc_id, "indexed", chunk_count=5, error_message=None)
        res = await db_logger.get_document(doc_id)
        assert res["status"] == "indexed"
        assert res["chunk_count"] == 5

        # Update status → error
        await db_logger.update_document_status(doc_id, "error", chunk_count=0, error_message="fail")
        res = await db_logger.get_document(doc_id)
        assert res["status"] == "error"
        assert res["error_message"] == "fail"

        # List docs for KB
        docs = await db_logger.get_kb_documents(kb_id)
        assert len(docs) == 1
        assert docs[0]["name"] == "file.txt"

        # Delete document
        await db_logger.delete_document(doc_id)
        res = await db_logger.get_document(doc_id)
        assert res is None

    @pytest.mark.asyncio
    async def test_delete_kb_documents(self, db_logger):
        kb = await db_logger.create_knowledge_base(
            name="bulk_kb", description="d", tags=[], vector_db_type="chroma",
            deployment_mode="builtin", embedding_model_id="m", embedding_region=None,
            chunk_size=100, chunk_overlap=10,
        )
        kb_id = kb["id"]

        await db_logger.create_document(kb_id=kb_id, name="a.txt", source_type="upload", source_uri=None, file_size_bytes=1)
        await db_logger.create_document(kb_id=kb_id, name="b.txt", source_type="upload", source_uri=None, file_size_bytes=1)
        docs = await db_logger.get_kb_documents(kb_id)
        assert len(docs) == 2

        await db_logger.delete_kb_documents(kb_id)
        docs = await db_logger.get_kb_documents(kb_id)
        assert docs == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, db_logger):
        result = await db_logger.get_document(999999)
        assert result is None

    # -----------------------------------------------------------------------
    # Action audit log
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_action_logging(self, db_logger):
        kb = await db_logger.create_knowledge_base(
            name="mykb3", description="d", tags=[], vector_db_type="chroma",
            deployment_mode="builtin", embedding_model_id="m", embedding_region=None,
            chunk_size=100, chunk_overlap=10,
        )
        kb_id = kb["id"]

        await db_logger.log_action(kb_id, "register", "user1", "registered new kb")
        await db_logger.log_action(kb_id, "update", "user2", None)
        actions = await db_logger.get_kb_actions(kb_id)
        assert len(actions) == 2
        # actions returned most-recent first
        assert actions[0]["action"] == "update"
        assert actions[0]["performed_by"] == "user2"
        assert actions[1]["action"] == "register"
        assert actions[1]["message"] == "registered new kb"

    # -----------------------------------------------------------------------
    # Data source & sync run operations
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_source_sync_operations(self, db_logger):
        kb = await db_logger.create_knowledge_base(
            name="mykb4", description="d", tags=[], vector_db_type="chroma",
            deployment_mode="builtin", embedding_model_id="m", embedding_region=None,
            chunk_size=100, chunk_overlap=10,
        )
        kb_id = kb["id"]

        # Create source
        source_id = await db_logger.create_data_source(
            kb_id=kb_id,
            source_type="web",
            display_name="My Website",
            config_encrypted="encrypted-config",
            config_public="public-config",
            sync_schedule=None,
        )
        assert source_id is not None

        # Get data source
        source = await db_logger.get_data_source(source_id)
        assert source["display_name"] == "My Website"
        assert source["sync_status"] == "never"

        # List KB sources
        sources = await db_logger.get_kb_data_sources(kb_id)
        assert len(sources) == 1
        assert sources[0]["id"] == source_id

        # Create sync run
        run_id = await db_logger.create_sync_run(source_id)
        assert run_id is not None

        runs = await db_logger.get_source_sync_runs(source_id)
        assert len(runs) == 1
        assert runs[0]["status"] == "running"

        # Update sync run
        await db_logger.update_sync_run(run_id, "success", document_count=5, error_message=None)
        runs = await db_logger.get_source_sync_runs(source_id)
        assert runs[0]["status"] == "success"
        assert runs[0]["document_count"] == 5

        # Update source sync status
        await db_logger.update_data_source_sync_status(source_id, "success", last_sync_error=None, document_count=5)
        source = await db_logger.get_data_source(source_id)
        assert source["sync_status"] == "success"
        assert source["document_count"] == 5

        # Delete source
        await db_logger.delete_data_source(source_id)
        source = await db_logger.get_data_source(source_id)
        assert source is None

    @pytest.mark.asyncio
    async def test_sync_run_with_error(self, db_logger):
        kb = await db_logger.create_knowledge_base(
            name="err_kb", description="d", tags=[], vector_db_type="chroma",
            deployment_mode="builtin", embedding_model_id="m", embedding_region=None,
            chunk_size=100, chunk_overlap=10,
        )
        source_id = await db_logger.create_data_source(
            kb_id=kb["id"], source_type="s3", display_name="S3",
            config_encrypted="{}", config_public="{}", sync_schedule=None,
        )
        run_id = await db_logger.create_sync_run(source_id)
        await db_logger.update_sync_run(run_id, "error", document_count=0, error_message="timeout")
        runs = await db_logger.get_source_sync_runs(source_id)
        assert runs[0]["status"] == "error"
        assert runs[0]["error_message"] == "timeout"

    @pytest.mark.asyncio
    async def test_get_nonexistent_data_source(self, db_logger):
        result = await db_logger.get_data_source("nonexistent-uuid")
        assert result is None

    # -----------------------------------------------------------------------
    # Disabled logging mode (no backend)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_logging_when_disabled(self, tmp_path):
        db_file = tmp_path / "disabled_test.db"
        with patch.dict(os.environ, {
            "REGISTRY_DB_LOGGING_ENABLED": "false",
            "SQLITE_DB_PATH": str(db_file),
        }):
            logger = KBDatabaseLogger(logger=logging.getLogger("test_disabled"))
            initialized = await logger.initialize()
            assert initialized is False
            assert logger._ready() is False

            # All read methods should return safe empty defaults
            assert await logger.get_knowledge_base("name") is None
            assert await logger.get_all_knowledge_bases() == []
            assert await logger.get_kb_configs(1) == {}
            assert await logger.get_document(1) is None
            assert await logger.get_kb_documents(1) == []
            assert await logger.get_kb_actions(1) == []
            assert await logger.get_data_source("s") is None
            assert await logger.get_kb_data_sources(1) == []
            assert await logger.get_source_sync_runs("s") == []

            # create_knowledge_base returns stub
            res = await logger.create_knowledge_base("name", None, [], "t", "m", "em", None, 1, 1)
            assert res["id"] is None
            assert res["name"] == "name"

            # create_document returns stub
            doc = await logger.create_document(1, "f.txt", "upload", None, 0)
            assert doc["id"] is None

            # create_data_source returns a UUID string even when disabled
            sid = await logger.create_data_source(1, "web", "X", "{}", "{}")
            assert isinstance(sid, str)

            # create_sync_run returns a UUID string even when disabled
            rid = await logger.create_sync_run("source-id")
            assert isinstance(rid, str)

            # All mutating methods are safe no-ops
            await logger.update_kb_status("name", "status")
            await logger.upsert_kb_config(1, "k", "v")
            await logger.delete_kb_configs(1)
            await logger.delete_knowledge_base("name")
            await logger.update_document_status(1, "s")
            await logger.delete_document(1)
            await logger.delete_kb_documents(1)
            await logger.log_action(1, "a", "u", "m")
            await logger.update_data_source_sync_status("s", "s")
            await logger.delete_data_source("s")
            await logger.update_sync_run("r", "s")

    # -----------------------------------------------------------------------
    # close() when no backend
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close_when_no_backend(self):
        logger = KBDatabaseLogger()
        # Should not raise
        await logger.close()

    # -----------------------------------------------------------------------
    # initialize() when both backends fail
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialize_all_backends_fail(self, tmp_path):
        with patch.dict(os.environ, {"REGISTRY_DB_LOGGING_ENABLED": "true"}):
            with patch(
                "oai_kb_registry.services.db.database_logger.PostgresBackend.initialize",
                new_callable=AsyncMock,
                return_value=False,
            ), patch(
                "oai_kb_registry.services.db.database_logger.SQLiteBackend.initialize",
                new_callable=AsyncMock,
                return_value=False,
            ):
                logger = KBDatabaseLogger(logger=logging.getLogger("test_fail"))
                result = await logger.initialize()
                assert result is False
                assert logger._ready() is False

    # -----------------------------------------------------------------------
    # Tags encoding helpers (encode_tags for postgres path)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_encode_tags_sqlite(self, db_logger):
        # SQLite backend: tags should be JSON-encoded string
        encoded = db_logger._encode_tags(["a", "b"])
        assert isinstance(encoded, str)
        assert "a" in encoded

    @pytest.mark.asyncio
    async def test_encode_tags_empty(self, db_logger):
        encoded = db_logger._encode_tags([])
        assert encoded == "[]"

    @pytest.mark.asyncio
    async def test_decode_tags_invalid_json(self, db_logger):
        row = {"tags": "not-valid-json", "name": "x"}
        result = db_logger._decode_tags(row)
        assert result["tags"] == []

    @pytest.mark.asyncio
    async def test_decode_tags_empty_string(self, db_logger):
        row = {"tags": "", "name": "x"}
        result = db_logger._decode_tags(row)
        assert result["tags"] == []

    # -----------------------------------------------------------------------
    # Exception paths (backend raises)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_kb_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("DB error"))
        result = await db_logger.create_knowledge_base(
            "errname", None, [], "chroma", "builtin", "m", None, 100, 10
        )
        assert result == {"id": None, "name": "errname"}

    @pytest.mark.asyncio
    async def test_get_kb_exception_path(self, db_logger):
        db_logger._backend.fetch_one = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_knowledge_base("x")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_kbs_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_all_knowledge_bases()
        assert result == []

    @pytest.mark.asyncio
    async def test_update_kb_status_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        # Should not raise
        await db_logger.update_kb_status("name", "x")

    @pytest.mark.asyncio
    async def test_delete_kb_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.delete_knowledge_base("x")

    @pytest.mark.asyncio
    async def test_upsert_kb_config_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.upsert_kb_config(1, "k", "v")

    @pytest.mark.asyncio
    async def test_get_kb_configs_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_kb_configs(1)
        assert result == {}

    @pytest.mark.asyncio
    async def test_delete_kb_configs_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.delete_kb_configs(1)

    @pytest.mark.asyncio
    async def test_create_document_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.create_document(1, "f.txt", "upload", None, 0)
        assert result == {"id": None, "name": "f.txt"}

    @pytest.mark.asyncio
    async def test_update_document_status_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.update_document_status(1, "error")

    @pytest.mark.asyncio
    async def test_get_document_exception_path(self, db_logger):
        db_logger._backend.fetch_one = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_document(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_kb_documents_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_kb_documents(1)
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_document_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.delete_document(1)

    @pytest.mark.asyncio
    async def test_delete_kb_documents_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.delete_kb_documents(1)

    @pytest.mark.asyncio
    async def test_log_action_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.log_action(1, "a", "u", "m")

    @pytest.mark.asyncio
    async def test_get_kb_actions_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_kb_actions(1)
        assert result == []

    @pytest.mark.asyncio
    async def test_create_data_source_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        sid = await db_logger.create_data_source(1, "web", "X", "{}", "{}")
        # Should still return a UUID string (generated before the error)
        assert isinstance(sid, str)

    @pytest.mark.asyncio
    async def test_get_data_source_exception_path(self, db_logger):
        db_logger._backend.fetch_one = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_data_source("x")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_kb_data_sources_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_kb_data_sources(1)
        assert result == []

    @pytest.mark.asyncio
    async def test_update_data_source_sync_status_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.update_data_source_sync_status("s", "success")

    @pytest.mark.asyncio
    async def test_delete_data_source_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.delete_data_source("s")

    @pytest.mark.asyncio
    async def test_create_sync_run_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        rid = await db_logger.create_sync_run("source")
        assert isinstance(rid, str)

    @pytest.mark.asyncio
    async def test_update_sync_run_exception_path(self, db_logger):
        db_logger._backend.execute = AsyncMock(side_effect=Exception("fail"))
        await db_logger.update_sync_run("run", "success")

    @pytest.mark.asyncio
    async def test_get_source_sync_runs_exception_path(self, db_logger):
        db_logger._backend.fetch = AsyncMock(side_effect=Exception("fail"))
        result = await db_logger.get_source_sync_runs("source")
        assert result == []
