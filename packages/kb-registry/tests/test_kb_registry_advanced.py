"""
Advanced unit tests for KBRegistry.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from oai_kb_registry.services.kb_registry import KBRegistry


@pytest.mark.asyncio
class TestKBRegistryAdvanced:
    """Advanced unit tests for KBRegistry."""

    @pytest.fixture
    def mock_db_logger(self):
        db = AsyncMock()
        db.get_knowledge_base = AsyncMock(return_value=None)
        db.get_kb_configs = AsyncMock(return_value={})
        db.create_knowledge_base = AsyncMock(return_value={"id": 1, "name": "testkb"})
        db.upsert_kb_config = AsyncMock()
        db.log_action = AsyncMock()
        db.update_kb_status = AsyncMock()
        db.delete_knowledge_base = AsyncMock()
        db.create_document = AsyncMock(return_value={"id": 1})
        db.update_document_status = AsyncMock()
        db.delete_document = AsyncMock()
        db.get_document = AsyncMock(return_value={"id": 1, "kb_id": 1, "name": "doc"})
        db.get_kb_documents = AsyncMock(return_value=[])
        db.close = AsyncMock()
        return db

    @pytest.fixture
    def registry(self, mock_db_logger):
        return KBRegistry(db_logger=mock_db_logger)

    async def test_close(self, registry, mock_db_logger):
        # Setup mock vector services started
        registry._vector_services_started.add("pgvector")

        with patch.object(registry, "_stop_vector_store_infra", new_callable=AsyncMock) as mock_stop_infra:
            await registry.close()
            mock_stop_infra.assert_called_once()
            mock_db_logger.close.assert_called_once()

    async def test_get_or_create_vector_store_cached(self, registry):
        mock_vs = MagicMock()
        registry._vector_stores["testkb"] = mock_vs
        res = await registry._get_or_create_vector_store("testkb")
        assert res == mock_vs

    async def test_get_or_create_vector_store_not_found(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = None
        with pytest.raises(ValueError, match="Knowledge base not found"):
            await registry._get_or_create_vector_store("missingkb")

    async def test_get_or_create_vector_store_success(self, registry, mock_db_logger):
        kb_info = {
            "id": 1,
            "name": "testkb",
            "vector_db_type": "postgres",
            "deployment_mode": "builtin",
            "embedding_model_id": "model-id",
            "embedding_region": "us-east-1"
        }
        mock_db_logger.get_knowledge_base.return_value = kb_info
        mock_db_logger.get_kb_configs.return_value = {"DB_PORT": "5435"}

        mock_vs = MagicMock()
        with patch("oai_kb_registry.services.provider_factory.VectorStoreProviderFactory.decrypt_configs", return_value={"DB_PORT": "5435"}), \
             patch("oai_kb_registry.services.provider_factory.VectorStoreProviderFactory.create_vector_store", return_value=mock_vs):
            res = await registry._get_or_create_vector_store("testkb")
            assert res == mock_vs
            assert registry._vector_stores["testkb"] == mock_vs

    async def test_ensure_vector_store_infra_external(self, registry):
        # external deployment is a no-op
        with patch("oai_kb_registry.services.infra_manager.InfraManager.ensure_vector_store_service", new_callable=AsyncMock) as mock_ensure:
            await registry._ensure_vector_store_infra("postgres", "external")
            mock_ensure.assert_not_called()

    async def test_ensure_vector_store_infra_s3(self, registry):
        # s3 vector type is a no-op
        with patch("oai_kb_registry.services.infra_manager.InfraManager.ensure_vector_store_service", new_callable=AsyncMock) as mock_ensure:
            await registry._ensure_vector_store_infra("s3", "builtin")
            mock_ensure.assert_not_called()

    async def test_ensure_vector_store_infra_builtin_success(self, registry):
        registry._infra_compose_file = "docker-compose.yaml"
        registry._infra_startup_timeout = 90
        with patch("oai_kb_registry.services.infra_manager.InfraManager.ensure_vector_store_service", new_callable=AsyncMock) as mock_ensure:
            await registry._ensure_vector_store_infra("postgres", "builtin")
            mock_ensure.assert_called_once()
            assert "pgvector" in registry._vector_services_started

    async def test_stop_vector_store_infra_success(self, registry):
        registry._vector_services_started.add("pgvector")
        with patch("oai_kb_registry.services.infra_manager.InfraManager.stop_services") as mock_stop:
            await registry._stop_vector_store_infra()
            assert len(registry._vector_services_started) == 0

    async def test_restore_from_db_success(self, registry, mock_db_logger):
        mock_db_logger.get_all_knowledge_bases = AsyncMock(return_value=[
            {
                "name": "kb1",
                "vector_db_type": "postgres",
                "deployment_mode": "builtin",
                "embedding_model_id": "model-id",
                "embedding_region": "us-east-1"
            }
        ])

        # Enable auto_start_infra
        registry._auto_start_infra = True

        with patch.object(registry, "_ensure_vector_store_infra", new_callable=AsyncMock) as mock_ensure, \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock) as mock_get_or_create:
            await registry.restore_from_db()
            mock_ensure.assert_called_once()
            mock_get_or_create.assert_called_once()

    async def test_register_knowledge_base_success(self, registry, mock_db_logger):
        kb_info = {
            "id": 1,
            "name": "testkb",
            "vector_db_type": "postgres",
            "deployment_mode": "builtin",
            "embedding_model_id": "model-id"
        }
        mock_db_logger.create_knowledge_base.return_value = kb_info
        
        with patch("oai_kb_registry.services.provider_factory.VectorStoreProviderFactory.encrypt_configs", return_value={"port": "1"}), \
             patch.object(registry, "_ensure_vector_store_infra", new_callable=AsyncMock) as mock_ensure, \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock) as mock_get_or_create:
            
            res = await registry.register_knowledge_base(
                name="testkb",
                description="desc",
                tags=["tag"],
                vector_db_type="postgres",
                deployment_mode="builtin",
                embedding_model_id="model-id",
                chunk_size=100,
                chunk_overlap=10,
                vector_db_config={"port": "1"},
                performed_by="user"
            )
            assert res == kb_info
            mock_ensure.assert_called_once()
            mock_get_or_create.assert_called_once()

    async def test_register_knowledge_base_warmup_error(self, registry, mock_db_logger):
        kb_info = {
            "id": 1,
            "name": "testkb",
            "vector_db_type": "postgres",
            "deployment_mode": "builtin",
            "embedding_model_id": "model-id"
        }
        mock_db_logger.create_knowledge_base.return_value = kb_info

        with patch.object(registry, "_ensure_vector_store_infra", new_callable=AsyncMock) as mock_ensure, \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, side_effect=RuntimeError("warmup failed")) as mock_get_or_create:
            
            res = await registry.register_knowledge_base(
                name="testkb",
                description="desc",
                tags=["tag"],
                vector_db_type="postgres",
                deployment_mode="builtin",
                embedding_model_id="model-id",
                chunk_size=100,
                chunk_overlap=10,
                vector_db_config=None,
                performed_by="user"
            )
            assert res == kb_info
            mock_ensure.assert_called_once()
            mock_get_or_create.assert_called_once()
            mock_db_logger.update_kb_status.assert_called_once_with("testkb", "error")

    async def test_delete_knowledge_base_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb"}
        mock_vs = MagicMock()
        
        with patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs):
            await registry.delete_knowledge_base("testkb")
            mock_vs.reset_collection.assert_called_once()
            mock_db_logger.delete_knowledge_base.assert_called_once_with("testkb")

    async def test_delete_knowledge_base_not_found(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = None
        with pytest.raises(ValueError, match="Knowledge base not found"):
            await registry.delete_knowledge_base("missing")

    async def test_assert_ingestable_neo4j(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"vector_db_type": "neo4j"}
        with pytest.raises(ValueError, match="Document ingestion is not supported"):
            await registry._assert_ingestable("neo4jkb")

    async def test_index_document_from_upload_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {
            "id": 1,
            "chunk_size": 100,
            "chunk_overlap": 10
        }
        mock_vs = MagicMock()
        mock_processor = MagicMock()
        mock_processor.process_upload.return_value = ["doc1", "doc2"]

        with patch("oai_kb_registry.services.kb_registry.DocumentProcessor", return_value=mock_processor), \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs), \
             patch.object(registry, "_index_documents_in_thread", new_callable=AsyncMock, return_value=2) as mock_index:
            
            res = await registry.index_document_from_upload(
                kb_name="testkb",
                filename="doc.txt",
                data=b"hello",
                performed_by="user"
            )
            assert res["id"] == 1
            mock_index.assert_called_once()
            mock_db_logger.update_document_status.assert_called_with(1, "indexed", 2)

    async def test_index_document_from_s3_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {
            "id": 1,
            "chunk_size": 100,
            "chunk_overlap": 10
        }
        mock_vs = MagicMock()
        mock_processor = MagicMock()
        mock_processor.process_s3_uri.return_value = ["doc1"]

        with patch("oai_kb_registry.services.kb_registry.DocumentProcessor", return_value=mock_processor), \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs), \
             patch.object(registry, "_index_documents_in_thread", new_callable=AsyncMock, return_value=1) as mock_index:
            
            res = await registry.index_document_from_s3(
                kb_name="testkb",
                bucket="mybucket",
                key="doc.txt",
                performed_by="user"
            )
            assert res["id"] == 1
            mock_index.assert_called_once()

    async def test_index_document_from_text_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {
            "id": 1,
            "chunk_size": 100,
            "chunk_overlap": 10
        }
        mock_vs = MagicMock()
        mock_processor = MagicMock()
        mock_processor.process_text.return_value = ["doc1"]

        with patch("oai_kb_registry.services.kb_registry.DocumentProcessor", return_value=mock_processor), \
             patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs), \
             patch.object(registry, "_index_documents_in_thread", new_callable=AsyncMock, return_value=1) as mock_index:
            
            res = await registry.index_document_from_text(
                kb_name="testkb",
                name="doc.txt",
                text="content",
                extra_metadata={"tag": "v1"},
                performed_by="user"
            )
            assert res["id"] == 1
            mock_index.assert_called_once()

    async def test_remove_document_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb"}
        mock_db_logger.get_document.return_value = {"id": 2, "kb_id": 1, "name": "doc"}

        await registry.remove_document("testkb", 2)
        mock_db_logger.delete_document.assert_called_once_with(2)

    async def test_remove_document_not_found(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb"}
        mock_db_logger.get_document.return_value = None
        with pytest.raises(ValueError, match="not found in KB"):
            await registry.remove_document("testkb", 999)

    async def test_remove_document_kb_mismatch(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb"}
        mock_db_logger.get_document.return_value = {"id": 2, "kb_id": 2, "name": "doc"} # kb_id=2
        with pytest.raises(ValueError, match="not found in KB"):
            await registry.remove_document("testkb", 2)

    async def test_query_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb"}
        mock_vs = MagicMock()
        
        mock_doc = MagicMock()
        mock_doc.page_content = "hello"
        mock_doc.metadata = {"doc_name": "a.txt", "doc_id": 2}
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]

        with patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs):
            res = await registry.query("testkb", "query text", k=5, score_threshold=0.8, filter_metadata={"tag": "v1"})
            assert len(res) == 1
            assert res[0]["content"] == "hello"
            mock_vs.similarity_search_with_score.assert_called_once_with(
                "query text",
                k=5,
                filter={"tag": "v1"}
            )

    async def test_graph_stats_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb", "vector_db_type": "neo4j"}
        mock_vs = MagicMock()
        mock_vs.graph_stats.return_value = {"nodes": 10}

        with patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs):
            res = await registry.graph_stats("testkb", sample_limit=5)
            assert res == {"nodes": 10}

    async def test_reindex_knowledge_base_success(self, registry, mock_db_logger):
        mock_db_logger.get_knowledge_base.return_value = {"id": 1, "name": "testkb", "chunk_size": 100, "chunk_overlap": 10}
        mock_db_logger.get_kb_documents.return_value = [{"id": 2, "status": "indexed", "source_type": "upload"}]
        mock_vs = MagicMock()

        with patch.object(registry, "_get_or_create_vector_store", new_callable=AsyncMock, return_value=mock_vs), \
             patch.object(registry, "_reindex_all_documents", new_callable=AsyncMock) as mock_reindex:
            res = await registry.reindex_knowledge_base("testkb")
            assert res == 1
            mock_reindex.assert_called_once()
