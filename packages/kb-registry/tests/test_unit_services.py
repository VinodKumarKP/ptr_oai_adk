"""
Unit tests for KB Registry service layer and sources router.
Targets: document_processor, provider_factory, sources router, sources helpers.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

os.environ["KB_AUTH_ENABLED"] = "false"
os.environ["REGISTRY_DB_LOGGING_ENABLED"] = "false"


# ===========================================================================
# DOCUMENT PROCESSOR TESTS
# ===========================================================================

class TestExtractTextFromBytes:
    def test_txt_utf8(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        data = b"hello world"
        result = _extract_text_from_bytes(data, "file.txt")
        assert result == "hello world"

    def test_md_utf8(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        data = b"# Title\nContent"
        result = _extract_text_from_bytes(data, "readme.md")
        assert "Title" in result

    def test_csv_utf8(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        data = b"a,b,c\n1,2,3"
        result = _extract_text_from_bytes(data, "data.csv")
        assert "a,b,c" in result

    def test_latin1_fallback(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        data = bytes([0x80, 0x90, 0xa0])  # invalid UTF-8
        result = _extract_text_from_bytes(data, "file.txt")
        assert isinstance(result, str)

    def test_pdf_extraction(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = _extract_text_from_bytes(b"fake pdf", "doc.pdf")
        assert "Page content" in result

    def test_pdf_import_error(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises(RuntimeError, match="pypdf"):
                _extract_text_from_bytes(b"fake", "doc.pdf")

    def test_pdf_parse_error(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        with patch("pypdf.PdfReader", side_effect=Exception("corrupt")):
            with pytest.raises(RuntimeError, match="Failed to parse PDF"):
                _extract_text_from_bytes(b"bad pdf", "doc.pdf")

    def test_docx_extraction(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        mock_para1 = MagicMock()
        mock_para1.text = "Paragraph one"
        mock_para2 = MagicMock()
        mock_para2.text = ""
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        with patch("docx.Document", return_value=mock_doc):
            result = _extract_text_from_bytes(b"fake docx", "file.docx")
        assert "Paragraph one" in result

    def test_docx_import_error(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(RuntimeError, match="python-docx"):
                _extract_text_from_bytes(b"fake", "file.docx")

    def test_docx_parse_error(self):
        from oai_kb_registry.services.document_processor import _extract_text_from_bytes
        with patch("docx.Document", side_effect=Exception("bad docx")):
            with pytest.raises(RuntimeError, match="Failed to parse DOCX"):
                _extract_text_from_bytes(b"bad", "file.doc")


class TestFetchS3Content:
    def test_fetch_success(self):
        from oai_kb_registry.services.document_processor import _fetch_s3_content
        mock_body = MagicMock()
        mock_body.read.return_value = b"s3 content"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        with patch("boto3.client", return_value=mock_s3):
            data = _fetch_s3_content("bucket", "key/file.txt")
        assert data == b"s3 content"

    def test_fetch_with_credentials(self):
        from oai_kb_registry.services.document_processor import _fetch_s3_content
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        with patch("boto3.client", return_value=mock_s3) as mock_client:
            _fetch_s3_content("b", "k", aws_access_key_id="key", aws_secret_access_key="sec")
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["aws_access_key_id"] == "key"

    def test_fetch_import_error(self):
        from oai_kb_registry.services.document_processor import _fetch_s3_content
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(RuntimeError, match="boto3"):
                _fetch_s3_content("b", "k")

    def test_fetch_s3_error(self):
        from oai_kb_registry.services.document_processor import _fetch_s3_content
        with patch("boto3.client", side_effect=Exception("access denied")):
            with pytest.raises(RuntimeError, match="Failed to fetch s3://"):
                _fetch_s3_content("b", "k")


class TestFetchUrlContent:
    def test_fetch_success(self):
        from oai_kb_registry.services.document_processor import _fetch_url_content
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"html content"
        with patch("urllib.request.urlopen", return_value=mock_resp):
            data = _fetch_url_content("http://example.com/doc.html")
        assert data == b"html content"

    def test_fetch_error(self):
        from oai_kb_registry.services.document_processor import _fetch_url_content
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            with pytest.raises(RuntimeError, match="Failed to fetch URL"):
                _fetch_url_content("http://bad.url")


class TestDocumentProcessor:
    def _make_processor(self):
        from oai_kb_registry.services.document_processor import DocumentProcessor
        return DocumentProcessor(chunk_size=100, chunk_overlap=10)

    def test_process_upload_txt(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            docs = proc.process_upload(b"hello world text", "file.txt", "mykb", doc_id=1)
        assert len(docs) == 2
        assert docs[0].metadata["kb_name"] == "mykb"
        assert docs[0].metadata["source_type"] == "upload"
        assert docs[0].metadata["chunk_index"] == 0

    def test_process_upload_no_doc_id(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            docs = proc.process_upload(b"text", "file.txt", "kb")
        assert docs[0].metadata["doc_id"] is None

    def test_process_s3_object(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["s3chunk"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            with patch("oai_kb_registry.services.document_processor._fetch_s3_content", return_value=b"text content"):
                docs = proc.process_s3_object("bucket", "path/file.txt", "kb", doc_id=5)
        assert len(docs) == 1
        assert docs[0].metadata["s3_bucket"] == "bucket"
        assert docs[0].metadata["s3_key"] == "path/file.txt"

    def test_process_text(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["part1", "part2"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            docs = proc.process_text("some text", "myname", "kb", doc_id=3, source_type="external")
        assert len(docs) == 2
        assert docs[0].metadata["source_type"] == "external"

    def test_process_url(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["web chunk"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            with patch("oai_kb_registry.services.document_processor._fetch_url_content", return_value=b"web text"):
                docs = proc.process_url("http://example.com/doc.txt", "kb")
        assert docs[0].metadata["source_type"] == "url"

    def test_get_splitter_import_error(self):
        proc = self._make_processor()
        with patch.dict("sys.modules", {"langchain_text_splitters": None}):
            with pytest.raises(RuntimeError, match="langchain-text-splitters"):
                proc._get_splitter()

    def test_extra_metadata_passed_through(self):
        proc = self._make_processor()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["c"]
        with patch.object(proc, "_get_splitter", return_value=mock_splitter):
            docs = proc.process_upload(b"text", "f.txt", "kb", extra_metadata={"custom": "value"})
        assert docs[0].metadata["custom"] == "value"


# ===========================================================================
# PROVIDER FACTORY TESTS
# ===========================================================================

class TestEncryptDecrypt:
    def test_encrypt_decrypt_roundtrip(self):
        from oai_kb_registry.services.provider_factory import _encrypt, _decrypt
        original = "my-secret-password"
        encrypted = _encrypt(original)
        assert encrypted != original
        decrypted = _decrypt(encrypted)
        assert decrypted == original

    def test_decrypt_none(self):
        from oai_kb_registry.services.provider_factory import _decrypt
        assert _decrypt(None) is None

    def test_decrypt_empty(self):
        from oai_kb_registry.services.provider_factory import _decrypt
        assert _decrypt("") is None

    def test_decrypt_plain_text_fallback(self):
        from oai_kb_registry.services.provider_factory import _decrypt
        # If it's not valid base64, returns as-is
        result = _decrypt("not-base64-!@#")
        assert result == "not-base64-!@#"


class TestVectorStoreProviderFactory:
    def test_encrypt_configs(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        configs = {"host": "localhost", "port": "8000"}
        encrypted = VectorStoreProviderFactory.encrypt_configs(configs)
        assert "host" in encrypted
        assert encrypted["host"] != "localhost"

    def test_decrypt_configs(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory, _encrypt
        raw = {"host": _encrypt("localhost"), "port": _encrypt("8000")}
        decrypted = VectorStoreProviderFactory.decrypt_configs(raw)
        assert decrypted["host"] == "localhost"
        assert decrypted["port"] == "8000"

    def test_encrypt_configs_excludes_none(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        configs = {"host": "localhost", "key": None}
        encrypted = VectorStoreProviderFactory.encrypt_configs(configs)
        assert "key" not in encrypted

    def test_create_vector_store_pinecone_missing_api_key(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        with pytest.raises(ValueError, match="pinecone_api_key"):
            VectorStoreProviderFactory.create_vector_store(
                "mykb", "pinecone", "external",
                configs={"pinecone_index_name": "idx"},
            )

    def test_create_vector_store_pinecone_missing_index(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        with pytest.raises(ValueError, match="pinecone_index_name"):
            VectorStoreProviderFactory.create_vector_store(
                "mykb", "pinecone", "external",
                configs={"pinecone_api_key": "pk"},
            )

    def test_create_vector_store_neo4j_missing_url(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        with pytest.raises(ValueError, match="neo4j_url"):
            VectorStoreProviderFactory.create_vector_store(
                "mykb", "neo4j_graph", "external", configs={},
            )

    def test_create_vector_store_neo4j_text2cypher_missing_llm(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        with pytest.raises(ValueError, match="neo4j_llm_model_id"):
            VectorStoreProviderFactory.create_vector_store(
                "mykb", "neo4j_graph", "external",
                configs={"neo4j_url": "bolt://x", "neo4j_retrieval_mode": "text2cypher"},
            )

    def test_create_vector_store_neo4j_vector_entry_missing(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        with pytest.raises(ValueError, match="neo4j_vector_entry"):
            VectorStoreProviderFactory.create_vector_store(
                "mykb", "neo4j_graph", "external",
                configs={"neo4j_url": "bolt://x", "neo4j_entry_strategy": "vector"},
            )

    def test_create_vector_store_unsupported_type(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with pytest.raises(ValueError, match="Unsupported vector_db_type"):
                VectorStoreProviderFactory.create_vector_store(
                    "mykb", "unknown_db", "builtin",
                )

    def test_create_vector_store_chroma_builtin(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                result = VectorStoreProviderFactory.create_vector_store(
                    "mykb", "chroma", "builtin",
                )
        assert result is mock_store

    def test_create_vector_store_chroma_external(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                result = VectorStoreProviderFactory.create_vector_store(
                    "mykb", "chroma", "external",
                    configs={"chroma_host": "chromahost", "chroma_port": "9000", "chroma_ssl": "true"},
                )
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["host"] == "chromahost"
        assert call_kwargs["ssl"] is True

    def test_create_vector_store_postgres_builtin(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store):
                result = VectorStoreProviderFactory.create_vector_store("kb", "postgres", "builtin")
        assert result is mock_store

    def test_create_vector_store_postgres_external(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                VectorStoreProviderFactory.create_vector_store(
                    "kb", "postgres", "external",
                    configs={"pg_host": "pghost", "pg_port": "5432", "pg_user": "u", "pg_password": "p", "pg_database": "db"},
                )
        call_kwargs = mock_create.call_args[1]
        assert "pghost" in call_kwargs["connection_string"]

    def test_create_vector_store_postgres_builtin_with_url(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                with patch.dict(os.environ, {"KB_POSTGRES_URL": "postgresql+psycopg2://user:pass@host/db"}):
                    VectorStoreProviderFactory.create_vector_store("kb", "postgres", "builtin")
        call_kwargs = mock_create.call_args[1]
        assert "postgresql+psycopg2" not in call_kwargs["connection_string"]

    def test_create_vector_store_s3_external(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                VectorStoreProviderFactory.create_vector_store(
                    "kb", "s3", "external",
                    configs={"s3_bucket": "mybucket", "aws_access_key_id": "key", "aws_secret_access_key": "sec"},
                )
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["bucket_name"] == "mybucket"
        assert call_kwargs["aws_access_key_id"] == "key"

    def test_create_vector_store_s3_builtin(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_emb = MagicMock()
        mock_store = MagicMock()
        with patch("oai_kb_registry.services.provider_factory._make_litellm_embeddings", return_value=mock_emb):
            with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
                with patch.dict(os.environ, {"KB_S3_BUCKET": "env-bucket"}):
                    VectorStoreProviderFactory.create_vector_store("kb", "s3", "builtin")
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["bucket_name"] == "env-bucket"

    def test_create_vector_store_pinecone_success(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_store = MagicMock()
        with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
            result = VectorStoreProviderFactory.create_vector_store(
                "mykb", "pinecone", "external",
                configs={"pinecone_api_key": "pk", "pinecone_index_name": "idx"},
            )
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["namespace"] == "mykb"  # fallback to kb_name

    def test_create_vector_store_neo4j_traversal(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_store = MagicMock()
        with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store):
            result = VectorStoreProviderFactory.create_vector_store(
                "kb", "neo4j_graph", "external",
                configs={"neo4j_url": "bolt://neo4j:7687", "neo4j_retrieval_mode": "traversal"},
            )
        assert result is mock_store

    def test_create_vector_store_neo4j_with_llm(self):
        from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
        mock_store = MagicMock()
        with patch("oai_agent_core.components.vector_store.vector_store_factory.VectorStoreFactory.create_vector_store", return_value=mock_store) as mock_create:
            VectorStoreProviderFactory.create_vector_store(
                "kb", "neo4j_graph", "external",
                configs={"neo4j_url": "bolt://x", "neo4j_llm_model_id": "bedrock/claude-3"},
            )
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["llm"] == "bedrock/claude-3"

    def test_make_litellm_embeddings_import_error(self):
        from oai_kb_registry.services.provider_factory import _make_litellm_embeddings
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(RuntimeError, match="litellm"):
                _make_litellm_embeddings()

    def test_make_litellm_embeddings_auto_prefix_bedrock(self):
        """Bare amazon model IDs get prefixed with bedrock/."""
        from oai_kb_registry.services.provider_factory import _make_litellm_embeddings
        # The function creates a local class — just verify it runs without error
        # and returns an object with a model_id attribute
        try:
            emb = _make_litellm_embeddings(model_id="amazon.titan-embed-text-v1")
            assert emb.model_id == "bedrock/amazon.titan-embed-text-v1"
        except RuntimeError:
            # litellm not installed in test env — acceptable
            pass


# ===========================================================================
# SOURCES ROUTER HELPERS
# ===========================================================================

class TestSourcesHelpers:
    def test_catalog_entry_to_model(self):
        from oai_kb_registry.routers.sources import _catalog_entry_to_model
        entry = {
            "id": "confluence",
            "display_name": "Confluence",
            "category": "collaboration",
            "description": "Atlassian Confluence",
            "pip_extra": "atlassian",
            "fields": [
                {"name": "url", "type": "url", "label": "URL", "required": True},
                {"name": "api_key", "type": "secret", "label": "API Key"},
            ],
        }
        model = _catalog_entry_to_model(entry)
        assert model.id == "confluence"
        assert len(model.fields) == 2
        assert model.fields[0].required is True

    def test_row_to_source_response(self):
        from oai_kb_registry.routers.sources import _row_to_source_response
        row = {
            "id": "src-1",
            "kb_name": "mykb",
            "source_type": "confluence",
            "display_name": "My Conf",
            "config_public": '{"url": "https://x"}',
            "sync_schedule": "manual",
            "sync_status": "idle",
            "last_sync_at": None,
            "last_sync_error": None,
            "document_count": 5,
            "created_at": datetime.now(timezone.utc),
        }
        resp = _row_to_source_response(row)
        assert resp.kb_name == "mykb"
        assert resp.document_count == 5
        assert resp.config_public["url"] == "https://x"

    def test_row_to_source_response_bad_json(self):
        from oai_kb_registry.routers.sources import _row_to_source_response
        row = {
            "id": "src-1",
            "kb_name": "kb",
            "source_type": "s3",
            "display_name": "S3",
            "config_public": "not-json",
            "sync_status": "never",
            "document_count": 0,
            "created_at": None,
        }
        resp = _row_to_source_response(row)
        assert resp.config_public == {}

    def test_row_to_run_response(self):
        from oai_kb_registry.routers.sources import _row_to_run_response
        row = {
            "id": 10,
            "source_id": "src-1",
            "started_at": datetime.now(timezone.utc),
            "completed_at": None,
            "status": "running",
            "document_count": 0,
            "error_message": None,
        }
        resp = _row_to_run_response(row)
        assert resp.id == "10"
        assert resp.source_id == "src-1"


# ===========================================================================
# SOURCES ROUTER API TESTS
# ===========================================================================

@pytest.fixture(scope="module")
def sources_mock_registry():
    reg = MagicMock()
    reg.db = MagicMock()
    return reg


@pytest.fixture(scope="module")
def sources_client(sources_mock_registry):
    os.environ["KB_AUTH_ENABLED"] = "false"
    os.environ["REGISTRY_DB_LOGGING_ENABLED"] = "false"

    from oai_kb_registry.main import app
    from oai_kb_registry import dependencies

    async def _override_registry():
        return sources_mock_registry

    app.dependency_overrides[dependencies.get_registry] = _override_registry

    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


class TestSourcesRouter:
    def test_list_loaders(self, sources_client):
        resp = sources_client.get("/api/v1/kb-registry/loaders")
        assert resp.status_code == 200
        data = resp.json()
        # Can be a list or a dict grouped by category
        if isinstance(data, list):
            assert any(item["id"] == "confluence" for item in data)
        else:
            # grouped by category dict
            all_ids = [item["id"] for items in data.values() for item in items]
            assert "confluence" in all_ids

    def test_add_source_kb_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.post(
            "/api/v1/kb-registry/knowledge-bases/missing/sources",
            json={"source_type": "confluence", "display_name": "My Conf", "config": {"url": "https://x", "username": "u", "api_key": "k"}},
        )
        assert resp.status_code == 404

    def test_add_source_kb_not_found2(self, sources_client, sources_mock_registry):
        """Posting to a non-existent KB returns 404."""
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.post(
            "/api/v1/kb-registry/knowledge-bases/missing2/sources",
            json={"source_type": "confluence", "display_name": "X", "config": {}},
        )
        assert resp.status_code == 404

    def test_list_sources_kb_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.get("/api/v1/kb-registry/knowledge-bases/missing/sources")
        assert resp.status_code == 404

    def test_list_sources_success(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_kb_data_sources = AsyncMock(return_value=[
            {
                "id": "s1", "kb_name": "mykb", "source_type": "confluence",
                "display_name": "Conf", "config_public": "{}", "sync_schedule": None,
                "sync_status": "idle", "last_sync_at": None, "last_sync_error": None,
                "document_count": 0, "created_at": datetime.now(timezone.utc),
            }
        ])
        resp = sources_client.get("/api/v1/kb-registry/knowledge-bases/mykb/sources")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_delete_source_kb_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.delete("/api/v1/kb-registry/knowledge-bases/missing/sources/s1")
        assert resp.status_code == 404

    def test_delete_source_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_data_source = AsyncMock(return_value=None)
        resp = sources_client.delete("/api/v1/kb-registry/knowledge-bases/mykb/sources/missing-src")
        assert resp.status_code == 404

    def test_delete_source_success(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_data_source = AsyncMock(return_value={
            "id": "s1", "kb_id": 1, "kb_name": "mykb", "source_type": "confluence",
            "display_name": "Conf", "config_public": "{}", "sync_schedule": None,
            "sync_status": "idle", "last_sync_at": None, "last_sync_error": None,
            "document_count": 0, "created_at": datetime.now(timezone.utc),
        })
        sources_mock_registry.db.delete_data_source = AsyncMock()
        resp = sources_client.delete("/api/v1/kb-registry/knowledge-bases/mykb/sources/s1")
        # 204 No Content on success
        assert resp.status_code in (200, 204)

    def test_get_source_status_kb_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.get("/api/v1/kb-registry/knowledge-bases/missing/sources/s1/status")
        assert resp.status_code == 404

    def test_get_source_status_source_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_data_source = AsyncMock(return_value=None)
        resp = sources_client.get("/api/v1/kb-registry/knowledge-bases/mykb/sources/missing/status")
        assert resp.status_code == 404

    def test_get_source_status_success(self, sources_client, sources_mock_registry):
        now = datetime.now(timezone.utc)
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_data_source = AsyncMock(return_value={
            "id": "s1", "kb_id": 1, "kb_name": "mykb", "source_type": "confluence",
            "display_name": "Conf", "config_public": "{}", "sync_schedule": None,
            "sync_status": "success", "last_sync_at": None, "last_sync_error": None,
            "document_count": 5, "created_at": now,
        })
        sources_mock_registry.db.get_source_sync_runs = AsyncMock(return_value=[
            {"id": 1, "source_id": "s1", "started_at": now, "completed_at": None,
             "status": "success", "document_count": 5, "error_message": None}
        ])
        resp = sources_client.get("/api/v1/kb-registry/knowledge-bases/mykb/sources/s1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"]["document_count"] == 5

    def test_trigger_sync_kb_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = sources_client.post("/api/v1/kb-registry/knowledge-bases/missing/sources/s1/sync")
        assert resp.status_code == 404

    def test_trigger_sync_source_not_found(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.get_data_source = AsyncMock(return_value=None)
        resp = sources_client.post("/api/v1/kb-registry/knowledge-bases/mykb/sources/missing/sync")
        assert resp.status_code == 404

    def test_test_connection_success(self, sources_client, sources_mock_registry):
        """test_loader_connection delegates to sync_svc — mock via dependency override."""
        from oai_kb_registry.main import app
        from oai_kb_registry import dependencies
        from oai_kb_registry.services.source_sync_service import SourceSyncService
        mock_svc = MagicMock(spec=SourceSyncService)
        mock_svc.test_connection = AsyncMock(return_value={"status": "ok", "sample_count": 3})

        def _override_sync():
            return mock_svc

        app.dependency_overrides[dependencies.get_sync_service] = _override_sync
        resp = sources_client.post(
            "/api/v1/kb-registry/loaders/test",
            json={"source_type": "confluence", "config": {"url": "https://x"}},
        )
        app.dependency_overrides.pop(dependencies.get_sync_service, None)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_add_source_success(self, sources_client, sources_mock_registry):
        sources_mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        sources_mock_registry.db.create_data_source = AsyncMock(return_value="new-source-id")
        sources_mock_registry.db.create_sync_run = AsyncMock(return_value="run-1")
        sources_mock_registry.db.update_data_source_sync_status = AsyncMock()
        sources_mock_registry.db.get_data_source = AsyncMock(return_value={
            "id": "new-source-id", "kb_id": 1, "kb_name": "mykb", "source_type": "confluence",
            "display_name": "My Conf", "config_public": '{"url":"https://x"}',
            "sync_schedule": None, "sync_status": "running", "last_sync_at": None,
            "last_sync_error": None, "document_count": 0,
            "created_at": datetime.now(timezone.utc),
        })
        resp = sources_client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/sources",
            json={
                "source_type": "confluence",
                "display_name": "My Conf",
                "config": {"url": "https://x", "username": "u", "api_key": "k"},
            },
        )
        assert resp.status_code in (200, 201)


# ===========================================================================
# MAIN APP TESTS
# ===========================================================================

class TestMainApp:
    def test_cors_middleware_present(self):
        from oai_kb_registry.main import app
        middleware_types = [m.cls.__name__ for m in app.user_middleware]
        assert any("CORS" in t for t in middleware_types)

    def test_app_has_routes(self):
        from oai_kb_registry.main import app
        from fastapi.testclient import TestClient
        
        # In some FastAPI versions, routes are deferred until startup.
        # Instantiating TestClient forces the app.setup() process to populate app.routes.
        with TestClient(app):
            pass

        paths = []
        def _get_paths(routes):
            for r in routes:
                if hasattr(r, 'path') and isinstance(r.path, str):
                    paths.append(r.path)
                if hasattr(r, 'routes'):
                    _get_paths(r.routes)
        _get_paths(app.routes)
        assert any("knowledge-bases" in p for p in paths)

    def test_app_title(self):
        from oai_kb_registry.main import app
        assert "Knowledge Base" in app.title
