"""
Comprehensive unit tests for KB Registry — targets ~90% coverage.

All external services (DB, vector stores, InfraManager, TokenManager, etc.)
are mocked so the tests run without any infrastructure.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Environment setup — must come BEFORE the app is imported
# ---------------------------------------------------------------------------
os.environ["KB_AUTH_ENABLED"] = "false"
os.environ["REGISTRY_DB_LOGGING_ENABLED"] = "false"


# ===========================================================================
# MODEL TESTS
# ===========================================================================

class TestEnums:
    def test_vector_db_type_values(self):
        from oai_kb_registry.models import VectorDBType
        assert VectorDBType.CHROMA == "chroma"
        assert VectorDBType.POSTGRES == "postgres"
        assert VectorDBType.NEO4J == "neo4j_graph"
        assert VectorDBType.PINECONE == "pinecone"
        assert VectorDBType.S3 == "s3"

    def test_deployment_mode_values(self):
        from oai_kb_registry.models import DeploymentMode
        assert DeploymentMode.BUILTIN == "builtin"
        assert DeploymentMode.EXTERNAL == "external"

    def test_kb_status_values(self):
        from oai_kb_registry.models import KBStatus
        assert KBStatus.ACTIVE == "active"
        assert KBStatus.INACTIVE == "inactive"
        assert KBStatus.PROVISIONING == "provisioning"
        assert KBStatus.ERROR == "error"

    def test_document_status_values(self):
        from oai_kb_registry.models import DocumentStatus
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.INDEXING == "indexing"
        assert DocumentStatus.INDEXED == "indexed"
        assert DocumentStatus.FAILED == "failed"

    def test_sync_status_values(self):
        from oai_kb_registry.models import SyncStatus
        assert SyncStatus.IDLE == "idle"
        assert SyncStatus.RUNNING == "running"
        assert SyncStatus.SUCCESS == "success"
        assert SyncStatus.FAILED == "failed"
        assert SyncStatus.NEVER == "never"

    def test_loader_field_type_values(self):
        from oai_kb_registry.models import LoaderFieldType
        assert LoaderFieldType.TEXT == "text"
        assert LoaderFieldType.SECRET == "secret"
        assert LoaderFieldType.BOOLEAN == "boolean"


class TestVectorDBConfig:
    def test_defaults(self):
        from oai_kb_registry.models import VectorDBConfig
        cfg = VectorDBConfig()
        assert cfg.chroma_port == 8000
        assert cfg.pg_port == 5432
        assert cfg.s3_prefix == "kb_vector_store"
        assert cfg.neo4j_database == "neo4j"
        assert cfg.neo4j_retrieval_mode == "traversal"
        assert cfg.neo4j_max_hops == 2
        assert cfg.neo4j_rel_limit == 50

    def test_full_config(self):
        from oai_kb_registry.models import VectorDBConfig
        cfg = VectorDBConfig(
            chroma_host="localhost",
            chroma_port=9000,
            pg_host="pghost",
            pg_user="user",
            pg_password="pass",
            pg_database="mydb",
            s3_bucket="mybucket",
            pinecone_api_key="pk-key",
            neo4j_url="bolt://neo4j:7687",
        )
        assert cfg.chroma_host == "localhost"
        assert cfg.chroma_port == 9000
        assert cfg.pg_host == "pghost"
        assert cfg.s3_bucket == "mybucket"


class TestEmbeddingConfig:
    def test_default_model(self):
        from oai_kb_registry.models import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.model_id == "bedrock/amazon.titan-embed-text-v1"
        assert cfg.region_name is None

    def test_custom_model(self):
        from oai_kb_registry.models import EmbeddingConfig
        cfg = EmbeddingConfig(model_id="openai/text-embedding-3-small", region_name="us-west-2")
        assert cfg.model_id == "openai/text-embedding-3-small"
        assert cfg.region_name == "us-west-2"


class TestChunkingConfig:
    def test_defaults(self):
        from oai_kb_registry.models import ChunkingConfig
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_bounds(self):
        from oai_kb_registry.models import ChunkingConfig
        import pydantic
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=50)  # below ge=100
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=9000)  # above le=8000


class TestRetrievalConfig:
    def test_defaults(self):
        from oai_kb_registry.models import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.top_k == 5
        assert cfg.score_threshold == 0.7

    def test_bounds(self):
        from oai_kb_registry.models import RetrievalConfig
        with pytest.raises(Exception):
            RetrievalConfig(top_k=0)
        with pytest.raises(Exception):
            RetrievalConfig(score_threshold=1.5)


class TestKBRegistration:
    def test_defaults(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, DeploymentMode
        reg = KBRegistration(name="test-kb")
        assert reg.vector_db_type == VectorDBType.CHROMA
        assert reg.deployment_mode == DeploymentMode.BUILTIN
        assert reg.tags == []

    def test_pinecone_forced_external(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, DeploymentMode
        reg = KBRegistration(
            name="pinecone-kb",
            vector_db_type=VectorDBType.PINECONE,
            deployment_mode=DeploymentMode.BUILTIN,
        )
        assert reg.deployment_mode == DeploymentMode.EXTERNAL

    def test_neo4j_forced_external(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, DeploymentMode
        reg = KBRegistration(
            name="neo4j-kb",
            vector_db_type=VectorDBType.NEO4J,
            deployment_mode=DeploymentMode.BUILTIN,
        )
        assert reg.deployment_mode == DeploymentMode.EXTERNAL

    def test_neo4j_text2cypher_requires_llm(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, VectorDBConfig
        with pytest.raises(ValueError, match="neo4j_llm_model_id"):
            KBRegistration(
                name="g",
                vector_db_type=VectorDBType.NEO4J,
                vector_db_config=VectorDBConfig(
                    neo4j_url="bolt://x",
                    neo4j_retrieval_mode="text2cypher",
                ),
            )

    def test_neo4j_text2cypher_with_llm_ok(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, VectorDBConfig
        reg = KBRegistration(
            name="g",
            vector_db_type=VectorDBType.NEO4J,
            vector_db_config=VectorDBConfig(
                neo4j_url="bolt://x",
                neo4j_retrieval_mode="text2cypher",
                neo4j_llm_model_id="bedrock/claude-3",
            ),
        )
        assert reg.vector_db_config.neo4j_llm_model_id == "bedrock/claude-3"

    def test_neo4j_traversal_no_llm_ok(self):
        from oai_kb_registry.models import KBRegistration, VectorDBType, VectorDBConfig
        reg = KBRegistration(
            name="g",
            vector_db_type=VectorDBType.NEO4J,
            vector_db_config=VectorDBConfig(
                neo4j_url="bolt://x",
                neo4j_retrieval_mode="traversal",
            ),
        )
        assert reg.vector_db_config.neo4j_retrieval_mode == "traversal"

    def test_name_required(self):
        from oai_kb_registry.models import KBRegistration
        with pytest.raises(Exception):
            KBRegistration(name="")


class TestQueryRequest:
    def test_defaults(self):
        from oai_kb_registry.models import QueryRequest
        r = QueryRequest(query="hello world")
        assert r.k == 5
        assert r.include_metadata is True
        assert r.score_threshold is None

    def test_empty_query_rejected(self):
        from oai_kb_registry.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="")

    def test_k_bounds(self):
        from oai_kb_registry.models import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="x", k=0)
        with pytest.raises(Exception):
            QueryRequest(query="x", k=51)


class TestMiscModels:
    def test_status_response(self):
        from oai_kb_registry.models import StatusResponse
        r = StatusResponse(status="ok", message="done")
        assert r.status == "ok"

    def test_reindex_response(self):
        from oai_kb_registry.models import ReindexResponse
        r = ReindexResponse(status="started", kb_name="kb1", documents_queued=3)
        assert r.documents_queued == 3

    def test_query_response(self):
        from oai_kb_registry.models import QueryResponse, QueryResult
        r = QueryResponse(
            kb_name="kb1",
            query="test",
            results=[QueryResult(content="chunk", score=0.9, metadata={}, document_name="doc", document_id=1)],
            total_results=1,
        )
        assert r.total_results == 1

    def test_s3_document_source(self):
        from oai_kb_registry.models import S3DocumentSource
        s = S3DocumentSource(bucket="b", key="k")
        assert s.region == "us-east-1"

    def test_data_source_create(self):
        from oai_kb_registry.models import DataSourceCreate
        d = DataSourceCreate(source_type="confluence", display_name="My Conf", config={"url": "https://x"})
        assert d.sync_schedule is None


# ===========================================================================
# SECURITY DEPENDENCIES TESTS
# ===========================================================================

class TestIsBypassPath:
    def test_bypass_paths(self):
        from oai_kb_registry.security.dependencies import _is_bypass_path
        assert _is_bypass_path("/health") is True
        assert _is_bypass_path("/status") is True
        assert _is_bypass_path("/") is True
        assert _is_bypass_path("/api/v1/kb-registry/health") is True

    def test_non_bypass_paths(self):
        from oai_kb_registry.security.dependencies import _is_bypass_path
        assert _is_bypass_path("/knowledge-bases") is False
        assert _is_bypass_path("/tokens") is False


class TestValidateToken:
    """Tests for _validate_token — use mock Requests."""

    def _make_request(self, path="/knowledge-bases", host="192.168.1.1", headers=None):
        req = MagicMock()
        req.url.path = path
        req.client = MagicMock()
        req.client.host = host
        req.headers = headers or {}
        req.state = MagicMock()
        return req

    def test_bypass_path_skips_auth(self):
        from oai_kb_registry.security.dependencies import _validate_token
        req = self._make_request(path="/health")
        _validate_token(req)  # should not raise

    def test_auth_disabled_skips(self):
        from oai_kb_registry.security.dependencies import _validate_token
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "false"}):
            req = self._make_request()
            _validate_token(req)  # should not raise

    def test_trusted_peer_skips_when_no_force_auth(self):
        from oai_kb_registry.security.dependencies import _validate_token
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "false"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=True):
                req = self._make_request()
                _validate_token(req)  # should not raise

    def test_no_token_raises_401(self):
        from oai_kb_registry.security.dependencies import _validate_token
        from fastapi import HTTPException
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value=None):
                    req = self._make_request()
                    with pytest.raises(HTTPException) as exc_info:
                        _validate_token(req)
                    assert exc_info.value.status_code == 401

    def test_valid_api_token(self):
        from oai_kb_registry.security.dependencies import _validate_token
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="token123"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=False):
                        mock_tm = MagicMock()
                        mock_tm.validate_token.return_value = {"user_id": "u1", "role_id": "admin"}
                        with patch("oai_kb_registry.security.dependencies._get_token_manager", return_value=mock_tm):
                            req = self._make_request()
                            _validate_token(req)
                            assert req.state.user_id == "u1"

    def test_invalid_api_token_raises_401(self):
        from oai_kb_registry.security.dependencies import _validate_token
        from fastapi import HTTPException
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="bad"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=False):
                        mock_tm = MagicMock()
                        mock_tm.validate_token.return_value = None
                        with patch("oai_kb_registry.security.dependencies._get_token_manager", return_value=mock_tm):
                            req = self._make_request()
                            with pytest.raises(HTTPException) as exc_info:
                                _validate_token(req)
                            assert exc_info.value.status_code == 401

    def test_saml_token_valid(self):
        from oai_kb_registry.security.dependencies import _validate_token
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="samltoken"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=True):
                        mock_result = MagicMock()
                        mock_result.is_valid = True
                        mock_result.email = "user@test.com"
                        mock_result.role = "admin"
                        mock_validator = MagicMock()
                        mock_validator.validate_token_and_get_role.return_value = mock_result
                        with patch("oai_kb_registry.security.dependencies.TokenValidator", return_value=mock_validator):
                            req = self._make_request()
                            _validate_token(req)
                            assert req.state.user_email == "user@test.com"

    def test_saml_token_invalid_raises_401(self):
        from oai_kb_registry.security.dependencies import _validate_token
        from fastapi import HTTPException
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="samltoken"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=True):
                        mock_result = MagicMock()
                        mock_result.is_valid = False
                        mock_result.error_message = "expired"
                        mock_validator = MagicMock()
                        mock_validator.validate_token_and_get_role.return_value = mock_result
                        with patch("oai_kb_registry.security.dependencies.TokenValidator", return_value=mock_validator):
                            req = self._make_request()
                            with pytest.raises(HTTPException) as exc_info:
                                _validate_token(req)
                            assert exc_info.value.status_code == 401

    def test_saml_validation_error_raises_401(self):
        from oai_kb_registry.security.dependencies import _validate_token
        from oai_platform_core.security.saml_token_validation import TokenValidationError
        from fastapi import HTTPException
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="samltoken"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=True):
                        mock_validator = MagicMock()
                        mock_validator.validate_token_and_get_role.side_effect = TokenValidationError("bad saml")
                        with patch("oai_kb_registry.security.dependencies.TokenValidator", return_value=mock_validator):
                            req = self._make_request()
                            with pytest.raises(HTTPException) as exc_info:
                                _validate_token(req)
                            assert exc_info.value.status_code == 401

    def test_saml_service_unavailable_raises_500(self):
        from oai_kb_registry.security.dependencies import _validate_token
        from fastapi import HTTPException
        with patch.dict(os.environ, {"KB_AUTH_ENABLED": "true", "FORCE_AUTH": "true"}):
            with patch("oai_kb_registry.security.dependencies.is_trusted_peer", return_value=False):
                with patch("oai_kb_registry.security.dependencies.extract_bearer_token", return_value="samltoken"):
                    with patch("oai_kb_registry.security.dependencies.is_saml_token", return_value=True):
                        with patch("oai_kb_registry.security.dependencies.TokenValidator", side_effect=Exception("unavailable")):
                            req = self._make_request()
                            with pytest.raises(HTTPException) as exc_info:
                                _validate_token(req)
                            assert exc_info.value.status_code == 500


# ===========================================================================
# FastAPI CLIENT FIXTURE
# ===========================================================================

@pytest.fixture(scope="module")
def mock_registry():
    """Shared mock KBRegistry."""
    reg = MagicMock()
    reg.db = MagicMock()
    return reg


@pytest.fixture(scope="module")
def client(mock_registry):
    """TestClient with auth disabled and registry mocked."""
    os.environ["KB_AUTH_ENABLED"] = "false"
    os.environ["REGISTRY_DB_LOGGING_ENABLED"] = "false"

    from oai_kb_registry.main import app
    from oai_kb_registry import dependencies

    async def _override_registry():
        return mock_registry

    app.dependency_overrides[dependencies.get_registry] = _override_registry

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ===========================================================================
# KNOWLEDGE BASE ROUTER TESTS
# ===========================================================================

class TestKBRouter:
    def test_health(self, client):
        resp = client.get("/api/v1/kb-registry/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root(self, client):
        resp = client.get("/api/v1/kb-registry/")
        assert resp.status_code == 200
        assert "endpoints" in resp.json()

    def test_list_kbs(self, client, mock_registry):
        mock_registry.db.get_all_knowledge_bases = AsyncMock(return_value=[
            {"id": 1, "name": "kb1", "status": "active"}
        ])
        resp = client.get("/api/v1/kb-registry/knowledge-bases")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_kb_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "mykb", "status": "active"
        })
        resp = client.get("/api/v1/kb-registry/knowledge-bases/mykb")
        assert resp.status_code == 200

    def test_get_kb_not_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = client.get("/api/v1/kb-registry/knowledge-bases/nonexistent")
        assert resp.status_code == 404

    def test_register_kb_success(self, client, mock_registry):
        mock_registry.register_knowledge_base = AsyncMock(return_value={
            "id": 2, "name": "newkb", "status": "active"
        })
        payload = {"name": "newkb"}
        resp = client.post("/api/v1/kb-registry/knowledge-bases", json=payload)
        assert resp.status_code == 201

    def test_register_kb_error(self, client, mock_registry):
        mock_registry.register_knowledge_base = AsyncMock(side_effect=Exception("db error"))
        resp = client.post("/api/v1/kb-registry/knowledge-bases", json={"name": "bad-kb"})
        assert resp.status_code == 500

    def test_delete_kb_success(self, client, mock_registry):
        mock_registry.delete_knowledge_base = AsyncMock()
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/mykb")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_kb_not_found(self, client, mock_registry):
        mock_registry.delete_knowledge_base = AsyncMock(side_effect=ValueError("not found"))
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/missing")
        assert resp.status_code == 404

    def test_delete_kb_error(self, client, mock_registry):
        mock_registry.delete_knowledge_base = AsyncMock(side_effect=Exception("oops"))
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/bad")
        assert resp.status_code == 500

    def test_get_kb_history(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "kb1"})
        mock_registry.db.get_kb_actions = AsyncMock(return_value=[
            {"id": 1, "action": "register", "created_at": None}
        ])
        resp = client.get("/api/v1/kb-registry/knowledge-bases/kb1/history")
        assert resp.status_code == 200
        assert resp.json()["kb_name"] == "kb1"

    def test_get_kb_history_not_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = client.get("/api/v1/kb-registry/knowledge-bases/missing/history")
        assert resp.status_code == 404

    def test_get_kb_config_chroma_builtin(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1,
            "name": "chromakb",
            "vector_db_type": "chroma",
            "deployment_mode": "builtin",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": '{"top_k": 5, "score_threshold": 0.7}',
            "description": "test",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs", return_value={}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/chromakb/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vector_store"]["type"] == "chroma"

    def test_get_kb_config_chroma_external(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1,
            "name": "chromakb",
            "vector_db_type": "chroma",
            "deployment_mode": "external",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": None,
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs",
                   return_value={"chroma_host": "chromahost", "chroma_port": "8000", "chroma_ssl": "false"}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/chromakb/config")
        assert resp.status_code == 200

    def test_get_kb_config_postgres_builtin(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 2,
            "name": "pgkb",
            "vector_db_type": "postgres",
            "deployment_mode": "builtin",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": "{}",
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs", return_value={}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/pgkb/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vector_store"]["type"] == "postgres"

    def test_get_kb_config_postgres_external(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 2,
            "name": "pgkb",
            "vector_db_type": "postgres",
            "deployment_mode": "external",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": "{}",
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs",
                   return_value={"pg_host": "pghost", "pg_port": "5432", "pg_user": "u", "pg_password": "p", "pg_database": "db"}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/pgkb/config")
        assert resp.status_code == 200

    def test_get_kb_config_s3(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 3,
            "name": "s3kb",
            "vector_db_type": "s3",
            "deployment_mode": "external",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "retrieval_config": "{}",
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs",
                   return_value={"s3_bucket": "mybucket", "aws_access_key_id": "key", "aws_secret_access_key": "secret"}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/s3kb/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vector_store"]["type"] == "s3"

    def test_get_kb_config_pinecone(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 4,
            "name": "pckb",
            "vector_db_type": "pinecone",
            "deployment_mode": "external",
            "embedding_model_id": "bedrock/amazon.titan-embed-text-v1",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": "{}",
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs",
                   return_value={"pinecone_api_key": "pk", "pinecone_index_name": "idx"}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/pckb/config")
        assert resp.status_code == 200

    def test_get_kb_config_unsupported_type(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 5,
            "name": "unknown",
            "vector_db_type": "unknown_type",
            "deployment_mode": "external",
            "embedding_model_id": "bedrock/x",
            "embedding_region": None,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_config": "{}",
            "description": "",
        })
        mock_registry.db.get_kb_configs = AsyncMock(return_value={})
        with patch("oai_kb_registry.routers.knowledge_bases.VectorStoreProviderFactory.decrypt_configs", return_value={}):
            resp = client.get("/api/v1/kb-registry/knowledge-bases/unknown/config")
        assert resp.status_code == 400

    def test_get_kb_config_not_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = client.get("/api/v1/kb-registry/knowledge-bases/missing/config")
        assert resp.status_code == 404

    def test_get_graph_stats_success(self, client, mock_registry):
        mock_registry.graph_stats = AsyncMock(return_value={"nodes": 100})
        resp = client.get("/api/v1/kb-registry/knowledge-bases/mykb/graph-stats")
        assert resp.status_code == 200

    def test_get_graph_stats_not_found(self, client, mock_registry):
        mock_registry.graph_stats = AsyncMock(side_effect=ValueError("not found"))
        resp = client.get("/api/v1/kb-registry/knowledge-bases/missing/graph-stats")
        assert resp.status_code == 404

    def test_get_graph_stats_error(self, client, mock_registry):
        mock_registry.graph_stats = AsyncMock(side_effect=Exception("connection error"))
        resp = client.get("/api/v1/kb-registry/knowledge-bases/mykb/graph-stats")
        assert resp.status_code == 500


# ===========================================================================
# DOCUMENT ROUTER TESTS
# ===========================================================================

class TestDocumentRouter:
    def test_upload_document_success(self, client, mock_registry):
        mock_registry.index_document_from_upload = AsyncMock(return_value={
            "id": 1, "name": "test.txt", "status": "indexed"
        })
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "indexed"

    def test_upload_document_unsupported_extension(self, client):
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/upload",
            files={"file": ("test.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_document_empty_file(self, client):
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/upload",
            files={"file": ("test.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_document_kb_not_found(self, client, mock_registry):
        mock_registry.index_document_from_upload = AsyncMock(side_effect=ValueError("kb not found"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/upload",
            files={"file": ("doc.pdf", b"pdf content", "application/pdf")},
        )
        assert resp.status_code == 404

    def test_upload_document_error(self, client, mock_registry):
        mock_registry.index_document_from_upload = AsyncMock(side_effect=Exception("indexing failed"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/upload",
            files={"file": ("doc.md", b"# Title", "text/markdown")},
        )
        assert resp.status_code == 500

    def test_index_s3_document_success(self, client, mock_registry):
        mock_registry.index_document_from_s3 = AsyncMock(return_value={
            "id": 2, "name": "s3doc.pdf", "status": "indexed"
        })
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/s3",
            json={"bucket": "mybucket", "key": "docs/file.pdf"},
        )
        assert resp.status_code == 201

    def test_index_s3_document_not_found(self, client, mock_registry):
        mock_registry.index_document_from_s3 = AsyncMock(side_effect=ValueError("kb not found"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/s3",
            json={"bucket": "b", "key": "k"},
        )
        assert resp.status_code == 404

    def test_index_s3_document_error(self, client, mock_registry):
        mock_registry.index_document_from_s3 = AsyncMock(side_effect=Exception("s3 error"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/mykb/documents/s3",
            json={"bucket": "b", "key": "k"},
        )
        assert resp.status_code == 500

    def test_list_documents(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={"id": 1, "name": "mykb"})
        mock_registry.db.get_kb_documents = AsyncMock(return_value=[
            {"id": 1, "name": "doc.txt", "status": "indexed"}
        ])
        resp = client.get("/api/v1/kb-registry/knowledge-bases/mykb/documents")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_documents_kb_not_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = client.get("/api/v1/kb-registry/knowledge-bases/missing/documents")
        assert resp.status_code == 404

    def test_remove_document_success(self, client, mock_registry):
        mock_registry.remove_document = AsyncMock()
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/mykb/documents/1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

    def test_remove_document_not_found(self, client, mock_registry):
        mock_registry.remove_document = AsyncMock(side_effect=ValueError("not found"))
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/mykb/documents/999")
        assert resp.status_code == 404

    def test_remove_document_error(self, client, mock_registry):
        mock_registry.remove_document = AsyncMock(side_effect=Exception("db error"))
        resp = client.delete("/api/v1/kb-registry/knowledge-bases/mykb/documents/1")
        assert resp.status_code == 500

    def test_reindex_success(self, client, mock_registry):
        mock_registry.reindex_knowledge_base = AsyncMock(return_value=3)
        resp = client.post("/api/v1/kb-registry/knowledge-bases/mykb/reindex")
        assert resp.status_code == 200
        assert resp.json()["documents_queued"] == 3

    def test_reindex_not_found(self, client, mock_registry):
        mock_registry.reindex_knowledge_base = AsyncMock(side_effect=ValueError("not found"))
        resp = client.post("/api/v1/kb-registry/knowledge-bases/missing/reindex")
        assert resp.status_code == 404

    def test_reindex_error(self, client, mock_registry):
        mock_registry.reindex_knowledge_base = AsyncMock(side_effect=Exception("error"))
        resp = client.post("/api/v1/kb-registry/knowledge-bases/mykb/reindex")
        assert resp.status_code == 500


# ===========================================================================
# QUERY ROUTER TESTS
# ===========================================================================

class TestQueryRouter:
    def _make_query_result(self):
        return [
            {
                "content": "chunk text",
                "score": 0.85,
                "metadata": {"source": "doc.pdf"},
                "document_name": "doc.pdf",
                "document_id": 1,
            }
        ]

    def test_query_success(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "kb1", "status": "active"
        })
        mock_registry.query = AsyncMock(return_value=self._make_query_result())
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/kb1/query",
            json={"query": "what is the policy?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 1
        assert data["results"][0]["content"] == "chunk text"

    def test_query_without_metadata(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "kb1", "status": "active"
        })
        mock_registry.query = AsyncMock(return_value=self._make_query_result())
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/kb1/query",
            json={"query": "test", "include_metadata": False},
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["metadata"] is None

    def test_query_kb_not_found(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value=None)
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/missing/query",
            json={"query": "test"},
        )
        assert resp.status_code == 404

    def test_query_kb_in_error_state(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "kb1", "status": "error"
        })
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/kb1/query",
            json={"query": "test"},
        )
        assert resp.status_code == 503

    def test_query_value_error(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "kb1", "status": "active"
        })
        mock_registry.query = AsyncMock(side_effect=ValueError("kb missing"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/kb1/query",
            json={"query": "test"},
        )
        assert resp.status_code == 404

    def test_query_generic_error(self, client, mock_registry):
        mock_registry.db.get_knowledge_base = AsyncMock(return_value={
            "id": 1, "name": "kb1", "status": "active"
        })
        mock_registry.query = AsyncMock(side_effect=Exception("embedding failed"))
        resp = client.post(
            "/api/v1/kb-registry/knowledge-bases/kb1/query",
            json={"query": "test"},
        )
        assert resp.status_code == 500


# ===========================================================================
# TOKEN ROUTER TESTS
# ===========================================================================

class TestTokenRouter:
    def _mock_token_manager(self):
        tm = MagicMock()
        tm.generate_token.return_value = "tok-abc123"
        tm.get_all_tokens.return_value = [
            {"token": "tok-abc123", "user_id": "u1", "is_expired": False}
        ]
        tm.revoke_all_tokens.return_value = 2
        tm.revoke_token.return_value = True
        return tm

    def test_generate_token(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.post("/api/v1/kb-registry/tokens/generate?user_id=u1&role_id=admin&ttl_seconds=3600")
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "tok-abc123"
        assert data["user_id"] == "u1"

    def test_generate_token_no_expiry(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.post("/api/v1/kb-registry/tokens/generate?ttl_seconds=-1")
        assert resp.status_code == 200

    def test_generate_token_too_many(self, client):
        tm = MagicMock()
        tm.generate_token.side_effect = ValueError("max tokens reached")
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.post("/api/v1/kb-registry/tokens/generate")
        assert resp.status_code == 429

    def test_generate_token_error(self, client):
        tm = MagicMock()
        tm.generate_token.side_effect = Exception("db error")
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.post("/api/v1/kb-registry/tokens/generate")
        assert resp.status_code == 500

    def test_list_tokens(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.get("/api/v1/kb-registry/tokens")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["can_generate"] is True

    def test_list_tokens_include_expired(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.get("/api/v1/kb-registry/tokens?include_expired=true")
        assert resp.status_code == 200

    def test_list_tokens_error(self, client):
        tm = MagicMock()
        tm.get_all_tokens.side_effect = Exception("db error")
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.get("/api/v1/kb-registry/tokens")
        assert resp.status_code == 500

    def test_revoke_all_tokens(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.delete("/api/v1/kb-registry/tokens")
        assert resp.status_code == 200
        assert resp.json()["revoked"] == 2

    def test_revoke_all_tokens_error(self, client):
        tm = MagicMock()
        tm.revoke_all_tokens.side_effect = Exception("db error")
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.delete("/api/v1/kb-registry/tokens")
        assert resp.status_code == 500

    def test_revoke_specific_token_success(self, client):
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=self._mock_token_manager()):
            resp = client.delete("/api/v1/kb-registry/tokens/tok-abc123")
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

    def test_revoke_specific_token_not_found(self, client):
        tm = MagicMock()
        tm.revoke_token.return_value = False
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.delete("/api/v1/kb-registry/tokens/unknown-tok")
        assert resp.status_code == 404

    def test_revoke_specific_token_error(self, client):
        tm = MagicMock()
        tm.revoke_token.side_effect = Exception("error")
        with patch("oai_kb_registry.routers.token._get_token_manager", return_value=tm):
            resp = client.delete("/api/v1/kb-registry/tokens/tok")
        assert resp.status_code == 500


# ===========================================================================
# DEPENDENCIES TESTS
# ===========================================================================

class TestDependencies:
    def test_get_registry_uninitialized(self):
        import oai_kb_registry.dependencies as deps
        original = deps._kb_registry
        deps._kb_registry = None
        with pytest.raises(RuntimeError, match="not initialised"):
            deps.get_registry()
        deps._kb_registry = original

    def test_get_sync_service(self):
        from oai_kb_registry.dependencies import get_sync_service
        from oai_kb_registry.services.source_sync_service import SourceSyncService
        mock_reg = MagicMock()
        svc = get_sync_service(registry=mock_reg)
        assert isinstance(svc, SourceSyncService)

    def test_get_auth_user_with_email(self):
        import asyncio
        from oai_kb_registry.dependencies import get_auth_user
        req = MagicMock()
        req.state.user_email = "test@example.com"
        req.state.user_id = None
        result = asyncio.run(
            get_auth_user(request=req, _auth=True)
        )
        assert result == "test@example.com"

    def test_get_auth_user_with_user_id(self):
        import asyncio
        from oai_kb_registry.dependencies import get_auth_user
        req = MagicMock()
        req.state.user_email = None
        req.state.user_id = "uid123"
        result = asyncio.run(
            get_auth_user(request=req, _auth=True)
        )
        assert result == "uid123"

    def test_get_auth_user_fallback(self):
        import asyncio
        from oai_kb_registry.dependencies import get_auth_user
        req = MagicMock(spec=[])
        req.state = MagicMock(spec=[])
        result = asyncio.run(
            get_auth_user(request=req, _auth=True)
        )
        assert result == "authenticated_user"

    def test_initialize_registry_no_auto_start(self):
        """initialize_registry without auto_start should create DB logger and registry."""
        import asyncio
        import oai_kb_registry.dependencies as deps

        mock_db = AsyncMock()
        mock_db.initialize = AsyncMock(return_value=True)
        mock_db.close = AsyncMock()

        mock_reg = MagicMock()
        mock_reg.restore_from_db = AsyncMock()

        with patch("oai_kb_registry.dependencies.KBDatabaseLogger", return_value=mock_db):
            with patch("oai_kb_registry.dependencies.KBRegistry", return_value=mock_reg):
                with patch.dict(os.environ, {"AUTO_START_INFRA": "false"}):
                    result = asyncio.run(
                        deps.initialize_registry()
                    )
        assert result is mock_reg

    def test_close_registry_no_infra(self):
        """close_registry when no infra was started."""
        import asyncio
        import oai_kb_registry.dependencies as deps

        mock_reg = AsyncMock()
        mock_reg.close = AsyncMock()
        orig_reg = deps._kb_registry
        orig_infra = deps._infra_started
        deps._kb_registry = mock_reg
        deps._infra_started = False

        asyncio.run(deps.close_registry())
        mock_reg.close.assert_called_once()

        deps._kb_registry = orig_reg
        deps._infra_started = orig_infra


# ===========================================================================
# LOADER CATALOG TESTS
# ===========================================================================

class TestLoaderCatalog:
    def test_catalog_has_entries(self):
        from oai_kb_registry.loaders.catalog import LOADER_CATALOG
        assert len(LOADER_CATALOG) > 0
        assert "confluence" in LOADER_CATALOG

    def test_confluence_entry_structure(self):
        from oai_kb_registry.loaders.catalog import LOADER_CATALOG
        entry = LOADER_CATALOG["confluence"]
        assert "fields" in entry
        assert entry["category"] == "collaboration"
        assert any(f["name"] == "url" for f in entry["fields"])

    def test_get_public_config(self):
        from oai_kb_registry.loaders.catalog import get_public_config
        config = {"api_key": "secret", "url": "https://x", "username": "u"}
        source_type = "confluence"
        public = get_public_config(source_type, config)
        # Secret fields should be masked
        assert public.get("api_key") != "secret"
        assert public.get("url") == "https://x"


# ===========================================================================
# DOCUMENT VALIDATOR TESTS
# ===========================================================================

class TestDocumentValidator:
    def test_allowed_extensions(self):
        from oai_kb_registry.routers.documents import _validate_upload
        from fastapi import HTTPException
        for ext in [".pdf", ".docx", ".txt", ".md", ".csv"]:
            upload = MagicMock()
            upload.filename = f"file{ext}"
            _validate_upload(upload)  # should not raise

    def test_disallowed_extension(self):
        from oai_kb_registry.routers.documents import _validate_upload
        from fastapi import HTTPException
        upload = MagicMock()
        upload.filename = "malware.exe"
        with pytest.raises(HTTPException) as exc_info:
            _validate_upload(upload)
        assert exc_info.value.status_code == 415

    def test_no_extension(self):
        from oai_kb_registry.routers.documents import _validate_upload
        from fastapi import HTTPException
        upload = MagicMock()
        upload.filename = "noextension"
        with pytest.raises(HTTPException):
            _validate_upload(upload)
