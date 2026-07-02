"""
Advanced unit tests for SourceSyncService.
"""

import pytest
import os
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock langchain_community before it gets imported by anything
mock_lc = MagicMock()
sys.modules['langchain_community'] = mock_lc
sys.modules['langchain_community.document_loaders'] = mock_lc.document_loaders
sys.modules['langchain_community.document_loaders.sitemap'] = mock_lc.document_loaders.sitemap

from oai_kb_registry.services.source_sync_service import (
    SourceSyncService,
    _resolve_env_vars,
    _doc_name,
    _doc_uri,
    _cap_for_test
)


class TestSourceSyncAdvanced:
    """Advanced unit tests for SourceSyncService."""

    @pytest.fixture
    def mock_registry(self):
        reg = AsyncMock()
        reg.index_document_from_text = AsyncMock(return_value={"id": 1})
        return reg

    @pytest.fixture
    def service(self, mock_registry):
        return SourceSyncService(registry=mock_registry)

    def test_resolve_env_vars(self):
        os.environ["MOCK_API_KEY"] = "my-secret-key"
        config = {
            "api_key": "${MOCK_API_KEY}",
            "regular_val": "regular",
            "number": 123
        }
        res = _resolve_env_vars(config)
        assert res["api_key"] == "my-secret-key"
        assert res["regular_val"] == "regular"
        assert res["number"] == 123

    def test_doc_name_variants(self):
        class DummyDoc:
            def __init__(self, metadata):
                self.metadata = metadata

        assert _doc_name(DummyDoc({"title": "T1"}), 0, "web") == "T1"
        assert _doc_name(DummyDoc({"source": "S1"}), 0, "web") == "S1"
        assert _doc_name(DummyDoc({"path": "P1"}), 0, "web") == "P1"
        assert _doc_name(DummyDoc({"file_path": "FP1"}), 0, "web") == "FP1"
        assert _doc_name(DummyDoc({"url": "U1"}), 0, "web") == "U1"
        assert _doc_name(DummyDoc({}), 5, "web") == "web_doc_5"

    def test_doc_uri_variants(self):
        class DummyDoc:
            def __init__(self, metadata):
                self.metadata = metadata

        assert _doc_uri(DummyDoc({"source": "S1"})) == "S1"
        assert _doc_uri(DummyDoc({"url": "U1"})) == "U1"
        assert _doc_uri(DummyDoc({"loc": "L1"})) == "L1"
        assert _doc_uri(DummyDoc({})) is None

    def test_cap_for_test(self):
        conf = {"limit": "10"}
        _cap_for_test(conf)
        assert conf["limit"] == 3

        conf2 = {}
        _cap_for_test(conf2)
        assert conf2["limit"] == 3

    @pytest.mark.asyncio
    async def test_test_connection_unknown_type(self, service):
        res = await service.test_connection("invalid_type", {})
        assert res["status"] == "error"
        assert "Unknown source type" in res["message"]

    @pytest.mark.asyncio
    async def test_test_connection_success(self, service):
        mock_docs = ["doc1", "doc2"]
        mock_load = MagicMock(return_value=mock_docs)
        with patch.dict("oai_kb_registry.services.source_sync_service._LOADER_DISPATCH", {"web": mock_load}):
            res = await service.test_connection("web", {"urls": "http://example.com"})
            assert res["status"] == "ok"
            assert res["sample_count"] == 2
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_connection_import_error(self, service):
        # Trigger Confluence ImportError inside loader
        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "langchain_community.document_loaders" or "ConfluenceLoader" in name:
                raise ImportError("mocked confluence import error")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            res = await service.test_connection("confluence", {"url": "http://conf", "username": "u", "api_key": "k"})
            assert res["status"] == "error"
            assert "mocked confluence import error" in res["message"] or "atlassian-python-api" in res["message"]

    @pytest.mark.asyncio
    async def test_test_connection_general_error(self, service):
        mock_load = MagicMock(side_effect=RuntimeError("connection timeout"))
        with patch.dict("oai_kb_registry.services.source_sync_service._LOADER_DISPATCH", {"web": mock_load}):
            res = await service.test_connection("web", {"urls": "http://example.com"})
            assert res["status"] == "error"
            assert "connection timeout" in res["message"]

    @pytest.mark.asyncio
    async def test_sync_source_unknown_type(self, service):
        count, err = await service.sync_source("s1", "kb1", "invalid_type", {})
        assert count == 0
        assert "Unknown source type" in err

    @pytest.mark.asyncio
    async def test_sync_source_load_error(self, service):
        mock_load = MagicMock(side_effect=RuntimeError("load failed"))
        with patch.dict("oai_kb_registry.services.source_sync_service._LOADER_DISPATCH", {"web": mock_load}):
            count, err = await service.sync_source("s1", "kb1", "web", {"urls": "http://example.com"})
            assert count == 0
            assert "load failed" in err

    @pytest.mark.asyncio
    async def test_sync_source_success(self, service, mock_registry):
        class DummyDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        mock_docs = [
            DummyDoc("doc1 content", {"title": "Doc 1"}),
            DummyDoc("  ", {}), # empty text, skipped
            DummyDoc("doc2 content", {"title": "Doc 2"})
        ]

        mock_load = MagicMock(return_value=mock_docs)
        with patch.dict("oai_kb_registry.services.source_sync_service._LOADER_DISPATCH", {"web": mock_load}):
            count, err = await service.sync_source("s1", "kb1", "web", {"urls": "http://example.com"})
            assert count == 2
            assert err is None
            assert mock_registry.index_document_from_text.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_source_indexing_errors(self, service, mock_registry):
        class DummyDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        mock_docs = [
            DummyDoc("doc1 content", {"title": "Doc 1"}),
            DummyDoc("doc2 content", {"title": "Doc 2"}),
            DummyDoc("doc3 content", {"title": "Doc 3"}),
            DummyDoc("doc4 content", {"title": "Doc 4"}),
        ]

        # First doc succeeds, rest fail
        mock_registry.index_document_from_text.side_effect = [
            {"id": 1},
            RuntimeError("index fail 2"),
            RuntimeError("index fail 3"),
            RuntimeError("index fail 4"),
        ]

        mock_load = MagicMock(return_value=mock_docs)
        with patch.dict("oai_kb_registry.services.source_sync_service._LOADER_DISPATCH", {"web": mock_load}):
            count, err = await service.sync_source("s1", "kb1", "web", {"urls": "http://example.com"})
            assert count == 1
            assert "index fail 2; index fail 3; index fail 4" in err

    def test_load_confluence_implementation(self):
        # Test confluence loader instantiation & loading
        config = {
            "url": "http://confluence",
            "username": "user",
            "api_key": "key",
            "space_key": "SPACE1, SPACE2",
            "limit": 5,
            "include_attachments": True
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc1", "doc2"]
        with patch("langchain_community.document_loaders.ConfluenceLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_confluence
            res = _load_confluence(config)
            assert len(res) == 4 # SPACE1 (2 docs) + SPACE2 (2 docs)
            assert mock_loader.load.call_count == 2

    def test_load_sharepoint_implementation(self):
        config = {
            "client_id": "cid",
            "client_secret": "csec",
            "tenant_id": "tid",
            "site_name": "site",
            "document_library_path": "/folder"
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc"]
        with patch("langchain_community.document_loaders.SharePointLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_sharepoint
            res = _load_sharepoint(config)
            assert res == ["doc"]

    def test_load_s3_directory_implementation(self):
        config = {
            "bucket": "b",
            "prefix": "p",
            "region": "us-west-2",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret"
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc"]
        with patch("langchain_community.document_loaders.S3DirectoryLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_s3_directory
            res = _load_s3_directory(config)
            assert res == ["doc"]

    def test_load_web_implementation_sitemap(self):
        config = {
            "urls": "http://site.com/sitemap.xml",
            "use_sitemap": True
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc"]
        with patch("langchain_community.document_loaders.sitemap.SitemapLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_web
            res = _load_web(config)
            assert res == ["doc"]

    def test_load_web_implementation_base(self):
        config = {
            "urls": "http://site.com/page1\nhttp://site.com/page2",
            "use_sitemap": False
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc"]
        with patch("langchain_community.document_loaders.WebBaseLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_web
            res = _load_web(config)
            assert res == ["doc"]

    def test_load_github_implementation(self):
        config = {
            "repo": "org/repo",
            "branch": "main",
            "access_token": "token",
            "file_filter": "*.txt"
        }
        mock_loader = MagicMock()
        mock_loader.load.return_value = ["doc"]
        with patch("langchain_community.document_loaders.GithubFileLoader", return_value=mock_loader):
            from oai_kb_registry.services.source_sync_service import _load_github
            res = _load_github(config)
            assert res == ["doc"]
