import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.components.vector_store.registry_proxy_vector_store import (
    RegistryProxyVectorStore,
    _make_document
)

def test_registry_proxy_vector_store_init():
    store = RegistryProxyVectorStore(
        kb_name="my_kb",
        registry_url="http://registry.com/",
        auth_token="token123",
        retrieval_settings={"score_threshold": 0.5}
    )
    assert store.kb_name == "my_kb"
    assert store.registry_url == "http://registry.com"
    assert store.auth_token == "token123"
    assert store.retrieval_settings == {"score_threshold": 0.5}
    assert store._query_url == "http://registry.com/api/v1/kb-registry/knowledge-bases/my_kb/query"

@patch("requests.post")
def test_similarity_search_with_score_success(mock_post):
    store = RegistryProxyVectorStore(
        kb_name="my_kb",
        registry_url="http://registry.com",
        auth_token="token123",
        retrieval_settings={"score_threshold": 0.5}
    )
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "content": "policy details",
                "score": 0.95,
                "document_name": "terms.pdf",
                "document_id": "doc123",
                "metadata": {"section": "benefits"}
            }
        ]
    }
    mock_post.return_value = mock_response
    
    results = store.similarity_search_with_score("do I get dental?", k=3, filter={"category": "dental"})
    assert len(results) == 1
    doc, score = results[0]
    
    assert doc.page_content == "policy details"
    assert doc.metadata["section"] == "benefits"
    assert doc.metadata["source"] == "terms.pdf"
    assert doc.metadata["doc_id"] == "doc123"
    assert doc.metadata["kb_name"] == "my_kb"
    assert score == 0.95
    
    mock_post.assert_called_once_with(
        "http://registry.com/api/v1/kb-registry/knowledge-bases/my_kb/query",
        json={
            "query": "do I get dental?",
            "k": 3,
            "score_threshold": 0.5,
            "filter_metadata": {"category": "dental"}
        },
        headers={"Authorization": "Bearer token123"},
        timeout=30
    )

@patch("requests.post")
def test_similarity_search_with_score_failure(mock_post):
    import requests
    mock_post.side_effect = requests.RequestException("network down")
    store = RegistryProxyVectorStore(
        kb_name="my_kb",
        registry_url="http://registry.com",
        auth_token="token123"
    )
    
    results = store.similarity_search_with_score("query")
    assert results == []

def test_unsupported_indexing():
    store = RegistryProxyVectorStore(
        kb_name="my_kb",
        registry_url="http://registry.com",
        auth_token="token123"
    )
    
    with pytest.raises(NotImplementedError):
        store.add_documents([])

def test_reset_collection():
    store = RegistryProxyVectorStore(
        kb_name="my_kb",
        registry_url="http://registry.com",
        auth_token="token123"
    )
    # Reset is noop
    store.reset_collection()

def test_make_document_fallback():
    # Force ImportError on langchain_core
    with patch.dict("sys.modules", {"langchain_core.documents": None}):
        item = {
            "content": "fallback text",
            "document_name": "terms.pdf",
            "document_id": "doc123",
            "metadata": {"section": "benefits"}
        }
        doc = _make_document(item, "my_kb")
        assert doc.page_content == "fallback text"
        assert doc.metadata["source"] == "terms.pdf"
        assert doc.metadata["kb_name"] == "my_kb"
