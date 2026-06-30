import pytest
import sys
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# Mock the pinecone package before importing the module
mock_pinecone = MagicMock()
sys.modules['pinecone'] = mock_pinecone

from oai_agent_core.components.vector_store.pinecone_vector_store import PineconeVectorStore

@pytest.fixture
def mock_index():
    index = MagicMock()
    # Configure mock_pinecone to return this index
    mock_pinecone.Pinecone.return_value.Index.return_value = index
    return index

def test_pinecone_vector_store_init(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx",
        namespace="my_ns"
    )
    assert store.collection_name == "my_col"
    assert store._api_key == "my_key"
    assert store._index_name == "my_idx"
    assert store._namespace == "my_ns"
    mock_pinecone.Pinecone.assert_called_once_with(api_key="my_key")
    mock_pinecone.Pinecone.return_value.Index.assert_called_once_with("my_idx")

def test_pinecone_vector_store_init_missing_args():
    with pytest.raises(ValueError):
        PineconeVectorStore(collection_name="", api_key="k", index_name="i")
    with pytest.raises(ValueError):
        PineconeVectorStore(collection_name="c", api_key="", index_name="i")
    with pytest.raises(ValueError):
        PineconeVectorStore(collection_name="c", api_key="k", index_name="")

def test_add_documents(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    docs = [
        Document(page_content="hello", metadata={"category": "test"}),
        Document(page_content="world")
    ]
    
    store.add_documents(docs, ids=["id1", "id2"])
    
    mock_index.upsert_records.assert_called_once()
    args, kwargs = mock_index.upsert_records.call_args
    assert kwargs["namespace"] == "my_col"
    assert len(kwargs["records"]) == 2
    assert kwargs["records"][0]["_id"] == "id1"
    assert kwargs["records"][0]["text"] == "hello"
    assert kwargs["records"][0]["category"] == "test"

def test_add_texts(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    ids = store.add_texts(["text1", "text2"], metadatas=[{"arg": 1}, {"arg": 2}])
    assert len(ids) == 2
    assert mock_index.upsert_records.call_count == 1

def test_similarity_search_with_score(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    # Mock search response (dict style)
    mock_index.search.return_value = {
        "result": {
            "hits": [
                {"fields": {"text": "hello text", "category": "test"}, "_score": 0.95}
            ]
        }
    }
    
    results = store.similarity_search_with_score("hello", k=2, filter_metadata={"category": "test"})
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "hello text"
    assert doc.metadata["category"] == "test"
    assert score == 0.95
    
    mock_index.search.assert_called_once_with(
        namespace="my_col",
        inputs={"text": "hello"},
        top_k=2,
        filter={"category": "test"}
    )

def test_similarity_search_with_score_object_response(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    # Mock search response (object style)
    mock_response = MagicMock()
    mock_hit = MagicMock()
    mock_hit.fields = {"text": "hello obj"}
    mock_hit._score = 0.88
    mock_response.result.hits = [mock_hit]
    mock_index.search.return_value = mock_response
    
    results = store.similarity_search_with_score("hello")
    assert len(results) == 1
    assert results[0][0].page_content == "hello obj"
    assert results[0][1] == 0.88

def test_similarity_search(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    mock_index.search.return_value = {
        "result": {
            "hits": [
                {"fields": {"text": "hello"}, "_score": 0.9}
            ]
        }
    }
    
    docs = store.similarity_search("hello")
    assert len(docs) == 1
    assert docs[0].page_content == "hello"

def test_count(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    # Dict style
    mock_index.describe_index_stats.return_value = {
        "namespaces": {
            "my_col": {"vector_count": 42}
        }
    }
    assert store.count() == 42
    
    # Object style
    mock_stats = MagicMock()
    mock_ns = MagicMock()
    mock_ns.vector_count = 100
    mock_stats.namespaces = {"my_col": mock_ns}
    mock_index.describe_index_stats.return_value = mock_stats
    assert store.count() == 100

def test_delete(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    assert store.delete(ids=["id1", "id2"]) is True
    mock_index.delete.assert_called_once_with(ids=["id1", "id2"], namespace="my_col")
    
    # Empty ids returns None
    assert store.delete(ids=[]) is None

def test_reset_collection(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    store.reset_collection()
    mock_index.delete.assert_called_once_with(delete_all=True, namespace="my_col")
    
    store.delete_collection()
    assert mock_index.delete.call_count == 2

def test_query(mock_index):
    store = PineconeVectorStore(
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    
    mock_index.search.return_value = {
        "result": {
            "hits": [
                {"fields": {"text": "hello"}, "_score": 0.9}
            ]
        }
    }
    
    docs = store.query(query_text="hello")
    assert len(docs) == 1
    assert docs[0].page_content == "hello"

def test_from_texts(mock_index):
    store = PineconeVectorStore.from_texts(
        texts=["hello"],
        collection_name="my_col",
        api_key="my_key",
        index_name="my_idx"
    )
    assert isinstance(store, PineconeVectorStore)
    assert mock_index.upsert_records.call_count == 1
