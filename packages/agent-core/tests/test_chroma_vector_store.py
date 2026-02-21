import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from oai_agent_core.components.vector_store.chroma_vector_store import ChromaVectorStore

@pytest.fixture
def mock_embedding_function():
    ef = MagicMock()
    ef.embed_documents.return_value = [[0.1, 0.2]]
    ef.embed_query.return_value = [0.1, 0.2]
    return ef

@pytest.fixture
def mock_client():
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.create_collection.return_value = collection
    return client, collection

@pytest.fixture
def store(mock_embedding_function, mock_client):
    client, collection = mock_client
    
    with patch('chromadb.PersistentClient', return_value=client):
        store = ChromaVectorStore(
            collection_name="test_coll",
            embedding_function=mock_embedding_function,
            persist_directory="/tmp"
        )
        return store

def test_init_local(store):
    assert store.collection_name == "test_coll"
    assert store.client is not None

def test_init_remote(mock_embedding_function, mock_client):
    client, _ = mock_client
    
    with patch('chromadb.HttpClient', return_value=client) as mock_http:
        store = ChromaVectorStore(
            collection_name="test_coll",
            embedding_function=mock_embedding_function,
            host="localhost",
            port=8000
        )
        mock_http.assert_called_once()

def test_init_missing_args():
    with pytest.raises(ValueError, match="collection_name is required"):
        ChromaVectorStore(embedding_function=MagicMock())
        
    with pytest.raises(ValueError, match="embedding_function is required"):
        ChromaVectorStore(collection_name="test")

def test_add_documents(store):
    docs = [Document(page_content="text", metadata={"a": 1})]
    store.add_documents(docs)
    
    store.collection.add.assert_called_once()
    call_args = store.collection.add.call_args[1]
    assert call_args['documents'] == ["text"]
    assert call_args['metadatas'] == [{"a": 1}]

def test_add_documents_empty(store):
    store.add_documents([])
    store.collection.add.assert_not_called()

def test_similarity_search(store):
    store.collection.query.return_value = {
        'documents': [['res']],
        'metadatas': [[{'a': 1}]]
    }
    
    results = store.similarity_search("query")

    assert len(results) == 1
    assert results[0].page_content == "res"
    assert results[0].metadata == {'a': 1}

def test_count(store):
    store.collection.count.return_value = 5
    assert store.count() == 5

def test_delete_collection(store):
    store.delete_collection()
    store.client.delete_collection.assert_called_with("test_coll")
    store.client.create_collection.assert_called_with(name="test_coll")

def test_reset_collection(store):
    store.reset_collection()
    store.client.delete_collection.assert_called()

def test_delete(store):
    assert store.delete() is True
    store.client.delete_collection.assert_called()
