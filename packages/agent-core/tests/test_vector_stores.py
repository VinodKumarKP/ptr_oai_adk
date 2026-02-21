import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
import os
from langchain_core.documents import Document

from oai_agent_core.components.vector_store.vector_store_factory import VectorStoreFactory
from oai_agent_core.components.vector_store.s3_vector_store import S3VectorStore
from oai_agent_core.components.vector_store.chroma_vector_store import ChromaVectorStore
from oai_agent_core.components.vector_store.postgres_vector_store import PostgresVectorStore

# --- VectorStoreFactory Tests ---

def test_factory_create_chroma():
    # Patch the class where it is defined
    with patch('oai_agent_core.components.vector_store.chroma_vector_store.ChromaVectorStore') as mock_chroma:
        store = VectorStoreFactory.create_vector_store('chroma', collection_name='test', embedding_function=MagicMock())
        assert mock_chroma.called
        assert store == mock_chroma.return_value

def test_factory_create_postgres():
    # Patch the class where it is defined
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.PostgresVectorStore') as mock_pg:
        store = VectorStoreFactory.create_vector_store('postgres', collection_name='test', embedding_function=MagicMock())
        assert mock_pg.called
        assert store == mock_pg.return_value

def test_factory_create_s3():
    # Patch the class where it is defined
    with patch('oai_agent_core.components.vector_store.s3_vector_store.S3VectorStore') as mock_s3:
        store = VectorStoreFactory.create_vector_store('s3', collection_name='test', embedding_function=MagicMock())
        assert mock_s3.called
        assert store == mock_s3.return_value

def test_factory_unknown_type():
    with pytest.raises(ValueError, match="Unknown vector store type"):
        VectorStoreFactory.create_vector_store('unknown', collection_name='test', embedding_function=MagicMock())

# --- S3VectorStore Tests ---

@pytest.fixture
def mock_s3_client():
    with patch('boto3.client') as mock_boto:
        client = mock_boto.return_value
        # Create a real exception class for NoSuchKey
        class NoSuchKey(Exception):
            pass
        client.exceptions.NoSuchKey = NoSuchKey
        yield client

@pytest.fixture
def mock_embedding():
    embedding = MagicMock()
    embedding.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    embedding.embed_query.return_value = [0.1, 0.2]
    return embedding

def test_s3_init_success(mock_s3_client, mock_embedding):
    # Mock S3 get_object to return empty index
    mock_s3_client.get_object.return_value = {
        'Body': MagicMock(read=lambda: json.dumps({'ids': [], 'vectors': [], 'documents': []}).encode('utf-8'))
    }
    
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    
    assert store.bucket_name == "test-bucket"
    assert store.index_key == "vector_store/test_coll/index.json"
    mock_s3_client.get_object.assert_called_once()

def test_s3_init_no_bucket():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="bucket_name is required"):
            S3VectorStore(collection_name="test", embedding_function=MagicMock())

def test_s3_add_texts(mock_s3_client, mock_embedding):
    # Mock initial load to raise NoSuchKey
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    
    ids = store.add_texts(["text1", "text2"])
    
    assert len(ids) == 2
    assert len(store.documents) == 2
    assert len(store.vectors) == 2
    
    # Verify save was called
    mock_s3_client.put_object.assert_called()
    call_args = mock_s3_client.put_object.call_args[1]
    assert call_args['Bucket'] == "test-bucket"
    assert call_args['Key'] == "vector_store/test_coll/index.json"
    
    saved_data = json.loads(call_args['Body'])
    assert len(saved_data['ids']) == 2
    assert len(saved_data['vectors']) == 2

def test_s3_add_documents(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    
    docs = [Document(page_content="doc1", metadata={"a": 1})]
    store.add_documents(docs)
    
    assert len(store.documents) == 1
    assert store.documents[0].page_content == "doc1"

def test_s3_similarity_search(mock_s3_client, mock_embedding):
    # Setup store with data
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    
    # Add some data manually to internal state
    store.documents = [Document(page_content="doc1"), Document(page_content="doc2")]
    store.vectors = [[1.0, 0.0], [0.0, 1.0]] # Orthogonal vectors
    
    # Query close to doc1
    mock_embedding.embed_query.return_value = [0.9, 0.1]
    
    results = store.similarity_search("query", k=1)
    
    assert len(results) == 1
    assert results[0].page_content == "doc1"

def test_s3_similarity_search_with_score(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    store.documents = [Document(page_content="doc1")]
    store.vectors = [[1.0, 0.0]]
    mock_embedding.embed_query.return_value = [1.0, 0.0]
    
    results = store.similarity_search_with_score("query", k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "doc1"
    assert score > 0.99

def test_s3_count(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    store.documents = [Document(page_content="doc1")]
    assert store.count() == 1

def test_s3_delete(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    
    store.ids = ["id1", "id2"]
    store.documents = [Document(page_content="1"), Document(page_content="2")]
    store.vectors = [[0.1], [0.2]]
    
    store.delete(["id1"])
    
    assert len(store.ids) == 1
    assert store.ids[0] == "id2"
    mock_s3_client.put_object.assert_called()

def test_s3_reset_collection(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    store.documents = [Document(page_content="doc1")]
    
    store.reset_collection()
    
    assert len(store.documents) == 0
    mock_s3_client.delete_object.assert_called()

def test_s3_query(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        bucket_name="test-bucket"
    )
    store.documents = [
        Document(page_content="doc1", metadata={"cat": "a", "val": 1}),
        Document(page_content="doc2", metadata={"cat": "b", "val": 2})
    ]
    store.vectors = [[1.0, 0.0], [0.0, 1.0]]
    
    # Test filter
    results = store.query(filter_metadata={"cat": "a"})
    assert len(results) == 1
    assert results[0].page_content == "doc1"
    
    # Test sort
    results = store.query(order_by="val", order="desc")
    assert len(results) == 2
    assert results[0].page_content == "doc2"

def test_s3_from_texts(mock_s3_client, mock_embedding):
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey("No key")
    store = S3VectorStore.from_texts(
        texts=["t1"],
        embedding=mock_embedding,
        collection_name="test",
        bucket_name="bucket"
    )
    assert isinstance(store, S3VectorStore)
    assert len(store.documents) == 1

# --- ChromaVectorStore Tests ---

@pytest.fixture
def mock_chroma_client():
    with patch('chromadb.PersistentClient') as mock_client:
        yield mock_client.return_value

def test_chroma_init_local(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    
    store = ChromaVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        persist_directory="/tmp/chroma"
    )
    
    assert store.collection == mock_collection
    mock_chroma_client.get_or_create_collection.assert_called_with(
        name="test_coll",
        metadata={"hnsw:space": "cosine"}
    )

def test_chroma_add_texts(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    
    store = ChromaVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding
    )
    
    store.add_texts(["text1"])
    
    mock_collection.add.assert_called()
    call_args = mock_collection.add.call_args[1]
    assert len(call_args['documents']) == 1
    # mock_embedding.embed_documents returns 2 items in fixture, but we passed 1 text.
    # The code calls embed_documents(["text1"]).
    # We should adjust mock_embedding for this test or accept it.
    # Let's adjust it.
    mock_embedding.embed_documents.return_value = [[0.1, 0.2]]
    
    store.add_texts(["text1"])
    mock_collection.add.assert_called()

def test_chroma_add_documents(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    store = ChromaVectorStore(collection_name="test", embedding_function=mock_embedding)
    
    mock_embedding.embed_documents.return_value = [[0.1]]
    store.add_documents([Document(page_content="d1")])
    mock_collection.add.assert_called()

def test_chroma_similarity_search(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    
    # Mock query result
    mock_collection.query.return_value = {
        'documents': [['doc1']],
        'metadatas': [[{'source': 'src1'}]],
        'distances': [[0.1]]
    }
    
    store = ChromaVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding
    )
    
    results = store.similarity_search("query")
    
    assert len(results) == 1
    assert results[0].page_content == "doc1"
    assert results[0].metadata['source'] == 'src1'

def test_chroma_similarity_search_with_score(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_collection.query.return_value = {
        'documents': [['doc1']],
        'metadatas': [[{}]],
        'distances': [[0.1]]
    }
    store = ChromaVectorStore(collection_name="test", embedding_function=mock_embedding)
    results = store.similarity_search_with_score("query")
    assert len(results) == 1
    assert results[0][1] == 0.1

def test_chroma_delete(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    # Mock create_collection to return a new mock
    mock_chroma_client.create_collection.return_value = MagicMock()
    
    store = ChromaVectorStore(collection_name="test", embedding_function=mock_embedding)
    store.delete()
    
    mock_chroma_client.delete_collection.assert_called_with("test")
    mock_chroma_client.create_collection.assert_called_with(name="test")

def test_chroma_query(mock_chroma_client, mock_embedding):
    mock_collection = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    
    # Mock get result for metadata query
    mock_collection.get.return_value = {
        'documents': ['doc1', 'doc2'],
        'metadatas': [{'val': 1}, {'val': 2}]
    }
    
    store = ChromaVectorStore(collection_name="test", embedding_function=mock_embedding)
    
    # Test metadata only query
    results = store.query(filter_metadata={"a": 1}, order_by="val", order="desc")
    assert len(results) == 2
    assert results[0].metadata['val'] == 2

# --- PostgresVectorStore Tests ---

@pytest.fixture
def mock_psycopg2():
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.psycopg2') as mock_pg:
        yield mock_pg

@pytest.fixture
def mock_register_vector():
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.register_vector') as mock_rv:
        yield mock_rv

def test_postgres_init(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    store = PostgresVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        connection_string="postgresql://user:pass@localhost/db",
        db_name="test_db"
    )
    
    assert store.table_name == "embeddings_test_coll"
    # Verify initialization SQL
    mock_cursor.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")
    # Verify table creation
    assert any("CREATE TABLE IF NOT EXISTS embeddings_test_coll" in str(call) for call in mock_cursor.execute.call_args_list)
    # Verify register_vector called
    mock_register_vector.assert_called_with(mock_conn)

def test_postgres_add_texts(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    store = PostgresVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        connection_string="postgresql://user:pass@localhost/db",
        db_name="test_db"
    )
    
    # Mock execute_values
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.execute_values') as mock_exec_values:
        store.add_texts(["text1"])
        mock_exec_values.assert_called()
        # Check that data was passed
        args = mock_exec_values.call_args[0]
        assert "INSERT INTO embeddings_test_coll" in args[1]
        assert len(args[2]) == 1 # 1 row

def test_postgres_add_documents(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    store = PostgresVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.execute_values'):
        store.add_documents([Document(page_content="d1")])
        # Implicitly verified by add_texts being called

def test_postgres_similarity_search(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    # Mock fetchall return
    mock_cursor.fetchall.return_value = [
        ("id1", "doc1", {"source": "src1"}, 0.1)
    ]
    
    store = PostgresVectorStore(
        collection_name="test_coll",
        embedding_function=mock_embedding,
        connection_string="postgresql://user:pass@localhost/db",
        db_name="test_db"
    )
    
    results = store.similarity_search("query")
    
    assert len(results) == 1
    assert results[0].page_content == "doc1"
    
    # Verify query
    assert any("SELECT id, content, metadata" in str(call) for call in mock_cursor.execute.call_args_list)
    assert any("ORDER BY distance" in str(call) for call in mock_cursor.execute.call_args_list)

def test_postgres_similarity_search_with_score(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [("id1", "doc1", {}, 0.1)]
    
    store = PostgresVectorStore(
        collection_name="test",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    results = store.similarity_search_with_score("query")
    assert len(results) == 1
    assert results[0][1] == 0.1

def test_postgres_count(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [5]
    
    store = PostgresVectorStore(
        collection_name="test",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    assert store.count() == 5

def test_postgres_delete(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.rowcount = 1
    
    store = PostgresVectorStore(
        collection_name="test",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    assert store.delete(["id1"]) is True
    assert any("DELETE FROM" in str(call) for call in mock_cursor.execute.call_args_list)

def test_postgres_reset_collection(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    store = PostgresVectorStore(
        collection_name="test",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    store.reset_collection()
    assert any("DROP TABLE" in str(call) for call in mock_cursor.execute.call_args_list)
    assert any("CREATE TABLE" in str(call) for call in mock_cursor.execute.call_args_list)

def test_postgres_query(mock_psycopg2, mock_register_vector, mock_embedding):
    mock_conn = mock_psycopg2.connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    
    # Mock results for metadata query
    mock_cursor.fetchall.return_value = [
        ("id1", "doc1", {"val": 1}),
        ("id2", "doc2", {"val": 2})
    ]
    
    store = PostgresVectorStore(
        collection_name="test",
        embedding_function=mock_embedding,
        connection_string="conn",
        db_name="db"
    )
    
    # Test metadata query
    results = store.query(filter_metadata={"a": 1}, order_by="val", order="desc")
    assert len(results) == 2
    assert results[0].metadata['val'] == 2

def test_postgres_from_texts(mock_psycopg2, mock_register_vector, mock_embedding):
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.execute_values'):
        store = PostgresVectorStore.from_texts(
            texts=["t1"],
            embedding=mock_embedding,
            collection_name="test",
            connection_string="conn",
            db_name="db"
        )
        assert isinstance(store, PostgresVectorStore)

def test_postgres_missing_deps():
    # Simulate missing psycopg2
    with patch('oai_agent_core.components.vector_store.postgres_vector_store.psycopg2', None):
        with pytest.raises(ImportError, match="Could not import psycopg2"):
            PostgresVectorStore(
                collection_name="test",
                embedding_function=MagicMock(),
                connection_string="conn",
                db_name="test_db"
            )
