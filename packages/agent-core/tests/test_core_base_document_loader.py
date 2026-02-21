import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.core.base_document_loader import BaseDocumentLoader, LoaderError
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class ConcreteDocumentLoader(BaseDocumentLoader):
    def reinitialize_database(self):
        return False

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.count.return_value = 0
    store.collection_name = "test_coll"
    return store

@pytest.fixture
def mock_embedding():
    return MagicMock()

@pytest.fixture
def loader(mock_vector_store, mock_embedding):
    return ConcreteDocumentLoader(
        vector_store=mock_vector_store,
        embedding=mock_embedding,
        persist_directory="/tmp"
    )

def test_init_success(loader, mock_vector_store, mock_embedding):
    assert loader.vector_store == mock_vector_store
    assert loader.persist_directory == "/tmp"
    assert loader.reinitialize is False

def test_init_missing_args():
    with pytest.raises(LoaderError, match="Either vector_store or embedding"):
        ConcreteDocumentLoader(vector_store=None, embedding=None)

def test_transform_documents_with_metadata(loader):
    docs = [Document(page_content="content", metadata={"title": "Title"})]
    transformed = loader.transform_documents_with_metadata(docs)
    
    assert len(transformed) == 1
    assert "Title: Title" in transformed[0].page_content
    assert "Content: content" in transformed[0].page_content

def test_transform_documents_empty(loader):
    assert loader.transform_documents_with_metadata([]) == []

def test_transform_documents_invalid_input(loader):
    with pytest.raises(TypeError):
        loader.transform_documents_with_metadata("not a list")

def test_get_loader_for_file_unsupported(loader):
    with pytest.raises(LoaderError, match="Unsupported file type"):
        loader._get_loader_for_file("file.xyz")

def test_get_loader_for_file_txt(loader):
    # Mock import of langchain_community.document_loaders
    with patch('builtins.__import__') as mock_import:
        mock_module = MagicMock()
        mock_loader_cls = MagicMock()
        mock_module.TextLoader = mock_loader_cls
        mock_import.return_value = mock_module
        
        loader._get_loader_for_file("file.txt")
        mock_loader_cls.assert_called_with("file.txt", encoding='utf-8')

def test_get_loader_base_loader(loader):
    # Create a mock that is an instance of BaseLoader
    mock_base_loader = MagicMock(spec=BaseLoader)
    # isinstance(mock, Class) returns True if spec is set to Class
    assert loader._get_loader(mock_base_loader) == mock_base_loader

def test_get_loader_directory(loader):
    with patch('os.path.isdir', return_value=True), \
         patch('oai_agent_core.core.base_document_loader.DirectoryLoader') as MockDirLoader:
        
        loader._get_loader("/path/to/dir")
        MockDirLoader.assert_called_with("/path/to/dir", loader_kwargs=None)

def test_get_loader_glob(loader):
    with patch('oai_agent_core.core.base_document_loader.DirectoryLoader') as MockDirLoader:
        loader._get_loader("/path/*.txt")
        MockDirLoader.assert_called()

def test_load_documents_empty(loader):
    with pytest.raises(ValueError):
        loader._load_documents([])

def test_load_documents_invalid_type(loader):
    with pytest.raises(TypeError):
        loader._load_documents("not a list")

def test_load_documents_success(loader):
    with patch.object(loader, '_get_loader') as mock_get_loader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [Document(page_content="content")]
        mock_get_loader.return_value = mock_loader_instance
        
        docs = loader._load_documents(["file.txt"])
        assert len(docs) == 1
        assert docs[0].page_content == "content"

def test_generate_chunk_success(loader):
    with patch.object(loader, '_load_documents', return_value=[Document(page_content="content")]), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.getsize', return_value=100):
        chunks, loaded_files = loader._generate_chunk({"test.txt": {
                "chunk_size": 2000
            }})
        assert len(chunks) == 1

def test_get_document_count(loader, mock_vector_store):
    mock_vector_store.count.return_value = 10
    assert loader._get_document_count() == 10
    
    # Test fallback to _collection
    del mock_vector_store.count
    mock_vector_store._collection.count.return_value = 5
    assert loader._get_document_count() == 5

def test_get_collection_name(loader, mock_vector_store):
    mock_vector_store.collection_name = "my_coll"
    assert loader._get_collection_name() == "my_coll"

def test_load_db(loader, mock_vector_store):
    with patch.object(loader, '_load_documents', return_value=[Document(page_content="text")]), \
         patch.object(loader, '_generate_chunk', return_value=([Document(page_content="text")], {})):
        
        loader.load_db(["file.txt"])
        
        mock_vector_store.add_documents.assert_called()

def test_load_db_reinitialize(loader, mock_vector_store):
    loader.reinitialize = True
    mock_vector_store.count.return_value = 5
    
    with patch.object(loader, '_generate_chunk', return_value=([Document(page_content="text")], {})):
        loader.load_db(["file.txt"])
        
        mock_vector_store.reset_collection.assert_called()
        mock_vector_store.add_documents.assert_called()

def test_query(loader, mock_vector_store):
    mock_vector_store.similarity_search.return_value = ["res"]
    res = loader.query("q")
    assert res == ["res"]
    mock_vector_store.similarity_search.assert_called_with("q", k=5)

def test_get_collection_stats(loader, mock_vector_store):
    mock_vector_store.count.return_value = 10
    mock_vector_store.collection_name = "coll"
    mock_vector_store.collection.metadata = {"meta": "data"}
    
    stats = loader.get_collection_stats()
    assert stats['count'] == 10
    assert stats['name'] == "coll"
    assert stats['metadata'] == {"meta": "data"}

def test_reset_collection(loader, mock_vector_store):
    loader.reset_collection()
    mock_vector_store.reset_collection.assert_called()

def test_load_db_batching(loader, mock_vector_store):
    # Create 150 documents to test batching (batch size is 50)
    mock_docs = [Document(page_content=f"test {i}", metadata={}) for i in range(150)]
    
    with patch.object(loader, '_generate_chunk', return_value=(mock_docs, {})):
        loader.load_db(["test.txt"])
        
        # Should be called 3 times for 150 docs with batch size 50
        assert mock_vector_store.add_documents.call_count == 3
