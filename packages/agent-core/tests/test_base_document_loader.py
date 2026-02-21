import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, ANY
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.document_loaders import BaseLoader

from oai_agent_core.core.base_document_loader import BaseDocumentLoader, LoaderError


class MockVectorStore:
    """Mock vector store for testing"""
    def __init__(self, collection_name="test_collection"):
        self.collection_name = collection_name
        self._documents = []
        self._count = 0
        
    def add_documents(self, documents, ids=None):
        self._documents.extend(documents)
        self._count += len(documents)
        
    def count(self):
        return self._count
        
    def reset_collection(self):
        self._documents.clear()
        self._count = 0
        
    def similarity_search(self, query, k=4):
        return self._documents[:k]


class MockEmbedding(Embeddings):
    """Mock embedding for testing"""
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class ConcreteDocumentLoader(BaseDocumentLoader):
    """Concrete implementation for testing"""
    def __init__(self, **kwargs):
        self._reinitialize = kwargs.pop('reinitialize', True)
        super().__init__(**kwargs)
    
    def reinitialize_database(self) -> bool:
        return self._reinitialize


class TestBaseDocumentLoaderInit:
    """Test initialization of BaseDocumentLoader"""
    
    def test_init_with_vector_store(self):
        vector_store = MockVectorStore()
        loader = ConcreteDocumentLoader(vector_store=vector_store)
        assert loader.vector_store == vector_store
        assert loader.persist_directory is not None
    
    def test_init_with_embedding(self):
        embedding = MockEmbedding()
        loader = ConcreteDocumentLoader(embedding=embedding)
        assert loader.vector_store is None
        assert loader.persist_directory is not None
    
    def test_init_with_both(self):
        vector_store = MockVectorStore()
        embedding = MockEmbedding()
        loader = ConcreteDocumentLoader(vector_store=vector_store, embedding=embedding)
        assert loader.vector_store == vector_store
    
    def test_init_with_neither_raises_error(self):
        with pytest.raises(LoaderError, match="Either vector_store or embedding must be provided"):
            ConcreteDocumentLoader()
    
    def test_init_with_custom_persist_directory(self):
        vector_store = MockVectorStore()
        custom_dir = "/tmp/custom"
        loader = ConcreteDocumentLoader(vector_store=vector_store, persist_directory=custom_dir)
        assert loader.persist_directory == custom_dir
    
    def test_init_with_db_name(self):
        vector_store = MockVectorStore()
        loader = ConcreteDocumentLoader(db_name="custom_db", vector_store=vector_store)
        # db_name is passed but not stored as attribute in base class


class TestTransformDocumentsWithMetadata:
    """Test document transformation functionality"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_transform_empty_list(self, loader):
        result = loader.transform_documents_with_metadata([])
        assert result == []
    
    def test_transform_valid_documents(self, loader):
        docs = [
            Document(
                page_content="Test content 1",
                metadata={"title": "Doc 1", "summary": "Summary 1", "source": "file1.txt"}
            ),
            Document(
                page_content="Test content 2",
                metadata={"title": "Doc 2", "summary": "Summary 2", "source": "file2.txt"}
            )
        ]
        
        result = loader.transform_documents_with_metadata(docs)
        
        assert len(result) == 2
        assert "Title: Doc 1" in result[0].page_content
        assert "Summary: Summary 1" in result[0].page_content
        assert "Content: Test content 1" in result[0].page_content
        assert "Source: file1.txt" in result[0].page_content
    
    def test_transform_documents_missing_metadata(self, loader):
        docs = [
            Document(page_content="Test content", metadata={}),
            Document(page_content="Test content 2")
        ]
        
        result = loader.transform_documents_with_metadata(docs)
        
        assert len(result) == 2
        assert "Title: Unknown Title" in result[0].page_content
        assert "Summary: No summary available" in result[0].page_content
        assert "Source: Unknown Source" in result[0].page_content
    
    def test_transform_documents_invalid_input_type(self, loader):
        with pytest.raises(TypeError, match="original_chunks must be a list"):
            loader.transform_documents_with_metadata("not a list")
    
    def test_transform_documents_non_document_objects(self, loader):
        docs = [
            Document(page_content="Valid doc", metadata={}),
            "not a document",
            {"not": "a document"}
        ]
        
        result = loader.transform_documents_with_metadata(docs)
        assert len(result) == 1  # Only valid document processed
    
    def test_transform_documents_missing_attributes(self, loader):
        # Create a mock document-like object without required attributes
        class FakeDocument:
            pass
        
        docs = [
            Document(page_content="Valid doc", metadata={}),
            FakeDocument(),
            FakeDocument()
        ]
        
        result = loader.transform_documents_with_metadata(docs)
        assert len(result) == 1  # Only valid document processed
    
    def test_transform_documents_all_fail(self, loader):
        docs = ["not", "valid", "documents"]
        
        with pytest.raises(LoaderError, match="All document transformations failed"):
            loader.transform_documents_with_metadata(docs)
    
    def test_transform_documents_exception_handling(self, loader):
        # Create a document that will cause an exception during transformation
        class ProblematicDocument:
            def __init__(self):
                self.page_content = "test"
            
            @property
            def metadata(self):
                raise Exception("Test error")
        
        problematic_doc = ProblematicDocument()
        
        with patch.object(loader, 'logger') as mock_logger:
            docs = [problematic_doc]
            with pytest.raises(LoaderError, match="All document transformations failed"):
                loader.transform_documents_with_metadata(docs)


class TestGetLoaderForFile:
    """Test file loader selection"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_get_loader_pdf(self, loader):
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_loader_class = MagicMock()
            mock_module.PyPDFLoader = mock_loader_class
            mock_import.return_value = mock_module
            
            result = loader._get_loader_for_file("test.pdf")
            mock_loader_class.assert_called_once_with("test.pdf")
    
    def test_get_loader_txt_with_encoding(self, loader):
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_loader_class = MagicMock()
            mock_module.TextLoader = mock_loader_class
            mock_import.return_value = mock_module
            
            result = loader._get_loader_for_file("test.txt")
            mock_loader_class.assert_called_once_with("test.txt", encoding='utf-8')
    
    def test_get_loader_unsupported_extension(self, loader):
        with pytest.raises(LoaderError, match="Unsupported file type 'xyz'"):
            loader._get_loader_for_file("test.xyz")
    
    def test_get_loader_no_extension(self, loader):
        with pytest.raises(LoaderError, match="Unsupported file type ''"):
            loader._get_loader_for_file("test")
    
    def test_get_loader_import_error(self, loader):
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(LoaderError, match="Required dependency for pdf files not found"):
                loader._get_loader_for_file("test.pdf")
    
    def test_get_loader_initialization_error(self, loader):
        with patch('builtins.__import__') as mock_import:
            mock_module = MagicMock()
            mock_loader_class = MagicMock(side_effect=Exception("Init failed"))
            mock_module.PyPDFLoader = mock_loader_class
            mock_import.return_value = mock_module
            
            # The error message includes the file path, not just extension
            with pytest.raises(LoaderError, match="Failed to initialize loader for test.pdf"):
                loader._get_loader_for_file("test.pdf")


class TestGetLoader:
    """Test general loader selection"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_get_loader_base_loader_instance(self, loader):
        mock_loader = MagicMock(spec=BaseLoader)
        result = loader._get_loader(mock_loader)
        assert result == mock_loader
    
    def test_get_loader_glob_pattern(self, loader):
        with patch('oai_agent_core.core.base_document_loader.DirectoryLoader') as mock_dir_loader:
            result = loader._get_loader("path/to/*.txt")
            mock_dir_loader.assert_called_once_with("path/to", glob="*.txt", loader_kwargs=None)
    
    def test_get_loader_glob_pattern_current_dir(self, loader):
        with patch('oai_agent_core.core.base_document_loader.DirectoryLoader') as mock_dir_loader:
            result = loader._get_loader("*.txt")
            mock_dir_loader.assert_called_once_with(".", glob="*.txt", loader_kwargs=None)
    
    def test_get_loader_wikipedia_wiki_prefix(self, loader):
        with patch('langchain_community.document_loaders.WikipediaLoader') as mock_wiki_loader:
            result = loader._get_loader("wiki/Python programming")
            mock_wiki_loader.assert_called_once_with(query="Python programming")
    
    def test_get_loader_wikipedia_query_prefix(self, loader):
        with patch('langchain_community.document_loaders.WikipediaLoader') as mock_wiki_loader:
            result = loader._get_loader("query: Python programming")
            mock_wiki_loader.assert_called_once_with(query="Python programming")
    
    def test_get_loader_directory(self, loader):
        with patch('os.path.isdir', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.DirectoryLoader') as mock_dir_loader:
            result = loader._get_loader("/path/to/dir")
            mock_dir_loader.assert_called_once_with("/path/to/dir", loader_kwargs=None)
    
    def test_get_loader_file(self, loader):
        with patch('os.path.isfile', return_value=True), \
             patch.object(loader, '_get_loader_for_file') as mock_file_loader:
            result = loader._get_loader("/path/to/file.txt")
            mock_file_loader.assert_called_once_with("/path/to/file.txt", loader_settings=None)
    
    def test_get_loader_unsupported_object(self, loader):
        with pytest.raises(LoaderError, match="Unsupported object type or path"):
            loader._get_loader(123)


class TestLoadDocuments:
    """Test document loading functionality"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_load_documents_empty_list(self, loader):
        with pytest.raises(ValueError, match="file_list cannot be empty"):
            loader._load_documents([])
    
    def test_load_documents_invalid_type(self, loader):
        with pytest.raises(TypeError, match="file_list must be a list"):
            loader._load_documents("not a list")
    
    def test_load_documents_success(self, loader):
        mock_loader = MagicMock()
        mock_docs = [Document(page_content="test", metadata={"source": "test.txt"})]
        mock_loader.load.return_value = mock_docs
        
        with patch.object(loader, '_get_loader', return_value=mock_loader):
            result = loader._load_documents(["test.txt"])
            
        assert len(result) == 1
        assert result[0].page_content == "test"
    
    def test_load_documents_directory_loader(self, loader):
        from langchain_community.document_loaders import DirectoryLoader
        mock_dir_loader = MagicMock(spec=DirectoryLoader)
        mock_file_loader = MagicMock()
        mock_docs = [Document(page_content="test", metadata={})]
        mock_file_loader.load.return_value = mock_docs
        
        with patch.object(loader, '_get_loader') as mock_get_loader, \
             patch('glob.glob', return_value=["file1.txt"]):
            
            # First call returns directory loader, second call returns file loader
            mock_get_loader.side_effect = [mock_dir_loader, mock_file_loader]
            
            result = loader._load_documents(["*.txt"])
            
        assert len(result) == 1
    
    def test_load_documents_no_content(self, loader):
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        
        with patch.object(loader, '_get_loader', return_value=mock_loader):
            with pytest.raises(LoaderError, match="No documents could be loaded"):
                loader._load_documents(["empty.txt"])
    
    def test_load_documents_partial_failure(self, loader):
        mock_loader1 = MagicMock()
        mock_loader1.load.return_value = [Document(page_content="success", metadata={})]
        
        mock_loader2 = MagicMock()
        mock_loader2.load.side_effect = Exception("Load failed")
        
        with patch.object(loader, '_get_loader', side_effect=[mock_loader1, mock_loader2]):
            result = loader._load_documents(["success.txt", "fail.txt"])
            
        assert len(result) == 1
        assert result[0].page_content == "success"


class TestGenerateChunk:
    """Test chunk generation"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_generate_chunk_success(self, loader):
        # Use MagicMock with spec=Document to ensure isinstance checks pass
        # and attributes are available
        mock_doc = MagicMock(spec=Document)
        mock_doc.page_content = "test content"
        mock_doc.metadata = {}
        mock_docs = [mock_doc]
        
        with patch.object(loader, '_load_documents', return_value=mock_docs), \
             patch('oai_agent_core.core.base_document_loader.os.path.exists', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.os.path.getsize', return_value=100), \
             patch.object(loader, '_split_text', return_value=mock_docs):
            result, loaded_files = loader._generate_chunk({"test.txt": {
                "chunk_size": 2000
            }})
            
        assert len(result) == 1
        assert result[0].page_content == "test content"
        assert "test.txt" in loaded_files
    
    def test_generate_chunk_empty_docs(self, loader):
        with patch.object(loader, '_load_documents', return_value=[]), \
             patch('oai_agent_core.core.base_document_loader.os.path.exists', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.os.path.getsize', return_value=100):
            
            # Should return empty list and loaded files, because loaded_files is populated
            chunks, loaded_files = loader._generate_chunk({"test.txt": {
                "chunk_size": 2000
            }})
            assert chunks == []
            assert "test.txt" in loaded_files
    
    def test_generate_chunk_invalid_type(self, loader):
        with patch.object(loader, '_load_documents', return_value="not a list"), \
             patch('oai_agent_core.core.base_document_loader.os.path.exists', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.os.path.getsize', return_value=100):
            with pytest.raises(AttributeError):
                loader._generate_chunk({"test.txt": {
                "chunk_size": 2000
            }})
    
    def test_generate_chunk_invalid_documents(self, loader):
        invalid_docs = ["not a document", Document(page_content="", metadata={})]
        
        with patch.object(loader, '_load_documents', return_value=invalid_docs), \
             patch('oai_agent_core.core.base_document_loader.os.path.exists', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.os.path.getsize', return_value=100):
            with patch.object(loader, '_split_text', return_value=invalid_docs):
                # Should return empty list and loaded files, not raise LoaderError
                # because we have loaded_files populated
                chunks, loaded_files = loader._generate_chunk({
                    "test.txt": {
                        "chunk_size": 2000
                    }})
                assert chunks == []
                assert "test.txt" in loaded_files
    
    def test_generate_chunk_filters_empty_content(self, loader):
        docs = [
            Document(page_content="valid content", metadata={}),
            Document(page_content="", metadata={}),
            Document(page_content="   ", metadata={})
        ]
        
        with patch.object(loader, '_load_documents', return_value=docs), \
             patch('oai_agent_core.core.base_document_loader.os.path.exists', return_value=True), \
             patch('oai_agent_core.core.base_document_loader.os.path.getsize', return_value=100), \
             patch.object(loader, '_split_text', return_value=docs):
            result, loaded_files = loader._generate_chunk({"test.txt": {
                "chunk_size": 2000
            }})
            
        assert len(result) == 1
        assert result[0].page_content == "valid content"


class TestLoadDocumentsPublic:
    """Test public load_documents method"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())

    def test_load_documents_saves_loaded_files(self, loader):
        mock_docs = [Document(page_content="test", metadata={})]
        loaded_files = {"test.txt": 100}
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, loaded_files)), \
             patch.object(loader, '_save_loaded_files') as mock_save:
            
            loader.load_documents({"test.txt": {}})
            
            mock_save.assert_called_once_with(loaded_files)

    def test_load_documents_does_not_save_on_failure(self, loader):
        mock_docs = [Document(page_content="test", metadata={})]
        loaded_files = {"test.txt": 100}
        
        # Mock add_documents to raise exception
        loader.vector_store.add_documents = MagicMock(side_effect=Exception("DB Error"))
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, loaded_files)), \
             patch.object(loader, '_save_loaded_files') as mock_save:
            
            with pytest.raises(Exception, match="DB Error"):
                loader.load_documents({"test.txt": {}})
            
            mock_save.assert_not_called()
            
    def test_load_documents_no_chunks(self, loader):
        with patch.object(loader, '_generate_chunk', return_value=([], {})), \
             patch.object(loader, '_save_loaded_files') as mock_save:
            
            loader.load_documents({"test.txt": {}})
            
            mock_save.assert_not_called()
            
    def test_load_documents_batches(self, loader):
        # Create 150 documents to test batching (batch size is 50)
        mock_docs = [Document(page_content=f"test {i}", metadata={}) for i in range(150)]
        loaded_files = {"test.txt": 100}
        
        # Mock add_documents on the instance
        loader.vector_store.add_documents = MagicMock()
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, loaded_files)), \
             patch.object(loader, '_save_loaded_files') as mock_save:
            
            loader.load_documents({"test.txt": {}})
            
            # Should be called 3 times for 150 docs with batch size 50
            assert loader.vector_store.add_documents.call_count == 3
            mock_save.assert_called_once_with(loaded_files)


class TestVectorStoreOperations:
    """Test vector store related operations"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore())
    
    def test_get_document_count_with_count_method(self, loader):
        loader.vector_store._count = 5
        assert loader._get_document_count() == 5
    
    def test_get_document_count_with_collection_count(self, loader):
        loader.vector_store = MagicMock(spec=['_collection'])
        loader.vector_store._collection = MagicMock()
        loader.vector_store._collection.count.return_value = 10
        
        assert loader._get_document_count() == 10
    
    def test_get_document_count_no_method(self, loader):
        loader.vector_store = MagicMock(spec=[])
        assert loader._get_document_count() == 0
    
    def test_get_collection_name_with_collection_name(self, loader):
        assert loader._get_collection_name() == "test_collection"
    
    def test_get_collection_name_with_collection_object(self, loader):
        loader.vector_store._collection = MagicMock()
        loader.vector_store._collection.name = "collection_from_object"
        del loader.vector_store.collection_name
        
        assert loader._get_collection_name() == "collection_from_object"
    
    def test_get_collection_name_unknown(self, loader):
        del loader.vector_store.collection_name
        assert loader._get_collection_name() == "unknown"


class TestLoadDb:
    """Test database loading functionality"""
    
    @pytest.fixture
    def loader(self):
        return ConcreteDocumentLoader(vector_store=MockVectorStore(), reinitialize=True)
    
    def test_load_db_reinitialize_empty_collection(self, loader):
        mock_docs = [Document(page_content="test", metadata={})]
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, {})):
            result = loader.load_db(["test.txt"])
            
        assert result == loader.vector_store
        assert loader.vector_store._count == 1
    
    def test_load_db_reinitialize_existing_collection(self, loader):
        loader.vector_store._count = 5  # Existing documents
        mock_docs = [Document(page_content="test", metadata={})]
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, {})):
            result = loader.load_db(["test.txt"])
            
        assert result == loader.vector_store
        # Should reset and add new documents
        assert loader.vector_store._count == 1
    
    def test_load_db_no_reinitialize_existing_collection(self, loader):
        loader._reinitialize = False
        loader.vector_store._count = 5  # Set existing documents
        
        # Mock the _generate_chunk to return some documents
        mock_docs = [Document(page_content="test", metadata={})]
        
        with patch.object(loader, '_generate_chunk', return_value=(mock_docs, {})), \
             patch.object(loader, '_get_document_count', return_value=5):
            result = loader.load_db(["test.txt"])
            
        assert result == loader.vector_store
    
    def test_load_db_no_chunks_generated(self, loader):
        with patch.object(loader, '_generate_chunk', return_value=([], {})):
            result = loader.load_db(["test.txt"])
            assert result == loader.vector_store
    
    def test_load_db_exception_handling(self, loader):
        with patch.object(loader, '_generate_chunk', side_effect=Exception("Chunk generation failed")):
            with pytest.raises(LoaderError, match="Failed to load documents into vector store"):
                loader.load_db(["test.txt"])


class TestQueryAndStats:
    """Test query and statistics functionality"""
    
    @pytest.fixture
    def loader(self):
        vector_store = MockVectorStore()
        vector_store._documents = [
            Document(page_content="test doc 1", metadata={}),
            Document(page_content="test doc 2", metadata={})
        ]
        return ConcreteDocumentLoader(vector_store=vector_store)
    
    def test_query_with_vector_store(self, loader):
        result = loader.query("test query", n_results=2)
        assert len(result) == 2
    
    def test_query_without_vector_store(self, loader):
        loader.vector_store = None
        result = loader.query("test query")
        assert result == []
    
    def test_get_collection_stats(self, loader):
        stats = loader.get_collection_stats()
        
        assert stats['name'] == "test_collection"
        assert stats['count'] == 0  # MockVectorStore count method returns _count
        assert 'metadata' in stats
    
    def test_get_collection_stats_with_metadata(self, loader):
        loader.vector_store.collection = MagicMock()
        loader.vector_store.collection.metadata = {"test": "metadata"}
        
        stats = loader.get_collection_stats()
        assert stats['metadata'] == {"test": "metadata"}
    
    def test_reset_collection_success(self, loader):
        loader.vector_store._count = 5
        loader.reset_collection()
        assert loader.vector_store._count == 0
    
    def test_reset_collection_no_support(self, loader):
        loader.vector_store = MagicMock(spec=['count', 'collection_name'])
        loader.reset_collection()
    
    def test_reset_collection_exception(self, loader):
        loader.vector_store.reset_collection = MagicMock(side_effect=Exception("Reset failed"))
        
        with pytest.raises(LoaderError, match="Failed to reset collection"):
            loader.reset_collection()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions"""
    
    def test_concrete_loader_reinitialize_method(self):
        loader = ConcreteDocumentLoader(vector_store=MockVectorStore(), reinitialize=False)
        assert loader.reinitialize_database() == False
        
        loader = ConcreteDocumentLoader(vector_store=MockVectorStore(), reinitialize=True)
        assert loader.reinitialize_database() == True
    
    def test_loader_error_exception(self):
        error = LoaderError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_transform_documents_non_string_page_content(self):
        loader = ConcreteDocumentLoader(vector_store=MockVectorStore())
        
        # Document with non-string page content - create manually to avoid validation
        doc = Document(page_content="123", metadata={"title": "Test"})
        # Manually set page_content to non-string after creation
        doc.page_content = 123
        
        result = loader.transform_documents_with_metadata([doc])
        
        assert len(result) == 1
        assert "Content: 123" in result[0].page_content
    
    def test_load_documents_with_source_metadata_addition(self):
        loader = ConcreteDocumentLoader(vector_store=MockVectorStore())
        
        # Test the source metadata addition logic in directory processing
        from langchain_community.document_loaders import DirectoryLoader
        mock_dir_loader = MagicMock(spec=DirectoryLoader)
        mock_file_loader = MagicMock()
        
        # Document without source metadata
        doc_without_source = Document(page_content="test", metadata={})
        mock_file_loader.load.return_value = [doc_without_source]
        
        with patch.object(loader, '_get_loader') as mock_get_loader, \
             patch('glob.glob', return_value=["file1.txt"]):
            
            mock_get_loader.side_effect = [mock_dir_loader, mock_file_loader]
            
            # This should trigger the directory processing path
            result = loader._load_documents(["*.txt"])
            
        # Note: The current implementation has a bug - it adds source to `documents` 
        # but extends `all_documents` with `tmp_documents`. This test documents the current behavior.
        assert len(result) == 1