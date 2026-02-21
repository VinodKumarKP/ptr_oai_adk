import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory
from langchain_core.documents import Document

class ConcreteKBFactory(BaseKnowledgeBaseFactory):
    def create_tool(self, name, description):
        return "tool"
    def create_load_tool(self, name, description):
        return "load_tool"

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    # Mock similarity_search_with_score to return list of (doc, score) tuples
    store.similarity_search_with_score.return_value = []
    return store

@pytest.fixture
def mock_document_loader(mock_vector_store):
    loader = MagicMock()
    loader_instance = MagicMock()
    loader.return_value = loader_instance
    loader_instance.load_db.return_value = mock_vector_store
    return loader

@pytest.fixture
def valid_config():
    return [
        {
            'name': 'default_knowledge_base',
            'settings': {
                'db_name': 'test_db',
                'embedding_model_id': 'test_model',
                'persist_directory': '/tmp',
            },
            'retrieval_settings': {
                'score_threshold': 0.7
            },
            'data_sources': [
                {'path': 'doc1.txt'}
            ]
        }
    ]

@pytest.fixture
def factory(mock_vector_store, mock_document_loader, valid_config):
    # Mock _create_embeddings to avoid litellm import
    with patch('oai_agent_core.core.base_knowledge_base_factory.BaseKnowledgeBaseFactory._create_embeddings', return_value="embeddings"):
        # Mock VectorStoreFactory to return our mock_vector_store
        with patch('oai_agent_core.core.base_knowledge_base_factory.VectorStoreFactory.create_vector_store', return_value=mock_vector_store):
            return ConcreteKBFactory(
                knowledge_base_config=valid_config,
                document_loader=mock_document_loader,
                vector_store=mock_vector_store,
                project_root="/tmp"
            )

class TestBaseKnowledgeBaseFactory:

    def test_init_success(self, factory, mock_vector_store):
        assert factory.vector_store == mock_vector_store
        assert factory.project_root == "/tmp"
        
        # Check if the KB was created and registered
        assert 'default_knowledge_base' in factory.knowledge_base_tools
        kb_data = factory.knowledge_base_tools['default_knowledge_base']
        assert kb_data['vector_store'] == mock_vector_store
        assert kb_data['retrieval_settings']['score_threshold'] == 0.7
        
        assert factory.similarity_threshold == 0.7

    def test_init_missing_settings(self, mock_document_loader):
        config = [{'custom_knowledge_base': {'docs': ['doc1.txt']}}]
        
        with patch('oai_agent_core.core.base_knowledge_base_factory.BaseKnowledgeBaseFactory._create_embeddings', return_value="embeddings"):
             with patch('oai_agent_core.core.base_knowledge_base_factory.VectorStoreFactory.create_vector_store'):
                factory = ConcreteKBFactory(
                    knowledge_base_config=config,
                    document_loader=mock_document_loader
                )
                assert factory is not None

    def test_initialize_vector_store(self, mock_document_loader, valid_config, mock_vector_store):
        with patch('oai_agent_core.core.base_knowledge_base_factory.BaseKnowledgeBaseFactory._create_embeddings', return_value="embeddings"):
            with patch('oai_agent_core.core.base_knowledge_base_factory.VectorStoreFactory.create_vector_store', return_value=mock_vector_store):
                factory = ConcreteKBFactory(
                    knowledge_base_config=valid_config,
                    document_loader=mock_document_loader,
                    project_root="/tmp"
                )
                
                # The document loader should be called to load docs
                assert mock_document_loader.called
                mock_document_loader.return_value.load_db.assert_called()

    def test_search_custom_knowledge_base_no_store(self, valid_config, mock_document_loader):
        with patch('oai_agent_core.core.base_knowledge_base_factory.BaseKnowledgeBaseFactory._create_embeddings', return_value="embeddings"):
            with patch('oai_agent_core.core.base_knowledge_base_factory.VectorStoreFactory.create_vector_store', return_value=None):
                factory = ConcreteKBFactory(
                    knowledge_base_config=[], 
                    document_loader=mock_document_loader,
                    vector_store=None
                )
                factory.vector_store = None
                factory.knowledge_base_tools = {}
                assert factory.search_custom_knowledge_base("query") == "No knowledge base available."

    def test_search_custom_knowledge_base_no_results(self, factory, mock_vector_store):
        mock_vector_store.similarity_search_with_score.return_value = []
        assert factory.search_custom_knowledge_base("query") == "No relevant information found in the knowledge base."

    def test_search_custom_knowledge_base_with_results(self, factory, mock_vector_store):
        doc = Document(page_content="content", metadata={"source": "src"})
        mock_vector_store.similarity_search_with_score.return_value = [(doc, 0.1)]
        
        result = factory.search_custom_knowledge_base("query")
        assert "Content: content" in result
        assert "Source: src" in result
        assert "Relevance:" in result

    def test_search_custom_knowledge_base_threshold_filtering(self, factory, mock_vector_store):
        # Update the threshold in the registered tool settings
        # because search_knowledge_base prioritizes tool settings over factory.similarity_threshold
        if 'default_knowledge_base' in factory.knowledge_base_tools:
            factory.knowledge_base_tools['default_knowledge_base']['retrieval_settings']['score_threshold'] = 0.9
        
        # Also set the instance variable just in case
        factory.similarity_threshold = 0.9
        
        doc1 = Document(page_content="good", metadata={"source": "s1"})
        doc2 = Document(page_content="bad", metadata={"source": "s2"})
        
        # doc1: score 0.05 (very close) -> similarity ~0.975 -> KEEP
        # doc2: score 0.5 (far) -> similarity ~0.75 -> DROP (since 0.75 < 0.9)
        mock_vector_store.similarity_search_with_score.return_value = [
            (doc1, 0.05),
            (doc2, 0.5)
        ]
        
        result = factory.search_custom_knowledge_base("query")
        assert "Content: good" in result
        assert "Content: bad" not in result

    def test_search_custom_knowledge_base_with_analyzer(self, factory, mock_vector_store):
        factory.query_analyzer = MagicMock()
        factory.query_analyzer.analyze.return_value = ["q1", "q2"]
        
        doc1 = Document(page_content="c1", metadata={"source": "s1"})
        doc2 = Document(page_content="c2", metadata={"source": "s2"})
        
        mock_vector_store.similarity_search_with_score.side_effect = [
            [(doc1, 0.1)], 
            [(doc2, 0.1)]
        ]
        
        result = factory.search_custom_knowledge_base("query")
        
        assert "Content: c1" in result
        assert "Content: c2" in result
        assert factory.query_analyzer.analyze.called

    def test_create_embeddings_success(self, factory):
        with patch.dict('sys.modules', {'litellm': MagicMock(), 'langchain_core.embeddings': MagicMock()}):
            embeddings = factory._create_embeddings("model")
            assert embeddings is not None

    def test_create_embeddings_import_error(self, factory):
        with patch.dict('sys.modules', {'litellm': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                 embeddings = factory._create_embeddings("model")
                 assert embeddings is None

    def test_abstract_methods_not_implemented(self):
        with pytest.raises(TypeError):
            BaseKnowledgeBaseFactory([])
