import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from oai_agent_core.core.base_memory_store import BaseMemoryStore

class TestBaseMemoryStore:
    @pytest.fixture
    def memory_config(self):
        return {
            'vector_store': {
                'type': 'chroma',
                'settings': {
                    'persist_directory': '/tmp/test_db',
                    'collection_name': 'test_collection'
                }
            },
            'embedding': {
                'model_id': 'test-model',
                'region_name': 'us-east-1'
            },
            'settings': {
                'max_recent_turns': 5,
                'max_relevant_turns': 3,
                'similarity_threshold': 0.6
            }
        }

    @pytest.fixture
    def mock_vector_store(self):
        return MagicMock()

    @pytest.fixture
    def memory_store(self, memory_config, mock_vector_store):
        with patch('oai_agent_core.core.base_memory_store.VectorStoreFactory.create_vector_store', return_value=mock_vector_store), \
             patch('oai_agent_core.core.base_memory_store.BaseMemoryStore._create_embeddings', return_value=MagicMock()):
            return BaseMemoryStore(memory_config, vector_store=mock_vector_store)

    def test_init(self, memory_config):
        with patch('oai_agent_core.core.base_memory_store.VectorStoreFactory.create_vector_store') as mock_create_vs, \
             patch('oai_agent_core.core.base_memory_store.BaseMemoryStore._create_embeddings') as mock_create_emb:
            
            store = BaseMemoryStore(memory_config)
            
            assert store.max_recent_turns == 5
            assert store.max_relevant_turns == 3
            assert store.similarity_threshold == 0.6
            mock_create_emb.assert_called_once()
            mock_create_vs.assert_called_once()

    def test_initialize_vector_store_with_persist_directory(self, memory_config):
        memory_config['vector_store']['settings']['persist_directory'] = 'relative/path'
        with patch('oai_agent_core.core.base_memory_store.VectorStoreFactory.create_vector_store') as mock_create_vs, \
             patch('oai_agent_core.core.base_memory_store.BaseMemoryStore._create_embeddings') as mock_create_emb:
            
            store = BaseMemoryStore(memory_config, project_root='/root')
            
            # Check if persist_directory was resolved correctly
            _, kwargs = mock_create_vs.call_args
            assert kwargs['persist_directory'] == '/root/relative/path'

    def test_create_embeddings_success(self, memory_store):
        with patch('builtins.__import__') as mock_import:
            mock_litellm = MagicMock()
            mock_import.return_value = mock_litellm
            
            embeddings = memory_store._create_embeddings('test-model')
            assert embeddings is not None

    def test_create_embeddings_import_error(self, memory_store):
        with patch('builtins.__import__', side_effect=ImportError):
            embeddings = memory_store._create_embeddings('test-model')
            assert embeddings is None

    def test_add_turn(self, memory_store, mock_vector_store):
        turn_id = memory_store.add_turn(
            session_id="session1",
            user_id="user1",
            user_message="Hello",
            agent_response="Hi there"
        )
        
        assert turn_id is not None
        mock_vector_store.add_texts.assert_called_once()
        call_args = mock_vector_store.add_texts.call_args
        assert call_args[1]['texts'][0] == "User: Hello\nAgent: Hi there"
        assert call_args[1]['metadatas'][0]['session_id'] == "session1"

    def test_get_recent_turns(self, memory_store, mock_vector_store):
        mock_vector_store.query.return_value = [
            MagicMock(metadata={'turn_id': '1', 'timestamp': '2023-01-01'}),
            MagicMock(metadata={'turn_id': '2', 'timestamp': '2023-01-02'})
        ]
        
        turns = memory_store._get_recent_turns("session1", "user1")
        
        assert len(turns) == 2
        # Should be reversed (chronological)
        assert turns[0]['turn_id'] == '2'
        assert turns[1]['turn_id'] == '1'

    def test_get_semantic_matches(self, memory_store, mock_vector_store):
        # Mock similarity search results
        doc1 = MagicMock()
        doc1.metadata = {'turn_id': '1', 'user_message': 'test'}
        
        # Setup mock to return tuple (doc, score)
        mock_vector_store.similarity_search_with_score.return_value = [(doc1, 0.1)]
        
        matches = memory_store._get_semantic_matches("query", "session1", "user1")
        
        assert len(matches) == 1
        assert matches[0]['turn_id'] == '1'
        assert 'similarity_score' in matches[0]

    def test_extract_text_from_response_string(self, memory_store):
        assert memory_store._extract_text_from_response("simple text") == "simple text"

    def test_extract_text_from_response_json(self, memory_store):
        json_str = '{"content": "json text"}'
        assert memory_store._extract_text_from_response(json_str) == "json text"

    def test_extract_text_from_response_dict(self, memory_store):
        resp_dict = {"content": [{"text": "part1"}, {"text": "part2"}]}
        assert memory_store._extract_text_from_response(resp_dict) == "part1\npart2"

    def test_normalize_score(self, memory_store):
        assert memory_store._normalize_score(0.0, 'cosine') == 1.0
        assert memory_store._normalize_score(2.0, 'cosine') == 0.0
        assert memory_store._normalize_score(0.0, 'euclidean') == 1.0
        assert memory_store._normalize_score(1.0, 'dot') == 1.0

    def test_detect_distance_type(self, memory_store):
        assert memory_store._detect_distance_type([0.5, 1.5]) == 'cosine'
        assert memory_store._detect_distance_type([-0.5, 0.5]) == 'dot'
        assert memory_store._detect_distance_type([100.0, 200.0]) == 'euclidean'

    def test_format_context_for_prompt(self, memory_store):
        recent = [{'user_message': 'hi', 'agent_response': 'hello'}]
        relevant = [{'user_message': 'help', 'agent_response': 'sure', 'similarity_score': 0.9}]
        
        context = memory_store.format_context_for_prompt(recent, relevant)
        
        assert "=== Recent Conversation ===" in context
        assert "=== Relevant Context from History ===" in context
        assert "User: hi" in context
        assert "User: help" in context

    def test_get_relevant_context(self, memory_store):
        with patch.object(memory_store, '_get_recent_turns', return_value=[{'turn_id': '1'}]), \
             patch.object(memory_store, '_get_semantic_matches', return_value=[{'turn_id': '2'}]):
            
            recent, relevant = memory_store.get_relevant_context("msg", "s1", "u1")
            
            assert len(recent) == 1
            assert len(relevant) == 1
