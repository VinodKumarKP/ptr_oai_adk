import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.openai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory

@pytest.fixture
def factory():
    # Provide a valid config structure with 'settings'
    config = [{
        'name': 'test_kb',
        'settings': {
            'db_name': 'test_db',
            'embedding_model_id': 'test-model',
            'persist_directory': '/tmp'
        }
    }]
    return KnowledgeBaseFactory(
        knowledge_base_config=config,
        project_root="/tmp"
    )

def test_create_tool(factory):
    with patch('oai_agent_core.openai_core.components.knowledge.knowledge_base_factory.function_tool') as mock_dec:
        # Mock decorator behavior for when it's called with arguments
        def decorator_factory(name_override=None, description_override=None):
            def decorator(func):
                return func
            return decorator
            
        mock_dec.side_effect = decorator_factory
        
        tool = factory.create_tool(name="test_kb_tool", description="Test description")
        
        assert callable(tool)
        assert tool.__name__ == "search_knowledge_base_test_kb_tool"
        assert tool.__doc__ == "Test description"
        
        # Test tool execution
        factory.search_knowledge_base = MagicMock(return_value="results")
        assert tool("query") == "results"
        factory.search_knowledge_base.assert_called_with("query", kb_name="test_kb_tool", source_list=None, session_id=None)
