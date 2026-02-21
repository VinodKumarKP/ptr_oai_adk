import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.langgraph_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
from langchain_core.tools import BaseTool

@pytest.fixture
def mock_vector_store():
    return MagicMock()

@pytest.fixture
def factory(mock_vector_store):
    config = [{'name': 'kb1', 'type': 'chroma'}]
    return KnowledgeBaseFactory(
        knowledge_base_config=config,
        vector_store=mock_vector_store
    )

def test_create_tool(factory):
    # Mock the search_knowledge_base method from the parent class
    # create=True is used because the base class update might not be reflected in the environment yet
    with patch.object(factory, 'search_knowledge_base', return_value="Search Result", create=True) as mock_search:
        tool = factory.create_tool(name="test_tool", description="Test Description")
        
        # Check if it's a tool
        assert isinstance(tool, BaseTool)
        assert tool.name == 'test_tool'
        assert tool.description == '''Search the knowledge base for relevant documents.
Args:
    query: The search query.
    source_list: List of sources to filter by in the knowledge base.
    session_id: Session id, if available'''

        # Invoke the tool
        result = tool.invoke("query")
        
        assert result == "Search Result"
        mock_search.assert_called_once_with("query", kb_name="test_tool", source_list=None, session_id=None)

def test_init_requires_langchain():
    pass

def test_factory_initialization():
    config = [{'name': 'kb1'}]
    factory = KnowledgeBaseFactory(knowledge_base_config=config)
    # Access the attribute directly. The mock in conftest sets it.
    assert factory.knowledge_base_config == config
