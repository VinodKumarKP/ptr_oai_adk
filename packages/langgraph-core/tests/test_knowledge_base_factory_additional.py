"""Additional coverage for LangGraph KnowledgeBaseFactory."""
from unittest.mock import MagicMock

import pytest

import oai_agent_core.langgraph_core.components.knowledge.knowledge_base_factory as kbf
from oai_agent_core.langgraph_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory


@pytest.fixture
def factory(monkeypatch):
    """Create a factory with mocked dependencies."""
    monkeypatch.setattr(kbf, "tool", lambda **kw: (lambda f: f))
    kb_factory = KnowledgeBaseFactory(
        knowledge_base_config=[{"name": "kb"}],
        logger=MagicMock()
    )
    # Mock the base class methods
    kb_factory.search_knowledge_base = MagicMock(return_value="search result")
    kb_factory.load_documents = MagicMock(return_value="load result")
    return kb_factory


def test_create_tool_default_name(factory):
    """Test tool creation with default name."""
    tool = factory.create_tool()
    assert tool.__name__ == "search_knowledge_base"
    assert callable(tool)


def test_create_tool_custom_name(factory):
    """Test tool creation with custom name."""
    tool = factory.create_tool(name="my_search", description="My search tool")
    assert tool.__name__ == "my_search"
    assert tool.__doc__ == "My search tool"


def test_create_tool_invocation(factory):
    """Test calling the created tool."""
    tool = factory.create_tool(name="search")
    result = tool("python", source_list=["docs"], session_id="sess1")
    assert result == "search result"
    factory.search_knowledge_base.assert_called_once_with(
        "python", kb_name="search", source_list=["docs"], session_id="sess1"
    )


def test_create_tool_no_sources(factory):
    """Test tool call without source list."""
    tool = factory.create_tool()
    result = tool("query")
    assert result == "search result"
    factory.search_knowledge_base.assert_called_once()


def test_create_load_tool_default_name(factory):
    """Test load tool creation with default name."""
    tool = factory.create_load_tool()
    assert tool.__name__ == "load_knowledge_base_load_knowledge_base"


def test_create_load_tool_custom_name(factory):
    """Test load tool creation with custom name."""
    tool = factory.create_load_tool(name="docs", description="Load docs")
    assert tool.__name__ == "load_knowledge_base_docs"
    assert tool.__doc__ == "Load docs"


def test_create_load_tool_invocation(factory):
    """Test calling the load tool."""
    tool = factory.create_load_tool(name="kb")
    result = tool(["doc1.txt", "doc2.txt"], session_id="sess1")
    assert result == "load result"
    factory.load_documents.assert_called_once_with(
        ["doc1.txt", "doc2.txt"], "sess1"
    )
