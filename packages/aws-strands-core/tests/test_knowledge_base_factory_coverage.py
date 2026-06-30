"""Additional coverage for KnowledgeBaseFactory."""
from unittest.mock import MagicMock

import oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory as kbf
from oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory


def make(monkeypatch):
    # Patch the module-level @tool decorator to a passthrough that accepts kwargs.
    monkeypatch.setattr(kbf, "tool", lambda **kw: (lambda f: f))
    factory = KnowledgeBaseFactory(knowledge_base_config=[{"name": "kb"}], logger=MagicMock())
    # Base class (mocked in conftest) doesn't provide these; stub them.
    factory.search_knowledge_base = MagicMock(return_value="search result")
    factory.load_documents = MagicMock(return_value="loaded")
    return factory


def test_create_tool(monkeypatch):
    factory = make(monkeypatch)
    tool_fn = factory.create_tool(name="my_kb", description="My KB search")
    assert tool_fn.__name__ == "my_kb"
    assert tool_fn.__doc__ == "My KB search"
    # invoking the tool delegates to search_knowledge_base
    result = tool_fn("query", source_list=["s1"], session_id="sess")
    assert result == "search result"
    factory.search_knowledge_base.assert_called_once()


def test_create_load_tool(monkeypatch):
    factory = make(monkeypatch)
    load_fn = factory.create_load_tool(name="my_kb", description="Load my KB")
    assert load_fn.__name__ == "load_knowledge_base_my_kb"
    assert load_fn.__doc__ == "Load my KB"
    result = load_fn(["doc1.txt"], session_id="sess")
    assert result == "loaded"
    factory.load_documents.assert_called_once()


def test_create_tool_defaults(monkeypatch):
    factory = make(monkeypatch)
    tool_fn = factory.create_tool()
    assert tool_fn.__name__ == "search_knowledge_base"
