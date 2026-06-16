"""Tests for the anthropic_core KnowledgeBaseFactory tool builders.

The base ``__init__`` does heavy setup (vector stores / registries), so we build a
bare instance via ``__new__`` and stub the two methods the closures call. This
unit-tests ``create_tool`` / ``create_load_tool`` in isolation.
"""

import asyncio
import logging

from oai_agent_core.anthropic_core.components.knowledge.knowledge_base_factory import (
    KnowledgeBaseFactory,
)


def _factory():
    kf = KnowledgeBaseFactory.__new__(KnowledgeBaseFactory)
    kf.logger = logging.getLogger("test")
    return kf


# ── create_tool ───────────────────────────────────────────────────────────────

def test_create_tool_name_and_doc():
    kf = _factory()
    kf.search_knowledge_base = lambda query, kb_name: ["r1", "r2"]
    fn = kf.create_tool("my-kb", "search desc")
    assert fn.__name__ == "search_my_kb"   # dashes -> underscores
    assert fn.__doc__ == "search desc"


def test_create_tool_returns_joined_results():
    kf = _factory()
    kf.search_knowledge_base = lambda query, kb_name: ["alpha", "beta"]
    fn = kf.create_tool("kb", "d")
    out = asyncio.run(fn("q"))
    assert "alpha" in out and "beta" in out


def test_create_tool_no_results():
    kf = _factory()
    kf.search_knowledge_base = lambda query, kb_name: []
    fn = kf.create_tool("kb", "d")
    assert "No results found" in asyncio.run(fn("missing"))


def test_create_tool_error_is_caught():
    kf = _factory()

    def boom(query, kb_name):
        raise RuntimeError("kb down")
    kf.search_knowledge_base = boom
    fn = kf.create_tool("kb", "d")
    assert "Knowledge base search error" in asyncio.run(fn("q"))


# ── create_load_tool ──────────────────────────────────────────────────────────

def test_create_load_tool_name_and_load():
    kf = _factory()
    seen = {}
    kf.load_documents = lambda doc_list: seen.update(docs=doc_list)
    fn = kf.create_load_tool("my-kb", "load desc")
    assert fn.__name__ == "load_my_kb"
    out = asyncio.run(fn("a.txt, b.txt , c.txt"))
    assert "Loaded 3 document(s)" in out
    assert seen["docs"] == ["a.txt", "b.txt", "c.txt"]


def test_create_load_tool_error_is_caught():
    kf = _factory()

    def boom(doc_list):
        raise RuntimeError("disk full")
    kf.load_documents = boom
    fn = kf.create_load_tool("kb", "d")
    assert "Knowledge base load error" in asyncio.run(fn("x.txt"))
