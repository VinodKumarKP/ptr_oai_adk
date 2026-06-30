"""Additional coverage for BaseMemoryStore."""
import sys
import types
from unittest.mock import MagicMock

import pytest

from oai_agent_core.core.base_memory_store import BaseMemoryStore
import oai_agent_core.core.base_memory_store as bms


def make_store(vector_store=None, config=None):
    return BaseMemoryStore(
        memory_config=config or {},
        logger=MagicMock(),
        vector_store=vector_store or MagicMock(),
    )


def test_init_reads_settings():
    store = make_store(config={"settings": {"max_recent_turns": 9, "similarity_threshold": 0.9}})
    assert store.max_recent_turns == 9
    assert store.similarity_threshold == 0.9


def test_initialize_vector_store_embeddings_none(monkeypatch):
    monkeypatch.setattr(BaseMemoryStore, "_create_embeddings", lambda self, *a, **k: None)
    store = BaseMemoryStore(memory_config={}, logger=MagicMock(), vector_store=None)
    assert store.vector_store is None


def test_initialize_vector_store_creation(monkeypatch):
    monkeypatch.setattr(BaseMemoryStore, "_create_embeddings", lambda self, *a, **k: object())
    fake_vs = MagicMock()
    monkeypatch.setattr(bms.VectorStoreFactory, "create_vector_store", lambda *a, **k: fake_vs)
    store = BaseMemoryStore(memory_config={}, logger=MagicMock(), vector_store=None)
    assert store.vector_store is fake_vs


def test_initialize_vector_store_creation_failure(monkeypatch):
    monkeypatch.setattr(BaseMemoryStore, "_create_embeddings", lambda self, *a, **k: object())
    monkeypatch.setattr(
        bms.VectorStoreFactory, "create_vector_store",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        BaseMemoryStore(memory_config={}, logger=MagicMock(), vector_store=None)


def test_initialize_persist_directory_relative(monkeypatch):
    monkeypatch.setattr(BaseMemoryStore, "_create_embeddings", lambda self, *a, **k: object())
    captured = {}

    def fake_create(vtype, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(bms.VectorStoreFactory, "create_vector_store", fake_create)
    cfg = {"vector_store": {"settings": {"persist_directory": "rel/dir"}}}
    BaseMemoryStore(memory_config=cfg, logger=MagicMock(), project_root="/root", vector_store=None)
    assert captured["persist_directory"] == "/root/rel/dir"


def test_create_embeddings_import_error(monkeypatch):
    store = make_store()
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert store._create_embeddings("amazon.titan") is None


def test_create_embeddings_success(monkeypatch):
    store = make_store()
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.embedding = lambda **k: {"data": [{"embedding": [0.1]}]}
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    emb = store._create_embeddings("amazon.titan-embed-text-v1")
    assert emb is not None
    assert emb.model_id == "bedrock/amazon.titan-embed-text-v1"
    assert emb.embed_query("hi") == [0.1]
    assert emb.embed_documents(["a"]) == [[0.1]]


def test_create_embeddings_no_bedrock_prefix(monkeypatch):
    store = make_store()
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.embedding = lambda **k: {"data": [{"embedding": [0.2]}]}
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    emb = store._create_embeddings("openai/text-embedding")
    assert emb.model_id == "openai/text-embedding"


def test_create_turn_id_deterministic():
    store = make_store()
    a = store._create_turn_id("s", "u", "t")
    b = store._create_turn_id("s", "u", "t")
    assert a == b


def test_add_turn_success():
    vs = MagicMock()
    store = make_store(vector_store=vs)
    turn_id = store.add_turn("s", "u", "hello", "world")
    assert vs.add_texts.called
    assert turn_id


def test_add_turn_failure():
    vs = MagicMock()
    vs.add_texts.side_effect = RuntimeError("db down")
    store = make_store(vector_store=vs)
    with pytest.raises(RuntimeError):
        store.add_turn("s", "u", "hello", "world")


def test_get_relevant_context_success(monkeypatch):
    store = make_store()
    monkeypatch.setattr(store, "_get_recent_turns", lambda s, u: [{"turn_id": "1"}])
    monkeypatch.setattr(store, "_get_semantic_matches", lambda *a, **k: [{"turn_id": "2"}])
    recent, relevant = store.get_relevant_context("msg", "s", "u")
    assert recent == [{"turn_id": "1"}]
    assert relevant == [{"turn_id": "2"}]


def test_get_relevant_context_exception(monkeypatch):
    store = make_store()
    monkeypatch.setattr(store, "_get_recent_turns", MagicMock(side_effect=RuntimeError("x")))
    assert store.get_relevant_context("msg", "s", "u") == ([], [])


def test_get_recent_turns_with_query():
    vs = MagicMock()
    r1 = MagicMock()
    r1.metadata = {"turn_id": "1"}
    vs.query.return_value = [r1]
    store = make_store(vector_store=vs)
    turns = store._get_recent_turns("s", "u")
    assert turns == [{"turn_id": "1"}]


def test_get_recent_turns_no_query_attr():
    class NoQuery:
        pass

    store = make_store(vector_store=NoQuery())
    assert store._get_recent_turns("s", "u") == []


def test_get_recent_turns_exception():
    vs = MagicMock()
    vs.query.side_effect = RuntimeError("x")
    store = make_store(vector_store=vs)
    assert store._get_recent_turns("s", "u") == []


def test_get_semantic_matches_tuple_results():
    vs = MagicMock()
    doc = MagicMock()
    doc.metadata = {"turn_id": "2"}
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]
    store = make_store(vector_store=vs, config={"settings": {"similarity_threshold": 0.0}})
    matches = store._get_semantic_matches("q", "s", "u")
    assert matches and matches[0]["turn_id"] == "2"
    assert "similarity_score" in matches[0]


def test_get_semantic_matches_excludes_ids():
    vs = MagicMock()
    doc = MagicMock()
    doc.metadata = {"turn_id": "2"}
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]
    store = make_store(vector_store=vs, config={"settings": {"similarity_threshold": 0.0}})
    assert store._get_semantic_matches("q", "s", "u", exclude_ids=["2"]) == []


def test_get_semantic_matches_non_tuple():
    vs = MagicMock()
    doc = MagicMock()
    doc.metadata = {"turn_id": "3"}
    doc.score = 0.5
    vs.similarity_search_with_score.return_value = [doc]
    store = make_store(vector_store=vs, config={"settings": {"similarity_threshold": 0.0}})
    matches = store._get_semantic_matches("q", "s", "u")
    assert matches[0]["turn_id"] == "3"


def test_get_semantic_matches_no_attr():
    class NoSearch:
        pass

    store = make_store(vector_store=NoSearch())
    assert store._get_semantic_matches("q", "s", "u") == []


def test_get_semantic_matches_exception():
    vs = MagicMock()
    vs.similarity_search_with_score.side_effect = RuntimeError("x")
    store = make_store(vector_store=vs)
    assert store._get_semantic_matches("q", "s", "u") == []


def test_extract_text_variants():
    store = make_store()
    assert store._extract_text_from_response("plain") == "plain"
    assert store._extract_text_from_response('{"text": "hi"}') == "hi"
    assert store._extract_text_from_response("{not json") == "{not json"
    assert store._extract_text_from_response(
        {"content": [{"text": "a"}, "b"]}
    ) == "a\nb"
    assert store._extract_text_from_response({"content": "direct"}) == "direct"
    assert store._extract_text_from_response({"message": "m"}) == "m"
    assert store._extract_text_from_response(123) == "123"


def test_normalize_score():
    store = make_store()
    assert store._normalize_score(0.0, "cosine") == 1.0
    assert store._normalize_score(0.0, "dot") == 0.5
    assert store._normalize_score(0.0, "euclidean") == 1.0
    assert store._normalize_score(5.0, "weird") == 5.0


def test_detect_distance_type():
    store = make_store()
    assert store._detect_distance_type([]) == "cosine"
    assert store._detect_distance_type([0.5, 1.5]) == "cosine"
    assert store._detect_distance_type([-1, 0.5]) == "dot"
    assert store._detect_distance_type([100, 150]) == "euclidean"
    assert store._detect_distance_type([5, 8]) == "cosine"


def test_format_context_for_prompt():
    store = make_store()
    recent = [{"user_message": "hi", "agent_response": "hello"}]
    relevant = [{
        "user_message": "q", "agent_response": "a",
        "similarity_score": 0.95, "timestamp": "2024",
    }]
    out = store.format_context_for_prompt(recent, relevant)
    assert "Recent Conversation" in out
    assert "Relevant Context" in out
    assert "0.95" in out
    assert store.format_context_for_prompt([], []) == ""
