"""Additional coverage for Neo4jGraphStore (neo4j driver mocked)."""
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

import neo4j
import oai_agent_core.components.vector_store.neo4j_graph_store as ngs
from oai_agent_core.components.vector_store.neo4j_graph_store import (
    Neo4jGraphStore,
    GraphReadOnlyError,
    UnsafeCypherError,
)


class FakeNode:
    def __init__(self, element_id, props=None, labels=None):
        self.element_id = element_id
        self._props = props or {}
        self.labels = labels if labels is not None else ["Entity"]

    def __contains__(self, key):
        return key in self._props

    def __getitem__(self, key):
        return self._props[key]


def make_store(monkeypatch, driver=None, **kwargs):
    driver = driver or MagicMock()
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: driver)
    settings = {"url": "bolt://localhost:7687", "username": "u", "password": "p"}
    settings.update(kwargs)
    store = Neo4jGraphStore(**settings)
    return store, driver


def test_requires_url(monkeypatch):
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: MagicMock())
    with pytest.raises(ValueError):
        Neo4jGraphStore(username="u")


def test_vector_strategy_requires_config(monkeypatch):
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: MagicMock())
    with pytest.raises(ValueError):
        Neo4jGraphStore(url="bolt://x", entry_strategy="vector")


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("MY_NEO4J_PW", "secret")
    store, _ = make_store(monkeypatch, password="${MY_NEO4J_PW}")
    assert store._password == "secret"


def test_assert_read_only():
    store, _ = make_store_simple()
    store._assert_read_only("MATCH (n) RETURN n")  # ok
    with pytest.raises(UnsafeCypherError):
        store._assert_read_only("CREATE (n:Node) RETURN n")


@pytest.fixture
def simple_store(monkeypatch):
    store, driver = make_store(monkeypatch)
    return store, driver


def make_store_simple():
    mp = pytest.MonkeyPatch()
    return make_store(mp)


def _fake_driver_with_rows(rows, raise_on_run=False):
    driver = MagicMock()
    tx = MagicMock()
    if raise_on_run:
        tx.run.side_effect = RuntimeError("query fail")
    else:
        tx.run.return_value = rows

    session = MagicMock()
    session.begin_transaction.return_value = tx

    @contextmanager
    def session_cm(**kwargs):
        yield session

    driver.session.side_effect = lambda **k: session_cm(**k)
    return driver, tx


def test_read_success(monkeypatch):
    rows = [{"c": 5}]
    driver, tx = _fake_driver_with_rows(rows)
    store, _ = make_store(monkeypatch, driver=driver)
    result = store._read("MATCH (n) RETURN count(n) AS c")
    assert result == rows
    assert tx.commit.called


def test_read_rolls_back_on_error(monkeypatch):
    driver, tx = _fake_driver_with_rows(None, raise_on_run=True)
    store, _ = make_store(monkeypatch, driver=driver)
    with pytest.raises(RuntimeError):
        store._read("MATCH (n) RETURN n")
    assert tx.rollback.called


def test_node_label_and_display(simple_store):
    store, _ = simple_store
    node = FakeNode("e1", props={"name": "Alice"}, labels=["Person"])
    assert store._node_label(node) == "Person"
    assert store._node_display(node) == "Alice"
    empty = FakeNode("e2", props={}, labels=[])
    assert store._node_label(empty) == "Node"
    assert store._node_display(empty) == "Node"


def test_entry_nodes_entity_linking(monkeypatch):
    store, _ = make_store(monkeypatch, entry_strategy="entity_linking")
    node = FakeNode("e1", {"name": "X"})
    monkeypatch.setattr(store, "_extract_entities", lambda q: ["term1"])
    monkeypatch.setattr(store, "_lookup_nodes", lambda term, k: [(node, 0.9)])
    hits = store._entry_nodes("query", 4)
    assert hits[0][0] is node


def test_entry_nodes_default_term(monkeypatch):
    store, _ = make_store(monkeypatch)
    node = FakeNode("e1", {"name": "X"})
    monkeypatch.setattr(store, "_lookup_nodes", lambda term, k: [(node, 1.0)])
    hits = store._entry_nodes("query", 4)
    assert len(hits) == 1


def test_lookup_nodes_fulltext_success(monkeypatch):
    store, _ = make_store(monkeypatch, fulltext_index="myindex")
    node = FakeNode("e1")
    monkeypatch.setattr(store, "_read", lambda c, p=None: [{"node": node, "score": 2.0}])
    res = store._lookup_nodes("term", 5)
    assert res[0][1] == 2.0


def test_lookup_nodes_fulltext_fallback(monkeypatch):
    store, _ = make_store(monkeypatch, fulltext_index="myindex")
    node = FakeNode("e1")
    calls = {"n": 0}

    def fake_read(cypher, params=None):
        calls["n"] += 1
        if "fulltext" in cypher:
            raise RuntimeError("index missing")
        return [{"node": node}]

    monkeypatch.setattr(store, "_read", fake_read)
    res = store._lookup_nodes("term", 5)
    assert res[0][1] == 1.0


def test_lookup_nodes_no_index(monkeypatch):
    store, _ = make_store(monkeypatch)
    node = FakeNode("e1")
    monkeypatch.setattr(store, "_read", lambda c, p=None: [{"node": node}])
    res = store._lookup_nodes("term", 5)
    assert res[0][1] == 1.0


def test_extract_entities_no_llm(monkeypatch):
    store, _ = make_store(monkeypatch)
    assert store._extract_entities("q") == []


def test_extract_entities_with_llm(monkeypatch):
    store, _ = make_store(monkeypatch, llm="model")
    monkeypatch.setattr(store, "_invoke_llm", lambda p: "Alice, Bob")
    assert store._extract_entities("q") == ["Alice", "Bob"]


def test_extract_entities_error(monkeypatch):
    store, _ = make_store(monkeypatch, llm="model")
    monkeypatch.setattr(store, "_invoke_llm", MagicMock(side_effect=RuntimeError("x")))
    assert store._extract_entities("q") == []


def test_resolve_model_id_variants():
    assert Neo4jGraphStore._resolve_model_id("bedrock/claude") == "bedrock/claude"

    class WithGetConfig:
        def get_config(self):
            return {"model_id": "m1"}

    assert Neo4jGraphStore._resolve_model_id(WithGetConfig()) == "m1"

    class WithConfigDict:
        config = {"model": "m2"}

    assert Neo4jGraphStore._resolve_model_id(WithConfigDict()) == "m2"

    class WithAttr:
        model_id = "m3"
        config = None

    assert Neo4jGraphStore._resolve_model_id(WithAttr()) == "m3"


def test_invoke_llm_litellm(monkeypatch):
    store, _ = make_store(monkeypatch, llm="bedrock/claude", llm_region="us-east-1")
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = lambda **k: {"choices": [{"message": {"content": "answer"}}]}
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    assert store._invoke_llm("prompt") == "answer"


def test_invoke_llm_langchain(monkeypatch):
    resp = MagicMock()
    resp.content = "lc answer"
    llm = MagicMock(spec=["invoke"])
    llm.invoke.return_value = resp
    store, _ = make_store(monkeypatch, llm=llm)
    # _resolve_model_id returns None for MagicMock(spec=['invoke'])
    assert store._invoke_llm("prompt") == "lc answer"


def test_invoke_llm_no_interface(monkeypatch):
    store, _ = make_store(monkeypatch)

    class Bare:
        pass

    store._llm = Bare()
    with pytest.raises(ValueError):
        store._invoke_llm("prompt")


def test_neighbourhood_text(monkeypatch):
    store, _ = make_store(monkeypatch)
    entry = FakeNode("e1", {"name": "Root"}, ["Root"])
    start = FakeNode("s1", {"name": "A"}, ["A"])
    end = FakeNode("d1", {"name": "B"}, ["B"])
    rel = MagicMock()
    rel.element_id = "r1"
    rel.start_node = start
    rel.end_node = end
    rel.type = "KNOWS"
    path = MagicMock()
    path.relationships = [rel, rel]  # duplicate filtered
    monkeypatch.setattr(store, "_read", lambda c, p=None: [{"p": path}])
    text, meta = store._neighbourhood_text(entry)
    assert "Root" in text
    assert "KNOWS" in text
    assert meta["relationships"] == 1


def test_clean_cypher():
    assert Neo4jGraphStore._clean_cypher("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"
    assert Neo4jGraphStore._clean_cypher("cypher query: MATCH (n) RETURN n") == "MATCH (n) RETURN n"


def test_rows_to_text():
    store, _ = make_store_simple()
    r1 = MagicMock()
    r1.data.return_value = {"a": 1, "b": 2}
    out = store._rows_to_text([r1], limit=5)
    assert "a: 1" in out


def test_text2cypher_no_llm(monkeypatch):
    store, _ = make_store(monkeypatch, retrieval_mode="text2cypher")
    with pytest.raises(ValueError):
        store._text2cypher("q", 4)


def test_text2cypher_success(monkeypatch):
    store, _ = make_store(monkeypatch, retrieval_mode="text2cypher", llm="model")
    monkeypatch.setattr(store, "get_schema", lambda: "SCHEMA")
    prompts = []

    def fake_invoke(prompt):
        prompts.append(prompt)
        if "Cypher query:" in prompt:
            return "MATCH (n) RETURN n"
        return "final answer"

    monkeypatch.setattr(store, "_invoke_llm", fake_invoke)
    r = MagicMock()
    r.data.return_value = {"n": "x"}
    monkeypatch.setattr(store, "_read", lambda c, p=None: [r])
    docs = store._text2cypher("question", 4)
    assert docs[0][0].page_content == "final answer"
    assert docs[0][0].metadata["result_rows"] == 1


def test_text2cypher_unsafe_generated(monkeypatch):
    store, _ = make_store(monkeypatch, retrieval_mode="text2cypher", llm="model")
    monkeypatch.setattr(store, "get_schema", lambda: "SCHEMA")
    monkeypatch.setattr(store, "_invoke_llm", lambda p: "CREATE (n) RETURN n" if "Cypher query:" in p else "ans")
    docs = store._text2cypher("q", 4)
    assert docs[0][0].metadata["result_rows"] == 0


def test_text2cypher_read_error(monkeypatch):
    store, _ = make_store(monkeypatch, retrieval_mode="text2cypher", llm="model")
    monkeypatch.setattr(store, "get_schema", lambda: "SCHEMA")
    monkeypatch.setattr(store, "_invoke_llm", lambda p: "MATCH (n) RETURN n" if "Cypher query:" in p else "ans")
    monkeypatch.setattr(store, "_read", MagicMock(side_effect=RuntimeError("boom")))
    docs = store._text2cypher("q", 4)
    assert docs[0][0].metadata["result_rows"] == 0


def test_similarity_search_with_score_traversal(monkeypatch):
    store, _ = make_store(monkeypatch)
    node = FakeNode("e1", {"name": "X"})
    monkeypatch.setattr(store, "_entry_nodes", lambda q, k: [(node, 2.0)])
    monkeypatch.setattr(store, "_neighbourhood_text", lambda n: ("text", {"source": "s"}))
    res = store.similarity_search_with_score("q", k=4)
    assert res[0][1] == 1.0


def test_similarity_search_with_score_empty(monkeypatch):
    store, _ = make_store(monkeypatch)
    monkeypatch.setattr(store, "_entry_nodes", lambda q, k: [])
    assert store.similarity_search_with_score("q") == []


def test_similarity_search_text2cypher_dispatch(monkeypatch):
    store, _ = make_store(monkeypatch, retrieval_mode="text2cypher", llm="m")
    monkeypatch.setattr(store, "_text2cypher", lambda q, k: [("doc", 1.0)])
    assert store.similarity_search_with_score("q") == [("doc", 1.0)]


def test_similarity_search_and_query(monkeypatch):
    store, _ = make_store(monkeypatch)
    from langchain_core.documents import Document
    monkeypatch.setattr(
        store, "similarity_search_with_score",
        lambda q, k=4, **kw: [(Document(page_content="d", metadata={}), 1.0)],
    )
    assert len(store.similarity_search("q")) == 1
    assert len(store.query(query_text="q")) == 1
    assert store.query() == []


def test_count(monkeypatch):
    store, _ = make_store(monkeypatch)
    monkeypatch.setattr(store, "_read", lambda c, p=None: [{"c": 42}])
    assert store.count() == 42


def test_count_error(monkeypatch):
    store, _ = make_store(monkeypatch)
    monkeypatch.setattr(store, "_read", MagicMock(side_effect=RuntimeError("x")))
    assert store.count() == 0


def test_graph_stats(monkeypatch):
    store, _ = make_store(monkeypatch, fulltext_index="idx")
    node = FakeNode("e1", {"name": "X"}, ["Person"])

    def fake_read(cypher, params=None):
        if "count(n)" in cypher:
            return [{"c": 10}]
        if "db.labels" in cypher:
            return [{"label": "Person"}]
        if "relationshipTypes" in cypher:
            return [{"relationshipType": "KNOWS"}]
        if "apoc.meta.stats" in cypher:
            return [{"labels": {"Person": 10}, "relCount": 5}]
        if "FULLTEXT INDEXES" in cypher:
            return [{"name": "idx"}]
        if "LIMIT" in cypher:
            return [{"n": node}]
        return []

    monkeypatch.setattr(store, "_read", fake_read)
    stats = store.graph_stats()
    assert stats["node_count"] == 10
    assert stats["labels"] == ["Person"]
    assert stats["relationship_types"] == ["KNOWS"]
    assert stats["relationship_count"] == 5
    assert stats["fulltext_index_present"] is True
    assert stats["sample_nodes"][0]["display"] == "X"


def test_graph_stats_errors(monkeypatch):
    store, _ = make_store(monkeypatch)
    monkeypatch.setattr(store, "_read", MagicMock(side_effect=RuntimeError("x")))
    stats = store.graph_stats()
    assert stats["node_count"] == 0


def test_write_seam_refused(monkeypatch):
    store, _ = make_store(monkeypatch)
    with pytest.raises(GraphReadOnlyError):
        store.add_documents([])
    with pytest.raises(GraphReadOnlyError):
        store.add_texts(["x"])
    with pytest.raises(GraphReadOnlyError):
        Neo4jGraphStore.from_texts(["x"])
    assert store.delete(["a"]) is None
    store.reset_collection()
    assert store.reinitialize_database() is False


def test_close(monkeypatch):
    store, driver = make_store(monkeypatch)
    store.close()
    assert driver.close.called


def test_get_schema(monkeypatch):
    store, _ = make_store(monkeypatch)
    fake_graph = MagicMock()
    fake_graph.schema = "SCHEMA"
    monkeypatch.setattr(store, "_graph", lambda: fake_graph)
    assert store.get_schema() == "SCHEMA"
    assert store.get_schema(refresh=True) == "SCHEMA"
    assert fake_graph.refresh_schema.called


def test_vector_entry_nodes(monkeypatch):
    store, _ = make_store(
        monkeypatch, entry_strategy="vector",
        vector_entry={"type": "postgres", "settings": {}},
    )
    from langchain_core.documents import Document
    fake_vstore = MagicMock()
    fake_vstore.similarity_search_with_score.return_value = [
        (Document(page_content="a", metadata={"node_id": "n1"}), 0.9),
        (Document(page_content="b", metadata={}), 0.8),  # missing id -> skipped
    ]
    monkeypatch.setattr(store, "_get_vector_entry_store", lambda: fake_vstore)
    node = FakeNode("e1", {"name": "X"})
    monkeypatch.setattr(store, "_fetch_node_by_id", lambda nid: node if nid == "n1" else None)
    res = store._entry_nodes("q", 4)
    assert len(res) == 1
    assert res[0][0] is node
