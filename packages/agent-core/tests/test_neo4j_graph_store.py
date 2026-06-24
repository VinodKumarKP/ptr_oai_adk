"""Unit tests for the Neo4j knowledge-graph store (no live DB required).

The neo4j driver / langchain-neo4j are imported lazily inside ``_connect`` and
the text2cypher path, so these tests patch ``_connect`` / ``_read`` and never
touch a real database or those optional packages.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from oai_agent_core.components.vector_store.vector_store_factory import VectorStoreFactory
from oai_agent_core.components.vector_store.neo4j_graph_store import (
    Neo4jGraphStore,
    GraphReadOnlyError,
    UnsafeCypherError,
)


class FakeNode(dict):
    """Minimal stand-in for a neo4j.graph.Node (dict-like props + labels/id)."""

    def __init__(self, labels, props, element_id="e1"):
        super().__init__(props)
        self.labels = labels
        self.element_id = element_id


class FakeRel:
    def __init__(self, rel_type, start, end, element_id="r1"):
        self.type = rel_type
        self.start_node = start
        self.end_node = end
        self.element_id = element_id


class FakePath:
    def __init__(self, relationships):
        self.relationships = relationships


def _make_store(**overrides):
    settings = dict(url="bolt://localhost:7687", username="neo4j", password="pw")
    settings.update(overrides)
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        return Neo4jGraphStore(**settings)


# --- Factory wiring -------------------------------------------------------

@pytest.mark.parametrize("type_name", ["neo4j_graph", "neo4j_kg", "NEO4J_GRAPH"])
def test_factory_dispatches_to_graph_store(type_name):
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        store = VectorStoreFactory.create_vector_store(
            type_name, embedding_function=None, url="bolt://x", username="u", password="p"
        )
    assert isinstance(store, Neo4jGraphStore)


def test_requires_url():
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        with pytest.raises(ValueError, match="url is required"):
            Neo4jGraphStore(username="u", password="p")


# --- Settings / env resolution -------------------------------------------

def test_password_env_expansion(monkeypatch):
    monkeypatch.setenv("NEO4J_PW", "s3cr3t")
    store = _make_store(password="${NEO4J_PW}")
    assert store._password == "s3cr3t"


def test_defaults():
    store = _make_store()
    assert store._mode == "traversal"
    assert store._entry_strategy == "fulltext"
    assert store._database == "neo4j"
    assert store._max_hops == 2
    assert store.returns_normalized_scores is True


# --- Cypher safety guard --------------------------------------------------

@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DELETE n",
        "CREATE (n:Person {name:'x'})",
        "MATCH (n) SET n.flag = true",
        "MERGE (n:Account {id:1})",
        "MATCH (n) DETACH DELETE n",
        "DROP INDEX foo",
        "LOAD CSV FROM 'file:///x.csv' AS row CREATE (:Row)",
    ],
)
def test_write_clauses_rejected(cypher):
    store = _make_store()
    with pytest.raises(UnsafeCypherError):
        store._assert_read_only(cypher)


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) RETURN n LIMIT 10",
        "CALL db.index.fulltext.queryNodes($i, $q) YIELD node, score RETURN node",
        "MATCH (n)-[r]-(m) RETURN type(r)",
    ],
)
def test_read_queries_pass(cypher):
    store = _make_store()
    store._assert_read_only(cypher)  # should not raise


# --- Read-only write seam -------------------------------------------------

def test_add_documents_refused():
    store = _make_store()
    with pytest.raises(GraphReadOnlyError):
        store.add_documents([Document(page_content="x")])


def test_add_texts_refused():
    store = _make_store()
    with pytest.raises(GraphReadOnlyError):
        store.add_texts(["x"])


def test_delete_and_reset_are_noops():
    store = _make_store()
    assert store.delete(["1"]) is None
    store.reset_collection()  # no raise
    assert store.reinitialize_database() is False


# --- Display helpers ------------------------------------------------------

def test_node_display_prefers_text_props():
    store = _make_store()
    node = FakeNode(["Policy"], {"name": "Home Shield"})
    assert store._node_label(node) == "Policy"
    assert store._node_display(node) == "Home Shield"


def test_node_display_falls_back_to_label():
    store = _make_store()
    node = FakeNode(["Coverage"], {"premium": 100})
    assert store._node_display(node) == "Coverage"


# --- Retrieval seam (scoring + serialisation) -----------------------------

def test_similarity_search_with_score_normalises_and_serialises():
    store = _make_store()
    n1 = FakeNode(["Policy"], {"name": "Policy A"}, element_id="p1")
    n2 = FakeNode(["Policy"], {"name": "Policy B"}, element_id="p2")
    cov = FakeNode(["Coverage"], {"name": "Fire"}, element_id="c1")

    with patch.object(store, "_entry_nodes", return_value=[(n1, 4.0), (n2, 2.0)]), \
         patch.object(store, "_read", return_value=[{"p": FakePath([FakeRel("HAS_COVERAGE", n1, cov)])}]):
        results = store.similarity_search_with_score("fire coverage", k=2)

    assert len(results) == 2
    # Scores normalised against the max raw score (4.0) → 1.0 and 0.5
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.5)
    # Sub-graph serialised into page_content with the relationship rendered
    assert "Policy A" in results[0][0].page_content
    assert "-[HAS_COVERAGE]->" in results[0][0].page_content
    assert results[0][0].metadata["entry_node"] == "Policy A"


def test_similarity_search_empty_when_no_entry_nodes():
    store = _make_store()
    with patch.object(store, "_entry_nodes", return_value=[]):
        assert store.similarity_search_with_score("nothing", k=3) == []


# --- entry strategies: fulltext vs entity_linking -------------------------

def test_entry_nodes_fulltext_uses_raw_query_as_single_term():
    """Default strategy looks the whole query up as one term (no LLM)."""
    store = _make_store(entry_strategy="fulltext")
    fire = FakeNode(["Coverage"], {"name": "Fire Damage"}, element_id="c1")

    with patch.object(store, "_extract_entities") as mock_extract, \
         patch.object(store, "_lookup_nodes", return_value=[(fire, 2.0)]) as mock_lookup:
        hits = store._entry_nodes("which customers filed fire claims?", k=5)

    mock_extract.assert_not_called()                     # no entity extraction
    assert mock_lookup.call_count == 1                   # single term = whole query
    assert mock_lookup.call_args[0][0] == "which customers filed fire claims?"
    assert [n.element_id for n, _ in hits] == ["c1"]


def test_entry_nodes_entity_linking_looks_up_each_entity_and_dedups():
    """entity_linking extracts entities, looks each up, and de-duplicates nodes."""
    store = _make_store(entry_strategy="entity_linking")
    store._llm = MagicMock()
    fire = FakeNode(["Coverage"], {"name": "Fire Damage"}, element_id="c1")
    claim = FakeNode(["Claim"], {"name": "Kitchen fire claim"}, element_id="clm1")

    def lookup(term, k):
        if term == "fire":
            return [(fire, 3.0)]
        if term == "claim":
            # 'claim' also surfaces the fire coverage again → must be de-duped
            return [(claim, 2.0), (fire, 1.0)]
        return []

    with patch.object(store, "_extract_entities", return_value=["fire", "claim"]) as mock_extract, \
         patch.object(store, "_lookup_nodes", side_effect=lookup) as mock_lookup:
        hits = store._entry_nodes("which customers filed fire claims?", k=5)

    mock_extract.assert_called_once()
    assert mock_lookup.call_count == 2                   # one lookup per entity
    ids = [n.element_id for n, _ in hits]
    assert ids == ["c1", "clm1"]                         # de-duped, sorted by score desc


def test_entity_linking_falls_back_to_query_when_no_entities():
    """If the LLM extracts nothing, fall back to the raw query as one term."""
    store = _make_store(entry_strategy="entity_linking")
    store._llm = MagicMock()
    node = FakeNode(["Policy"], {"name": "Home Shield 360"}, element_id="p1")

    with patch.object(store, "_extract_entities", return_value=[]), \
         patch.object(store, "_lookup_nodes", return_value=[(node, 1.0)]) as mock_lookup:
        hits = store._entry_nodes("tell me about home shield", k=5)

    assert mock_lookup.call_args[0][0] == "tell me about home shield"
    assert [n.element_id for n, _ in hits] == ["p1"]


def test_extract_entities_parses_llm_csv():
    store = _make_store(entry_strategy="entity_linking")
    store._llm = MagicMock()
    with patch.object(store, "_invoke_llm", return_value=" fire , claim , Acme Corp "):
        assert store._extract_entities("which customers filed fire claims?") == [
            "fire", "claim", "Acme Corp",
        ]


def test_extract_entities_returns_empty_without_llm():
    store = _make_store(entry_strategy="entity_linking")
    store._llm = None
    assert store._extract_entities("anything") == []


# --- vector entry (hybrid, node-level) ------------------------------------

def test_vector_entry_requires_config():
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        with pytest.raises(ValueError, match="requires a 'vector_entry'"):
            Neo4jGraphStore(
                url="bolt://x", username="u", password="p", entry_strategy="vector"
            )


def test_vector_entry_resolves_ids_to_nodes_and_traverses():
    vector_entry = {"type": "postgres", "settings": {"collection_name": "nodes"}}
    store = _make_store(entry_strategy="vector", vector_entry=vector_entry)

    cov = FakeNode(["Coverage"], {"id": "COV-02", "name": "Flood Damage"}, element_id="c1")
    pol = FakeNode(["Policy"], {"id": "P-1001", "name": "Home Shield 360"}, element_id="p1")

    # Fake node-level vector index: returns Documents whose metadata carries node_id
    fake_index = MagicMock()
    fake_index.similarity_search_with_score.return_value = [
        (Document(page_content="Flood Damage ...", metadata={"node_id": "COV-02"}), 0.12),
        (Document(page_content="Home Shield ...", metadata={"node_id": "P-1001"}), 0.30),
    ]

    def fake_read(cypher, params=None):
        # node fetch by id
        if "n[$prop]" in cypher:
            return [{"n": cov}] if params["id"] == "COV-02" else [{"n": pol}]
        # neighbourhood traversal
        return [{"p": FakePath([FakeRel("HAS_COVERAGE", pol, cov)])}]

    with patch.object(store, "_get_vector_entry_store", return_value=fake_index), \
         patch.object(store, "_read", side_effect=fake_read):
        results = store.similarity_search_with_score("what if my basement floods?", k=5)

    fake_index.similarity_search_with_score.assert_called_once()
    # Two entry nodes resolved; rank-based scores → top normalised to 1.0
    assert len(results) == 2
    assert results[0][1] == pytest.approx(1.0)
    assert "Flood Damage" in results[0][0].page_content


def test_vector_entry_skips_missing_id_and_unresolved_nodes(caplog):
    import logging

    vector_entry = {"type": "chroma", "settings": {}}
    store = _make_store(entry_strategy="vector", vector_entry=vector_entry)

    fake_index = MagicMock()
    fake_index.similarity_search_with_score.return_value = [
        (Document(page_content="no id here", metadata={}), 0.1),          # missing id
        (Document(page_content="ghost", metadata={"node_id": "X-999"}), 0.2),  # not in graph
    ]
    with patch.object(store, "_get_vector_entry_store", return_value=fake_index), \
         patch.object(store, "_read", return_value=[]), \
         caplog.at_level(logging.WARNING):
        nodes = store._vector_entry_nodes("q", k=5)

    assert nodes == []
    assert any("missing 'node_id'" in r.message for r in caplog.records)
    assert any("not found in graph" in r.message for r in caplog.records)


def test_fetch_node_by_id_uses_parameterised_property():
    vector_entry = {"type": "postgres", "settings": {}}
    store = _make_store(entry_strategy="vector", vector_entry=vector_entry,
                        entry_id_property="businessId")
    node = FakeNode(["Policy"], {"businessId": "P-1"}, element_id="p1")
    captured = {}

    def fake_read(cypher, params=None):
        captured["cypher"] = cypher
        captured["params"] = params
        return [{"n": node}]

    with patch.object(store, "_read", side_effect=fake_read):
        got = store._fetch_node_by_id("P-1")

    assert got is node
    assert "n[$prop]" in captured["cypher"]           # parameterised, no interpolation
    assert captured["params"] == {"prop": "businessId", "id": "P-1"}


def test_text2cypher_requires_llm():
    store = _make_store(retrieval_mode="text2cypher")  # llm not provided
    with pytest.raises(ValueError, match="requires an LLM"):
        store.similarity_search_with_score("how many policies?", k=5)


def test_count_uses_node_count():
    store = _make_store()
    with patch.object(store, "_read", return_value=[{"c": 42}]):
        assert store.count() == 42


# --- LLM model-id resolution ---------------------------------------------

def test_resolve_model_id_from_strands_get_config():
    class FakeStrandsModel:
        def get_config(self):
            return {"model_id": "bedrock/claude-sonnet"}
    assert Neo4jGraphStore._resolve_model_id(FakeStrandsModel()) == "bedrock/claude-sonnet"


def test_resolve_model_id_from_string_and_attrs():
    assert Neo4jGraphStore._resolve_model_id("gpt-4o") == "gpt-4o"

    class M:  # plain attribute style
        model_id = "azure/gpt-4o"
    assert Neo4jGraphStore._resolve_model_id(M()) == "azure/gpt-4o"


def test_invoke_llm_passes_region_to_litellm():
    import sys
    import types

    store = _make_store()
    store._llm = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    store._llm_region = "us-west-2"

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = fake_completion
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        out = store._invoke_llm("hi")

    assert out == "ok"
    assert captured["model"] == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert captured["aws_region_name"] == "us-west-2"


def test_invoke_llm_omits_region_when_unset():
    import sys
    import types

    store = _make_store()
    store._llm = "gpt-4o"
    store._llm_region = None

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = fake_completion
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        store._invoke_llm("hi")

    assert "aws_region_name" not in captured


# --- text2cypher (LLM → Cypher → guarded read → LLM answer) ---------------

class FakeRecord(dict):
    def data(self):
        return dict(self)


def test_text2cypher_generates_contains_query_and_answers():
    store = _make_store(retrieval_mode="text2cypher")
    store._llm = MagicMock()

    generated = ("MATCH (p:Policy)-[:HAS_COVERAGE]->(c) "
                 "WHERE toLower(c.name) CONTAINS toLower('flood') RETURN c.name")
    with patch.object(store, "get_schema", return_value="(:Policy)-[:HAS_COVERAGE]->(:Coverage)"), \
         patch.object(store, "_invoke_llm", side_effect=[generated, "Flood damage is covered."]) as mock_llm, \
         patch.object(store, "_read", return_value=[FakeRecord({"c.name": "Flood Damage"})]) as mock_read:
        results = store.similarity_search_with_score("is flood covered?", k=5)

    # Cypher generation prompt then QA prompt → two LLM calls
    assert mock_llm.call_count == 2
    mock_read.assert_called_once()
    doc, score = results[0]
    assert score == 1.0
    assert doc.page_content == "Flood damage is covered."
    assert "CONTAINS" in doc.metadata["cypher"]
    assert doc.metadata["result_rows"] == 1


def test_text2cypher_strips_code_fences():
    store = _make_store(retrieval_mode="text2cypher")
    store._llm = MagicMock()
    fenced = "```cypher\nMATCH (n:Policy) RETURN n.name\n```"
    with patch.object(store, "get_schema", return_value="schema"), \
         patch.object(store, "_invoke_llm", side_effect=[fenced, "answer"]), \
         patch.object(store, "_read", return_value=[FakeRecord({"n.name": "Home Shield 360"})]):
        results = store.similarity_search_with_score("list policies", k=5)
    assert results[0][0].metadata["cypher"] == "MATCH (n:Policy) RETURN n.name"


def test_text2cypher_warns_on_empty_result(caplog):
    import logging

    store = _make_store(retrieval_mode="text2cypher")
    store._llm = MagicMock()
    with patch.object(store, "get_schema", return_value="schema"), \
         patch.object(store, "_invoke_llm", side_effect=["MATCH (n:Policy) RETURN n", "Could not find it."]), \
         patch.object(store, "_read", return_value=[]):
        with caplog.at_level(logging.WARNING):
            results = store.similarity_search_with_score("is flood covered?", k=5)

    assert results[0][0].metadata["result_rows"] == 0
    assert any("returned 0 rows" in r.message for r in caplog.records)


def test_text2cypher_refuses_generated_write_without_crashing(caplog):
    import logging

    store = _make_store(retrieval_mode="text2cypher")
    store._llm = MagicMock()
    with patch.object(store, "get_schema", return_value="schema"), \
         patch.object(store, "_invoke_llm", side_effect=["MATCH (n) DETACH DELETE n", "I couldn't retrieve that."]), \
         patch.object(store, "_read") as mock_read:
        with caplog.at_level(logging.WARNING):
            results = store.similarity_search_with_score("delete everything", k=5)

    # The write query is never executed, and the tool returns safely (no raise)
    mock_read.assert_not_called()
    assert results[0][0].metadata["result_rows"] == 0
    assert any("Refused non-read-only" in r.message for r in caplog.records)
