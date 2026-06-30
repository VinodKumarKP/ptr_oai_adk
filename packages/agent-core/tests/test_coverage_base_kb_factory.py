"""Additional coverage for BaseKnowledgeBaseFactory."""
import sys
import types
from unittest.mock import MagicMock

import pytest

import oai_agent_core.core.base_knowledge_base_factory as kbf
from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory
from langchain_core.documents import Document


class ConcreteKB(BaseKnowledgeBaseFactory):
    def create_tool(self, name, description):
        return f"tool:{name}"

    def create_load_tool(self, name, description):
        return f"loadtool:{name}"


def make(config=None, **kwargs):
    return ConcreteKB(knowledge_base_config=config or [], logger=MagicMock(), **kwargs)


# ---- initialization / normalization ----

def test_initialize_dict_style(monkeypatch):
    created = []
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_registry_proxy_knowledge_base",
                        lambda self, cfg: created.append(cfg))
    config = {
        "registry": {"url": "http://reg", "token": "tok"},
        "sources": [{"name": "kb1"}],
    }
    make(config)
    assert created[0]["registry_url"] == "http://reg"
    assert created[0]["auth_token"] == "tok"


def test_initialize_list_dispatch(monkeypatch):
    proxy = []
    single = []
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_registry_proxy_knowledge_base",
                        lambda self, cfg: proxy.append(cfg))
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_single_knowledge_base",
                        lambda self, cfg: single.append(cfg))
    make([{"name": "a", "registry_url": "http://r"}, {"name": "b"}])
    assert len(proxy) == 1 and len(single) == 1


# ---- _process_data_sources ----

def test_process_data_sources_all_types():
    fac = make()
    fac.project_root = "/root"
    sources = [
        {"type": "dynamic", "loader": "WebLoader", "settings": {}, "ttl_seconds": 10},
        {"loader": "my.module.Loader"},  # inferred dynamic, dotted
        {"type": "file", "path": "docs/a.txt"},
        {"type": "file"},  # missing path -> skip
        {"type": "s3", "bucket": "b", "key": "k"},
        {"type": "s3"},  # missing bucket -> skip
        {"type": "weird"},  # unknown -> skip
    ]
    result = fac._process_data_sources(sources, {"chunk_size": 100}, {"opt": 1})
    assert "__dynamic_0__" in result
    assert result["__dynamic_0__"]["loader_class"] == "langchain_community.document_loaders.WebLoader"
    assert result["__dynamic_1__"]["loader_class"] == "my.module.Loader"
    assert "/root/docs/a.txt" in result
    assert any(k.startswith("s3://b/k") for k in result)


def test_process_data_sources_absolute_file():
    fac = make()
    fac.project_root = "/root"
    result = fac._process_data_sources([{"path": "/abs/file.txt"}], {}, {})
    assert "/abs/file.txt" in result


# ---- registry proxy ----

def test_create_registry_proxy(monkeypatch):
    fake_store = MagicMock()
    monkeypatch.setattr(
        "oai_agent_core.components.vector_store.registry_proxy_vector_store.RegistryProxyVectorStore",
        lambda **k: fake_store,
    )
    fac = make()
    fac._create_registry_proxy_knowledge_base({
        "name": "kb1", "description": "d",
        "registry_url": "http://reg", "auth_token": "tok",
        "retrieval_settings": {"top_k": 3},
    })
    assert "kb1" in fac.knowledge_base_tools
    assert fac.vector_store is fake_store


# ---- single knowledge base ----

def _patch_embeddings(monkeypatch, value=object()):
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_embeddings", lambda self, *a, **k: value)


def test_create_single_kb_with_loader(monkeypatch, tmp_path):
    fake_vs = MagicMock()
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store", lambda *a, **k: fake_vs)
    _patch_embeddings(monkeypatch)
    loader_instance = MagicMock()
    document_loader = MagicMock(return_value=loader_instance)
    fac = make(document_loader=document_loader, project_root=str(tmp_path))
    fac._create_single_knowledge_base({
        "name": "kb1",
        "vector_store": {"type": "chroma", "settings": {"persist_directory": "db", "collection_name": "c"}},
        "data_sources": [{"path": str(tmp_path / "f.txt")}],
        "retrieval_settings": {"score_threshold": 0.5},
    })
    assert "kb1" in fac.knowledge_base_tools
    assert loader_instance.load_db.called


def test_create_single_kb_default_loader(monkeypatch, tmp_path):
    fake_vs = MagicMock()
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store", lambda *a, **k: fake_vs)
    _patch_embeddings(monkeypatch)
    fake_loader_module = types.ModuleType("oai_agent_core.components.loaders.document_loader")
    fake_loader_module.DocumentLoader = MagicMock()
    monkeypatch.setitem(sys.modules, "oai_agent_core.components.loaders.document_loader", fake_loader_module)
    fac = make(project_root=str(tmp_path))
    fac._create_single_knowledge_base({
        "name": "kb1",
        "vector_store": {"type": "chroma", "settings": {}},
    })
    assert "kb1" in fac.knowledge_base_tools


def test_create_single_kb_embeddings_none(monkeypatch):
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_embeddings", lambda self, *a, **k: None)
    fac = make()
    fac._create_single_knowledge_base({"name": "kb1", "vector_store": {"type": "chroma"}})
    assert "kb1" not in fac.knowledge_base_tools


def test_create_single_kb_vector_store_error(monkeypatch):
    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store",
                        MagicMock(side_effect=ValueError("bad")))
    fac = make()
    fac._create_single_knowledge_base({"name": "kb1", "vector_store": {"type": "chroma"}})
    assert "kb1" not in fac.knowledge_base_tools


def test_create_single_kb_vector_store_unexpected_error(monkeypatch):
    _patch_embeddings(monkeypatch)
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store",
                        MagicMock(side_effect=RuntimeError("boom")))
    fac = make()
    fac._create_single_knowledge_base({"name": "kb1", "vector_store": {"type": "chroma"}})
    assert "kb1" not in fac.knowledge_base_tools


def test_create_single_kb_graph_store(monkeypatch):
    fake_vs = MagicMock()
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store", lambda *a, **k: fake_vs)
    fac = make(llm="model")
    fac._create_single_knowledge_base({
        "name": "graph",
        "vector_store": {"type": "neo4j_graph", "settings": {}},
    })
    assert fac.knowledge_base_tools["graph"]["vector_load_type"] == "graph"


def test_create_single_kb_graph_store_needs_embeddings(monkeypatch):
    fake_vs = MagicMock()
    monkeypatch.setattr(kbf.VectorStoreFactory, "create_vector_store", lambda *a, **k: fake_vs)
    _patch_embeddings(monkeypatch)
    fac = make(llm="model")
    fac._create_single_knowledge_base({
        "name": "graph",
        "vector_store": {"type": "neo4j_graph", "settings": {"entry_strategy": "vector"}},
    })
    assert "graph" in fac.knowledge_base_tools


def test_create_single_kb_graph_embeddings_none(monkeypatch):
    monkeypatch.setattr(BaseKnowledgeBaseFactory, "_create_embeddings", lambda self, *a, **k: None)
    fac = make(llm="model")
    fac._create_single_knowledge_base({
        "name": "graph",
        "vector_store": {"type": "neo4j_graph", "settings": {"entry_strategy": "vector"}},
    })
    assert "graph" not in fac.knowledge_base_tools


# ---- search ----

def _make_with_tool(monkeypatch, vector_store, retrieval_settings=None, vector_load_type="custom"):
    fac = make()
    fac.knowledge_base_tools["kb1"] = {
        "vector_store": vector_store,
        "description": "d",
        "retrieval_settings": retrieval_settings or {},
        "vector_load_type": vector_load_type,
    }
    return fac


def test_search_kb_not_found():
    fac = make()
    assert "not found" in fac.search_knowledge_base("q", kb_name="missing")


def test_search_no_kb_available():
    fac = make()
    assert "No knowledge base available" in fac.search_knowledge_base("q")


def test_search_with_results(monkeypatch):
    vs = MagicMock()
    doc = Document(page_content="hello", metadata={"source": "f1"})
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]  # cosine -> high sim
    fac = _make_with_tool(monkeypatch, vs, {"top_k": 3, "score_threshold": 0.0})
    out = fac.search_knowledge_base("q", kb_name="kb1", source_list=["f1"], session_id="s1")
    assert "hello" in out
    assert "Relevance" in out


def test_search_threshold_filters_all(monkeypatch):
    vs = MagicMock()
    doc = Document(page_content="x", metadata={})
    vs.similarity_search_with_score.return_value = [(doc, 1.9)]  # cosine -> low sim
    fac = _make_with_tool(monkeypatch, vs, {"score_threshold": 0.99})
    assert "No relevant information" in fac.search_knowledge_base("q", kb_name="kb1")


def test_search_default_via_vector_store(monkeypatch):
    vs = MagicMock()
    doc = Document(page_content="x", metadata={"source": "s"})
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]
    fac = make()
    fac.vector_store = vs
    fac.knowledge_base_tools["kb1"] = {
        "vector_store": vs, "description": "d",
        "retrieval_settings": {"score_threshold": 0.0}, "vector_load_type": "custom",
    }
    out = fac.search_knowledge_base("q")
    assert "x" in out


def test_search_default_first_tool(monkeypatch):
    vs = MagicMock()
    doc = Document(page_content="y", metadata={})
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]
    fac = make()
    fac.knowledge_base_tools["kb1"] = {
        "vector_store": vs, "description": "d",
        "retrieval_settings": {"score_threshold": 0.0}, "vector_load_type": "custom",
    }
    out = fac.search_knowledge_base("q")
    assert "y" in out


def test_search_prenormalized_scores(monkeypatch):
    vs = MagicMock()
    vs.returns_normalized_scores = True
    doc = Document(page_content="z", metadata={})
    vs.similarity_search_with_score.return_value = [(doc, 0.95)]
    fac = _make_with_tool(monkeypatch, vs, {"score_threshold": 0.5})
    out = fac.search_knowledge_base("q", kb_name="kb1")
    assert "z" in out


class _ScoredDoc:
    def __init__(self, page_content, metadata, score):
        self.page_content = page_content
        self.metadata = metadata
        self.score = score


def test_search_non_tuple_results(monkeypatch):
    vs = MagicMock()
    doc = _ScoredDoc("w", {}, 0.1)
    vs.similarity_search_with_score.return_value = [doc]
    fac = _make_with_tool(monkeypatch, vs, {"score_threshold": 0.0})
    out = fac.search_knowledge_base("q", kb_name="kb1")
    assert "w" in out


def test_search_with_query_analyzer(monkeypatch):
    vs = MagicMock()
    doc = Document(page_content="a", metadata={})
    vs.similarity_search_with_score.return_value = [(doc, 0.1)]
    fac = _make_with_tool(monkeypatch, vs, {"score_threshold": 0.0})
    fac.query_analyzer = MagicMock()
    fac.query_analyzer.analyze.return_value = ["q1", "q2"]
    out = fac.search_knowledge_base("q", kb_name="kb1")
    assert "a" in out


def test_search_custom_knowledge_base(monkeypatch):
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = []
    fac = make()
    fac.vector_store = vs
    assert "No relevant" in fac.search_custom_knowledge_base("q")


# ---- score helpers ----

def test_normalize_score():
    fac = make()
    assert fac._normalize_score(0.0, "cosine") == 1.0
    assert fac._normalize_score(0.0, "dot") == 0.5
    assert fac._normalize_score(0.0, "euclidean") == 1.0
    assert fac._normalize_score(7.0, "unknown") == 7.0


def test_detect_distance_type():
    fac = make()
    assert fac._detect_distance_type([]) == "cosine"
    assert fac._detect_distance_type([0.5]) == "cosine"
    assert fac._detect_distance_type([-1]) == "dot"
    assert fac._detect_distance_type([50]) == "euclidean"
    assert fac._detect_distance_type([5, 9]) == "cosine"


# ---- misc ----

def test_load_documents():
    fac = make()
    fac.loader = MagicMock()
    fac.load_documents(["doc1", "doc2"], session_id="s1")
    assert fac.loader.load_documents.called


def test_get_tools():
    fac = make()
    fac.knowledge_base_tools["kb1"] = {"description": "d"}
    tools = fac.get_tools()
    assert "tool:kb1" in tools
    assert "loadtool:kb1" in tools


def test_create_embeddings_import_error(monkeypatch):
    fac = make()
    monkeypatch.setitem(sys.modules, "litellm", None)
    assert fac._create_embeddings("amazon.titan") is None


def test_create_embeddings_success(monkeypatch):
    fac = make()
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.embedding = lambda **k: {"data": [{"embedding": [0.5]}]}
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    emb = fac._create_embeddings("amazon.titan-embed-text-v1")
    assert emb.model_id == "bedrock/amazon.titan-embed-text-v1"
    assert emb.embed_query("x") == [0.5]
    assert emb.embed_documents(["a"]) == [[0.5]]
