"""Additional coverage for ChromaVectorStore."""
from unittest.mock import MagicMock

import pytest

import oai_agent_core.components.vector_store.chroma_vector_store as cvs
from oai_agent_core.components.vector_store.chroma_vector_store import ChromaVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]


def make_store(monkeypatch, collection=None):
    collection = collection or MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.create_collection.return_value = collection
    monkeypatch.setattr(cvs.chromadb, "PersistentClient", lambda **k: client)
    monkeypatch.setattr(cvs.chromadb, "HttpClient", lambda **k: client)
    store = ChromaVectorStore(
        collection_name="c", embedding_function=FakeEmbeddings()
    )
    store._client = client
    store._collection = collection
    return store, client, collection


def test_requires_collection_name():
    with pytest.raises(ValueError):
        ChromaVectorStore(embedding_function=FakeEmbeddings())


def test_requires_embedding(monkeypatch):
    with pytest.raises(ValueError):
        ChromaVectorStore(collection_name="c")


def test_remote_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(cvs.chromadb, "HttpClient", lambda **k: client)
    store = ChromaVectorStore(
        collection_name="c", embedding_function=FakeEmbeddings(),
        host="localhost", port=8000, ssl=True,
    )
    assert store.client is client


def test_client_init_failure(monkeypatch):
    def boom(**k):
        raise RuntimeError("nope")

    monkeypatch.setattr(cvs.chromadb, "PersistentClient", boom)
    with pytest.raises(ValueError):
        ChromaVectorStore(collection_name="c", embedding_function=FakeEmbeddings())


def test_collection_init_failure(monkeypatch):
    client = MagicMock()
    client.get_or_create_collection.side_effect = RuntimeError("bad")
    monkeypatch.setattr(cvs.chromadb, "PersistentClient", lambda **k: client)
    with pytest.raises(ValueError):
        ChromaVectorStore(collection_name="c", embedding_function=FakeEmbeddings())


def test_from_texts_is_noop():
    assert ChromaVectorStore.from_texts(["t"], FakeEmbeddings()) is None


def test_add_texts_empty(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    assert store.add_texts([]) is None
    collection.add.assert_not_called()


def test_add_texts_generates_ids(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    store.add_texts(["a", "b"])
    assert collection.add.called
    kwargs = collection.add.call_args.kwargs
    assert len(kwargs["ids"]) == 2


def test_add_documents(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    from langchain_core.documents import Document

    assert store.add_documents([]) is None
    store.add_documents([Document(page_content="x", metadata={"k": "v"})])
    assert collection.add.called


def test_similarity_search_with_filters(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.query.return_value = {
        "documents": [["doc1"]],
        "metadatas": [[{"a": 1}]],
        "distances": [[0.5]],
    }
    # multi-value list -> $or; plus a scalar -> $and across 2 filters
    res = store.similarity_search("q", filter={"tag": ["x", "y"], "cat": "z"})
    assert len(res) == 1
    res2 = store.similarity_search_with_score("q", filter={"tag": ["only"]})
    assert res2[0][1] == 0.5


def test_similarity_search_single_filter(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.query.return_value = {
        "documents": [["d"]], "metadatas": [[{}]], "distances": [[0.1]],
    }
    res = store.similarity_search("q", filter={"only": "v"})
    assert len(res) == 1


def test_count_and_delete(monkeypatch):
    store, client, collection = make_store(monkeypatch)
    collection.count.return_value = 7
    assert store.count() == 7
    assert store.delete() is True
    assert client.delete_collection.called
    store.reset_collection()


def test_query_with_text_and_order(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.query.return_value = {
        "documents": [["d1", "d2"]],
        "metadatas": [[{"timestamp": 2}, {"timestamp": 1}]],
    }
    docs = store.query(query_text="q", filter_metadata={"u": "1"}, order_by="timestamp", n_results=2)
    assert docs[0].metadata["timestamp"] == 2  # desc sort


def test_query_metadata_only(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.get.return_value = {
        "documents": ["a", "b"],
        "metadatas": [{"timestamp": 1}, {"timestamp": 3}],
    }
    docs = store.query(filter_metadata={"u": ["x", "y"]}, order_by="timestamp", order="asc")
    assert docs[0].metadata["timestamp"] == 1  # asc


def test_query_metadata_only_no_order(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.get.return_value = {"documents": ["a"], "metadatas": [{}]}
    docs = store.query(filter_metadata={"u": "x"})
    assert len(docs) == 1


def test_reinitialize_database(monkeypatch):
    store, _, collection = make_store(monkeypatch)
    collection.count.return_value = 0
    assert store.reinitialize_database() is True
    collection.count.return_value = 3
    assert store.reinitialize_database() is False
