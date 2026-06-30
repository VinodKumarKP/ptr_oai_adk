"""Additional coverage for S3VectorStore (boto3 mocked)."""
import json
from unittest.mock import MagicMock

import pytest

import oai_agent_core.components.vector_store.s3_vector_store as s3mod
from oai_agent_core.components.vector_store.s3_vector_store import S3VectorStore
from langchain_core.documents import Document


class NoSuchKey(Exception):
    pass


class FakeBody:
    def __init__(self, data):
        self._data = data

    def read(self):
        return json.dumps(self._data).encode("utf-8")


class FakeS3Client:
    def __init__(self, index_data=None, raise_no_key=False, raise_other=False):
        self._index_data = index_data
        self._raise_no_key = raise_no_key
        self._raise_other = raise_other
        self.exceptions = MagicMock()
        self.exceptions.NoSuchKey = NoSuchKey
        self.put_object = MagicMock()
        self.delete_object = MagicMock()

    def get_object(self, Bucket, Key):
        if self._raise_no_key:
            raise NoSuchKey()
        if self._raise_other:
            raise RuntimeError("network")
        return {"Body": FakeBody(self._index_data or {})}


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def make_store(monkeypatch, client=None):
    client = client or FakeS3Client(raise_no_key=True)
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: client)
    store = S3VectorStore(
        collection_name="docs",
        embedding_function=FakeEmbeddings(),
        bucket_name="bucket",
    )
    return store, client


def test_requires_collection_name(monkeypatch):
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: FakeS3Client())
    with pytest.raises(ValueError):
        S3VectorStore(embedding_function=FakeEmbeddings(), bucket_name="b")


def test_requires_embedding(monkeypatch):
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: FakeS3Client())
    with pytest.raises(ValueError):
        S3VectorStore(collection_name="c", bucket_name="b")


def test_bucket_from_env(monkeypatch):
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: FakeS3Client(raise_no_key=True))
    monkeypatch.setenv("S3_VECTOR_BUCKET", "envbucket")
    store = S3VectorStore(collection_name="docs", embedding_function=FakeEmbeddings())
    assert store.bucket_name == "envbucket"


def test_missing_bucket(monkeypatch):
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: FakeS3Client())
    monkeypatch.delenv("S3_VECTOR_BUCKET", raising=False)
    monkeypatch.delenv("DOCS_VECTOR_BUCKET", raising=False)
    with pytest.raises(ValueError):
        S3VectorStore(collection_name="docs", embedding_function=FakeEmbeddings())


def test_load_index_existing(monkeypatch):
    data = {
        "ids": ["1"],
        "vectors": [[1.0, 0.0]],
        "documents": [{"page_content": "hello", "metadata": {"k": "v"}}],
    }
    client = FakeS3Client(index_data=data)
    store, _ = make_store(monkeypatch, client)
    assert store.count() == 1


def test_load_index_other_error(monkeypatch):
    client = FakeS3Client(raise_other=True)
    store, _ = make_store(monkeypatch, client)
    assert store.count() == 0


def test_add_texts_and_save(monkeypatch):
    store, client = make_store(monkeypatch)
    ids = store.add_texts(["a", "b"], metadatas=[{"x": 1}, {"x": 2}])
    assert len(ids) == 2
    assert client.put_object.called


def test_add_texts_empty(monkeypatch):
    store, _ = make_store(monkeypatch)
    assert store.add_texts([]) == []


def test_add_texts_default_ids_metadata(monkeypatch):
    store, _ = make_store(monkeypatch)
    ids = store.add_texts(["a"])
    assert len(ids) == 1


def test_save_index_error(monkeypatch):
    store, client = make_store(monkeypatch)
    client.put_object.side_effect = RuntimeError("s3 down")
    with pytest.raises(RuntimeError):
        store.add_texts(["a"])


def test_add_documents(monkeypatch):
    store, _ = make_store(monkeypatch)
    assert store.add_documents([]) is None
    store.add_documents([Document(page_content="x", metadata={"k": "v"})])
    assert store.count() == 1


def test_similarity_search_empty(monkeypatch):
    store, _ = make_store(monkeypatch)
    assert store.similarity_search("q") == []


def test_similarity_search_with_score(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a", "b"], metadatas=[{"t": "x"}, {"t": "y"}])
    res = store.similarity_search_with_score("q", k=1)
    assert len(res) == 1
    docs = store.similarity_search("q", k=2)
    assert len(docs) == 2


def test_similarity_search_with_filter(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a", "b"], metadatas=[{"t": "x"}, {"t": "y"}])
    res = store.similarity_search_with_score("q", filter={"t": "x"})
    assert all(d.metadata["t"] == "x" for d, _ in res)
    res_list = store.similarity_search_with_score("q", filter={"t": ["x", "y"]})
    assert len(res_list) == 2


def test_similarity_search_no_valid_indices(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a"], metadatas=[{"t": "x"}])
    assert store.similarity_search_with_score("q", filter={"t": "nomatch"}) == []


def test_similarity_search_zero_query_norm(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a"])
    store.embedding_function = MagicMock()
    store.embedding_function.embed_query.return_value = [0.0, 0.0]
    assert store.similarity_search_with_score("q") == []


def test_delete_all(monkeypatch):
    store, client = make_store(monkeypatch)
    store.add_texts(["a"])
    assert store.delete() is True
    assert store.count() == 0


def test_delete_empty_ids(monkeypatch):
    store, _ = make_store(monkeypatch)
    assert store.delete([]) is False


def test_delete_by_ids(monkeypatch):
    store, _ = make_store(monkeypatch)
    ids = store.add_texts(["a", "b"])
    assert store.delete([ids[0]]) is True
    assert store.count() == 1


def test_delete_by_ids_no_match(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a"])
    assert store.delete(["nonexistent"]) is False


def test_reset_collection_error(monkeypatch):
    store, client = make_store(monkeypatch)
    client.delete_object.side_effect = RuntimeError("x")
    store.add_texts(["a"])
    store.reset_collection()  # error swallowed, in-memory cleared
    assert store.count() == 0


def test_query_with_text_and_order(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a", "b"], metadatas=[{"ts": 1}, {"ts": 2}])
    docs = store.query(query_text="q", order_by="ts", order="desc", n_results=2)
    assert docs[0].metadata["ts"] == 2


def test_query_filter_only(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a", "b"], metadatas=[{"t": "x"}, {"t": "y"}])
    docs = store.query(filter_metadata={"t": "x"})
    assert len(docs) == 1


def test_query_filter_list(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a", "b"], metadatas=[{"t": "x"}, {"t": "z"}])
    docs = store.query(filter_metadata={"t": ["x", "y"]})
    assert len(docs) == 1


def test_query_zero_norm(monkeypatch):
    store, _ = make_store(monkeypatch)
    store.add_texts(["a"])
    store.embedding_function = MagicMock()
    store.embedding_function.embed_query.return_value = [0.0, 0.0]
    docs = store.query(query_text="q")
    assert len(docs) == 1


def test_from_texts(monkeypatch):
    client = FakeS3Client(raise_no_key=True)
    monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: client)
    store = S3VectorStore.from_texts(
        ["a"], FakeEmbeddings(), collection_name="docs", bucket_name="b"
    )
    assert store.count() == 1
