"""Additional coverage for PostgresVectorStore (psycopg2 mocked)."""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

import oai_agent_core.components.vector_store.postgres_vector_store as pvs
from oai_agent_core.components.vector_store.postgres_vector_store import PostgresVectorStore
from langchain_core.documents import Document


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def make_conn(cursor):
    conn = MagicMock()

    @contextmanager
    def cursor_cm():
        yield cursor

    conn.cursor.side_effect = lambda *a, **k: cursor_cm()
    return conn


@pytest.fixture
def pg(monkeypatch):
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = [0]
    cursor.rowcount = 1
    conn = make_conn(cursor)
    monkeypatch.setattr(pvs, "psycopg2", MagicMock(connect=lambda *a, **k: conn))
    monkeypatch.setattr(pvs, "register_vector", lambda c: None)
    monkeypatch.setattr(pvs, "execute_values", MagicMock())
    store = PostgresVectorStore(
        collection_name="docs",
        embedding_function=FakeEmbeddings(),
        connection_string="host=x dbname=y user=u password=p",
    )
    return store, conn, cursor


def test_missing_psycopg2(monkeypatch):
    monkeypatch.setattr(pvs, "psycopg2", None)
    with pytest.raises(ImportError):
        PostgresVectorStore(collection_name="c", embedding_function=FakeEmbeddings())


def test_requires_collection_name(monkeypatch):
    monkeypatch.setattr(pvs, "psycopg2", MagicMock())
    with pytest.raises(ValueError):
        PostgresVectorStore(embedding_function=FakeEmbeddings())


def test_requires_embedding(monkeypatch):
    monkeypatch.setattr(pvs, "psycopg2", MagicMock())
    with pytest.raises(ValueError):
        PostgresVectorStore(collection_name="c")


def test_connection_string_from_env(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = [0]
    conn = make_conn(cursor)
    monkeypatch.setattr(pvs, "psycopg2", MagicMock(connect=lambda *a, **k: conn))
    monkeypatch.setattr(pvs, "register_vector", lambda c: None)
    monkeypatch.setenv("DB_NAME", "mydb")
    monkeypatch.setenv("DB_HOST", "h")
    monkeypatch.setenv("DB_PORT", "5432")
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "pw")
    store = PostgresVectorStore(collection_name="c", embedding_function=FakeEmbeddings())
    assert "dbname=mydb" in store.connection_string


def test_missing_db_params(monkeypatch):
    monkeypatch.setattr(pvs, "psycopg2", MagicMock())
    for var in ["DB_NAME", "DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AttributeError):
        # db_name is None -> .upper() raises AttributeError inside __init__
        PostgresVectorStore(collection_name="c", embedding_function=FakeEmbeddings())


def test_init_database_failure(monkeypatch):
    monkeypatch.setattr(pvs, "psycopg2", MagicMock(connect=MagicMock(side_effect=RuntimeError("down"))))
    monkeypatch.setattr(pvs, "register_vector", lambda c: None)
    with pytest.raises(ValueError):
        PostgresVectorStore(
            collection_name="c", embedding_function=FakeEmbeddings(),
            connection_string="host=x",
        )


def test_init_database_hnsw_fallback(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone.return_value = [0]

    def execute(sql, *a, **k):
        if "hnsw" in sql:
            raise RuntimeError("no hnsw")

    cursor.execute.side_effect = execute
    conn = make_conn(cursor)
    monkeypatch.setattr(pvs, "psycopg2", MagicMock(connect=lambda *a, **k: conn))
    monkeypatch.setattr(pvs, "register_vector", lambda c: None)
    store = PostgresVectorStore(
        collection_name="c", embedding_function=FakeEmbeddings(),
        connection_string="host=x",
    )
    assert store.vector_dimensions == 3


def test_add_texts_empty(pg):
    store, _, _ = pg
    assert store.add_texts([]) == []


def test_add_texts_success(pg):
    store, _, _ = pg
    store.add_texts(["a", "b"], metadatas=[{"x": 1}, {"x": 2}])
    assert pvs.execute_values.called


def test_add_texts_generates_ids_and_metadata(pg):
    store, _, _ = pg
    store.add_texts(["a"])  # no ids, no metadatas
    assert pvs.execute_values.called


def test_add_texts_embedding_error(pg, monkeypatch):
    store, _, _ = pg
    store.embedding_function = MagicMock()
    store.embedding_function.embed_documents.side_effect = RuntimeError("emb")
    with pytest.raises(RuntimeError):
        store.add_texts(["a"])


def test_add_texts_insert_error(pg, monkeypatch):
    store, conn, _ = pg
    monkeypatch.setattr(pvs, "execute_values", MagicMock(side_effect=RuntimeError("insert")))
    with pytest.raises(RuntimeError):
        store.add_texts(["a"])
    assert conn.rollback.called


def test_add_documents(pg):
    store, _, _ = pg
    assert store.add_documents([]) is None
    store.add_documents([Document(page_content="x", metadata={"k": "v"})])


def test_similarity_search_with_filters(pg):
    store, _, cursor = pg
    cursor.fetchall.return_value = [("id1", "content", {"a": 1}, 0.2)]
    docs = store.similarity_search("q", filter={"tag": ["x", "y"], "cat": "z"})
    assert docs[0].page_content == "content"


def test_similarity_search_embedding_error(pg):
    store, _, _ = pg
    store.embedding_function = MagicMock()
    store.embedding_function.embed_query.side_effect = RuntimeError("e")
    with pytest.raises(RuntimeError):
        store.similarity_search("q")


def test_similarity_search_db_error(pg):
    store, _, cursor = pg
    cursor.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        store.similarity_search("q")


def test_similarity_search_with_score(pg):
    store, _, cursor = pg
    cursor.fetchall.return_value = [("id1", "c", None, 0.3)]
    res = store.similarity_search_with_score("q", filter={"k": "v"})
    assert res[0][1] == 0.3
    assert res[0][0].metadata == {}


def test_similarity_search_with_score_embedding_error(pg):
    store, _, _ = pg
    store.embedding_function = MagicMock()
    store.embedding_function.embed_query.side_effect = RuntimeError("e")
    with pytest.raises(RuntimeError):
        store.similarity_search_with_score("q")


def test_similarity_search_with_score_db_error(pg):
    store, _, cursor = pg
    cursor.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        store.similarity_search_with_score("q")


def test_count_success_and_error(pg):
    store, _, cursor = pg
    cursor.fetchone.return_value = [42]
    assert store.count() == 42
    cursor.execute.side_effect = RuntimeError("x")
    assert store.count() == 0


def test_delete_collection(pg):
    store, conn, _ = pg
    store.delete_collection()
    assert conn.commit.called


def test_delete_collection_error(pg):
    store, conn, cursor = pg
    cursor.execute.side_effect = RuntimeError("x")
    with pytest.raises(RuntimeError):
        store.delete_collection()
    assert conn.rollback.called


def test_delete_no_ids(pg):
    store, _, _ = pg
    assert store.delete() is False


def test_delete_success(pg):
    store, _, cursor = pg
    cursor.rowcount = 2
    assert store.delete(["a", "b"]) is True


def test_delete_error(pg):
    store, conn, cursor = pg
    cursor.execute.side_effect = RuntimeError("x")
    assert store.delete(["a"]) is False
    assert conn.rollback.called


def test_reset_collection(pg, monkeypatch):
    store, _, _ = pg
    monkeypatch.setattr(store, "delete_collection", MagicMock())
    monkeypatch.setattr(store, "_initialize_database", MagicMock())
    store.reset_collection()


def test_reset_collection_error(pg, monkeypatch):
    store, _, _ = pg
    monkeypatch.setattr(store, "delete_collection", MagicMock(side_effect=RuntimeError("x")))
    with pytest.raises(RuntimeError):
        store.reset_collection()


def test_query_with_text(pg, monkeypatch):
    store, _, _ = pg
    monkeypatch.setattr(store, "similarity_search", lambda **k: [
        Document(page_content="a", metadata={"ts": 2}),
        Document(page_content="b", metadata={"ts": 1}),
    ])
    docs = store.query(query_text="q", order_by="ts", n_results=2)
    assert docs[0].metadata["ts"] == 2


def test_query_filter_only(pg, monkeypatch):
    store, _, _ = pg
    monkeypatch.setattr(store, "_get_by_metadata", lambda f, n: [Document(page_content="x", metadata={})])
    docs = store.query(filter_metadata={"k": "v"})
    assert len(docs) == 1


def test_query_no_args(pg):
    store, _, _ = pg
    assert store.query() == []


def test_query_sort_error_handled(pg, monkeypatch):
    store, _, _ = pg
    bad_doc = MagicMock()
    bad_doc.metadata = None  # .get will fail
    monkeypatch.setattr(store, "similarity_search", lambda **k: [bad_doc])
    docs = store.query(query_text="q", order_by="ts")
    assert docs == [bad_doc]


def test_query_outer_error(pg, monkeypatch):
    store, _, _ = pg
    monkeypatch.setattr(store, "similarity_search", MagicMock(side_effect=RuntimeError("x")))
    with pytest.raises(RuntimeError):
        store.query(query_text="q")


def test_get_by_metadata(pg):
    store, _, cursor = pg
    cursor.fetchall.return_value = [("id", "content", {"a": 1})]
    docs = store._get_by_metadata({"tag": ["x", "y"], "cat": "z"})
    assert docs[0].page_content == "content"


def test_get_by_metadata_error(pg):
    store, _, cursor = pg
    cursor.execute.side_effect = RuntimeError("x")
    assert store._get_by_metadata({"k": "v"}) == []


def test_from_texts(pg, monkeypatch):
    monkeypatch.setattr(PostgresVectorStore, "add_texts", MagicMock())
    cursor = MagicMock()
    cursor.fetchone.return_value = [0]
    conn = make_conn(cursor)
    monkeypatch.setattr(pvs, "psycopg2", MagicMock(connect=lambda *a, **k: conn))
    monkeypatch.setattr(pvs, "register_vector", lambda c: None)
    store = PostgresVectorStore.from_texts(
        ["a"], FakeEmbeddings(), collection_name="c",
        connection_string="host=x",
    )
    assert isinstance(store, PostgresVectorStore)
