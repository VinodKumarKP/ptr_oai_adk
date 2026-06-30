"""Additional coverage for BaseDocumentLoader."""
import json
import os
import time
import types
from unittest.mock import MagicMock

import pytest

import oai_agent_core.core.base_document_loader as bdl
from oai_agent_core.core.base_document_loader import BaseDocumentLoader, LoaderError
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class ConcreteLoader(BaseDocumentLoader):
    reinit_value = False

    def reinitialize_database(self) -> bool:
        return self.reinit_value


def make(tmp_path, vector_store=None, reinit=False):
    ConcreteLoader.reinit_value = reinit
    vs = vector_store if vector_store is not None else MagicMock()
    return ConcreteLoader(
        db_name="db", vector_store=vs, persist_directory=str(tmp_path),
        collection_name="col",
    )


def test_init_requires_store_or_embedding():
    with pytest.raises(LoaderError):
        ConcreteLoader()


# ---- transform_documents_with_metadata ----

def test_transform_empty(tmp_path):
    loader = make(tmp_path)
    assert loader.transform_documents_with_metadata([]) == []


def test_transform_not_list(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(TypeError):
        loader.transform_documents_with_metadata("notalist")


def test_transform_skips_invalid(tmp_path):
    loader = make(tmp_path)
    docs = [
        "not a doc",
        Document(page_content="hi", metadata={"title": "T", "source": "s"}),
    ]
    out = loader.transform_documents_with_metadata(docs)
    assert len(out) == 1
    assert "Title: T" in out[0].page_content


def test_transform_all_failed(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader.transform_documents_with_metadata(["x", "y"])


# ---- _get_loader_for_file ----

def test_get_loader_for_file_txt(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("hello")
    inst = loader._get_loader_for_file(str(f))
    assert inst is not None


def test_get_loader_for_file_custom_class(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("hello")
    settings = {"txt": {"class": "langchain_community.document_loaders.TextLoader",
                        "settings": {"encoding": "utf-8"}}}
    inst = loader._get_loader_for_file(str(f), loader_settings=settings)
    assert inst is not None


def test_get_loader_for_file_unsupported(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_loader_for_file("file.unknownext")


def test_get_loader_for_file_no_extension(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_loader_for_file("noext")


# ---- _get_loader ----

def test_get_loader_baseloader_passthrough(tmp_path):
    loader = make(tmp_path)

    class MyLoader(BaseLoader):
        def load(self):
            return []

    inst = MyLoader()
    assert loader._get_loader(inst) is inst


def test_get_loader_glob(tmp_path):
    loader = make(tmp_path)
    result = loader._get_loader(str(tmp_path / "*.txt"))
    assert result is not None


def test_get_loader_isdir(tmp_path):
    loader = make(tmp_path)
    result = loader._get_loader(str(tmp_path))
    assert result is not None


def test_get_loader_isfile(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")
    result = loader._get_loader(str(f))
    assert result is not None


def test_get_loader_unsupported(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_loader(12345)


def test_get_loader_wiki(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_mod = types.ModuleType("langchain_community.document_loaders")
    fake_mod.WikipediaLoader = lambda query: f"wiki:{query}"
    monkeypatch.setitem(__import__("sys").modules, "langchain_community.document_loaders", fake_mod)
    assert loader._get_loader("wiki/Python") == "wiki:Python"
    assert loader._get_loader("query: Foo") == "wiki:Foo"


# ---- _load_documents ----

def test_load_documents_empty(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(ValueError):
        loader._load_documents([])


def test_load_documents_success(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("content here")
    docs = loader._load_documents([str(f)])
    assert len(docs) >= 1


def test_load_documents_all_failed(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._load_documents([str(tmp_path / "missing.txt")])


def test_load_documents_directory(tmp_path, monkeypatch):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("content")
    # glob path triggers DirectoryLoader branch
    docs = loader._load_documents([str(tmp_path / "*.txt")])
    assert isinstance(docs, list)


# ---- _process_s3_source ----

def test_process_s3_no_boto3(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "boto3", None)
    files, loaded = loader._process_s3_source("s3://bucket/prefix", {}, None, {})
    assert files == [] and loaded == {}


def _fake_boto3(monkeypatch, pages, download_error=False, list_error=False):
    fake = types.ModuleType("boto3")
    client = MagicMock()
    paginator = MagicMock()
    if list_error:
        paginator.paginate.side_effect = RuntimeError("list fail")
    else:
        paginator.paginate.return_value = pages
    client.get_paginator.return_value = paginator
    if download_error:
        client.download_file.side_effect = RuntimeError("dl fail")
    fake.client = lambda *a, **k: client
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
    return client


def test_process_s3_success(tmp_path, monkeypatch):
    loader = make(tmp_path)
    pages = [{"Contents": [
        {"Key": "doc.txt", "Size": 10},
        {"Key": "folder/", "Size": 0},  # skipped (dir)
    ]}]
    _fake_boto3(monkeypatch, pages)
    files, loaded = loader._process_s3_source("s3://bucket/prefix", {}, None, {"region": "us-east-1"})
    assert len(files) == 1


def test_process_s3_skip_already_loaded(tmp_path, monkeypatch):
    loader = make(tmp_path)
    pages = [{"Contents": [{"Key": "doc.txt", "Size": 10}]}]
    _fake_boto3(monkeypatch, pages)
    loaded_files = {"s3://bucket/doc.txt": 10}
    files, loaded = loader._process_s3_source("s3://bucket", loaded_files, None, {})
    assert files == []


def test_process_s3_download_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    pages = [{"Contents": [{"Key": "doc.txt", "Size": 10}]}]
    _fake_boto3(monkeypatch, pages, download_error=True)
    files, loaded = loader._process_s3_source("s3://bucket", {}, "sess", {})
    assert files == []


def test_process_s3_list_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    _fake_boto3(monkeypatch, [], list_error=True)
    files, loaded = loader._process_s3_source("s3://bucket", {}, None, {})
    assert files == [] and loaded == {}


def test_process_s3_no_contents(tmp_path, monkeypatch):
    loader = make(tmp_path)
    _fake_boto3(monkeypatch, [{}])
    files, loaded = loader._process_s3_source("s3://bucket", {}, None, {})
    assert files == []


# ---- _process_local_source ----

def test_process_local_glob(tmp_path):
    loader = make(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    files, loaded = loader._process_local_source(str(tmp_path / "*.txt"), {})
    assert len(files) == 1


def test_process_local_dir(tmp_path):
    loader = make(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    files, loaded = loader._process_local_source(str(tmp_path), {})
    assert len(files) == 1


def test_process_local_single_file(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")
    files, loaded = loader._process_local_source(str(f), {})
    assert len(files) == 1


def test_process_local_skip_loaded(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")
    size = os.path.getsize(f)
    files, loaded = loader._process_local_source(str(f), {"a.txt": size})
    assert files == []


def test_process_local_session_id(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")
    files, loaded = loader._process_local_source(str(f), {}, session_id="s1")
    assert any(k.startswith("s1::") for k in loaded)


# ---- _get_dynamic_loader ----

def test_get_dynamic_loader_invalid_path(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_dynamic_loader("NoDotsClass", {})


def test_get_dynamic_loader_import_error(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_dynamic_loader("nonexistent_module_zzz.SomeClass", {})


def test_get_dynamic_loader_attribute_error(tmp_path):
    loader = make(tmp_path)
    with pytest.raises(LoaderError):
        loader._get_dynamic_loader("langchain_community.document_loaders.NoSuchLoaderZZZ", {})


def test_get_dynamic_loader_instantiate_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_mod = types.ModuleType("fake_loader_mod")

    class BadLoader:
        def __init__(self):  # accepts no kwargs
            pass

    fake_mod.BadLoader = BadLoader
    monkeypatch.setitem(__import__("sys").modules, "fake_loader_mod", fake_mod)
    with pytest.raises(LoaderError):
        loader._get_dynamic_loader("fake_loader_mod.BadLoader", {"unexpected": 1})


def test_get_dynamic_loader_success(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_mod = types.ModuleType("fake_loader_mod2")

    class GoodLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod.GoodLoader = GoodLoader
    monkeypatch.setitem(__import__("sys").modules, "fake_loader_mod2", fake_mod)
    inst = loader._get_dynamic_loader("fake_loader_mod2.GoodLoader", {"a": 1})
    assert inst.kwargs == {"a": 1}


# ---- serialisable settings / cache key ----

def test_serialisable_settings(tmp_path):
    loader = make(tmp_path)
    out = loader._serialisable_settings({"a": 1, "fn": lambda x: x, "b": "s"})
    assert out == {"a": 1, "b": "s"}


def test_dynamic_loader_cache_key(tmp_path):
    loader = make(tmp_path)
    k1 = loader._dynamic_loader_cache_key("Loader", {"a": 1, "fn": lambda x: x})
    k2 = loader._dynamic_loader_cache_key("Loader", {"a": 1})
    assert k1 == k2


# ---- _process_dynamic_source ----

def test_process_dynamic_cache_hit(tmp_path, monkeypatch):
    loader = make(tmp_path)
    key = loader._dynamic_loader_cache_key("L", {})
    loaded = {key: {"__dynamic__": True, "last_fetched": time.time()}}
    docs, updates = loader._process_dynamic_source("L", {}, loaded, ttl_seconds=3600)
    assert docs == [] and updates == {}


def test_process_dynamic_success(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_loader = MagicMock()
    fake_loader.load.return_value = [Document(page_content="x", metadata={})]
    monkeypatch.setattr(loader, "_get_dynamic_loader", lambda p, s: fake_loader)
    docs, updates = loader._process_dynamic_source("L", {}, {}, ttl_seconds=0)
    assert docs[0].metadata["source"] == "L"
    assert updates


def test_process_dynamic_load_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_loader = MagicMock()
    fake_loader.load.side_effect = RuntimeError("boom")
    monkeypatch.setattr(loader, "_get_dynamic_loader", lambda p, s: fake_loader)
    with pytest.raises(LoaderError):
        loader._process_dynamic_source("L", {}, {}, ttl_seconds=0)


def test_process_dynamic_no_docs(tmp_path, monkeypatch):
    loader = make(tmp_path)
    fake_loader = MagicMock()
    fake_loader.load.return_value = []
    monkeypatch.setattr(loader, "_get_dynamic_loader", lambda p, s: fake_loader)
    docs, updates = loader._process_dynamic_source("L", {}, {}, ttl_seconds=0)
    assert docs == [] and updates == {}


# ---- _update_s3_metadata ----

def test_update_s3_metadata(tmp_path):
    loader = make(tmp_path)
    src = os.path.join(str(tmp_path), "s3_bucket", "mybucket", "doc.txt")
    doc = Document(page_content="x", metadata={"source": src})
    loader._update_s3_metadata([doc])
    assert doc.metadata["source"].startswith("s3://")


def test_update_s3_metadata_abs_path(tmp_path):
    loader = make(tmp_path)
    doc = Document(page_content="x", metadata={"source": "/other/s3_bucket/mybucket/doc.txt"})
    loader._update_s3_metadata([doc])
    assert doc.metadata["source"].startswith("s3://")


# ---- loaded files log ----

def test_loaded_files_log_roundtrip(tmp_path):
    loader = make(tmp_path)
    loader._save_loaded_files({"a": 1, "d": {"fn": lambda x: x, "n": 2}})
    out = loader._get_loaded_files()
    assert out["a"] == 1
    assert out["d"] == {"n": 2}


def test_get_loaded_files_corrupt(tmp_path):
    loader = make(tmp_path)
    with open(loader.loaded_files_log, "w") as f:
        f.write("{not json")
    assert loader._get_loaded_files() == {}


# ---- _split_text ----

def test_split_text(tmp_path):
    loader = make(tmp_path)
    docs = [Document(page_content="word " * 500, metadata={})]
    docs_dict = {"doc": {"chunk_size": 100, "chunk_overlap": 10}}
    out = loader._split_text("doc", docs, docs_dict)
    assert len(out) >= 1


# ---- document count / collection name ----

def test_get_document_count(tmp_path):
    vs = MagicMock()
    vs.count.return_value = 7
    loader = make(tmp_path, vector_store=vs)
    assert loader._get_document_count() == 7


def test_get_document_count_collection(tmp_path):
    vs = MagicMock(spec=["_collection"])
    vs._collection = MagicMock()
    vs._collection.count.return_value = 3
    loader = make(tmp_path, vector_store=vs)
    assert loader._get_document_count() == 3


def test_get_collection_name(tmp_path):
    vs = MagicMock()
    vs.collection_name = "mycol"
    loader = make(tmp_path, vector_store=vs)
    assert loader._get_collection_name() == "mycol"


# ---- _generate_chunk ----

def test_generate_chunk_local(tmp_path, monkeypatch):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("some content here that is non empty")
    docs_dict = {str(f): {"type": "file", "chunk_size": 100, "chunk_overlap": 0}}
    chunks, loaded = loader._generate_chunk(docs_dict)
    assert len(chunks) >= 1


def test_generate_chunk_all_loaded(tmp_path):
    loader = make(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("content")
    size = os.path.getsize(f)
    # pre-populate loaded files log
    loader._save_loaded_files({"a.txt": size})
    docs_dict = {str(f): {"type": "file"}}
    chunks, loaded = loader._generate_chunk(docs_dict)
    assert chunks == []


def test_generate_chunk_dynamic(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setattr(
        loader, "_process_dynamic_source",
        lambda **k: ([Document(page_content="dyn content", metadata={})], {"key": {"__dynamic__": True}}),
    )
    docs_dict = {"dyn1": {"type": "dynamic", "loader_class": "X.Loader", "chunk_size": 100, "chunk_overlap": 0}}
    chunks, loaded = loader._generate_chunk(docs_dict)
    assert len(chunks) >= 1


def test_generate_chunk_dynamic_missing_class(tmp_path):
    loader = make(tmp_path)
    docs_dict = {"dyn1": {"type": "dynamic"}}
    with pytest.raises(LoaderError):
        loader._generate_chunk(docs_dict)


def test_generate_chunk_s3(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setattr(loader, "_process_s3_source",
                        lambda *a, **k: (["/local/f.txt"], {"s3://b/f.txt": 5}))
    monkeypatch.setattr(loader, "_load_documents",
                        lambda knowledge_base_list, loader_settings=None: [Document(page_content="c", metadata={})])
    docs_dict = {"s3://b/prefix": {"type": "s3", "chunk_size": 100, "chunk_overlap": 0}}
    chunks, loaded = loader._generate_chunk(docs_dict)
    assert len(chunks) >= 1


# ---- load_db / load_documents / query / stats / reset ----

def test_load_db_and_documents(tmp_path, monkeypatch):
    vs = MagicMock()
    vs.count.return_value = 0
    loader = make(tmp_path, vector_store=vs)
    f = tmp_path / "a.txt"
    f.write_text("hello content here")
    docs_dict = {str(f): {"type": "file", "chunk_size": 100, "chunk_overlap": 0}}
    loader.load_db(docs_dict)
    assert vs.add_documents.called


def test_load_db_reinitialize(tmp_path):
    vs = MagicMock()
    vs.count.return_value = 5
    loader = make(tmp_path, vector_store=vs, reinit=True)
    f = tmp_path / "a.txt"
    f.write_text("hello content")
    loader.load_db({str(f): {"type": "file", "chunk_size": 100, "chunk_overlap": 0}})
    assert vs.reset_collection.called


def test_load_db_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setattr(loader, "_get_document_count", MagicMock(side_effect=RuntimeError("x")))
    with pytest.raises(LoaderError):
        loader.load_db({})


def test_load_documents_no_chunks(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setattr(loader, "_generate_chunk", lambda docs_dict: ([], {}))
    loader.load_documents({})  # no error, just returns


def test_query(tmp_path):
    vs = MagicMock()
    vs.similarity_search.return_value = [Document(page_content="r", metadata={})]
    loader = make(tmp_path, vector_store=vs)
    assert len(loader.query("q")) == 1


def test_query_no_store(tmp_path):
    loader = make(tmp_path)
    loader.vector_store = None
    assert loader.query("q") == []


def test_get_collection_stats(tmp_path):
    vs = MagicMock()
    vs.collection_name = "c"
    vs.count.return_value = 2
    vs.collection = MagicMock()
    vs.collection.metadata = {"k": "v"}
    loader = make(tmp_path, vector_store=vs)
    stats = loader.get_collection_stats()
    assert stats["count"] == 2
    assert stats["metadata"] == {"k": "v"}


def test_reset_collection(tmp_path):
    vs = MagicMock()
    vs.collection_name = "c"
    loader = make(tmp_path, vector_store=vs)
    loader._save_loaded_files({"a": 1})
    loader.reset_collection()
    assert vs.reset_collection.called
    assert not os.path.exists(loader.loaded_files_log)


def test_reset_collection_error(tmp_path, monkeypatch):
    loader = make(tmp_path)
    monkeypatch.setattr(loader, "_get_collection_name", MagicMock(side_effect=RuntimeError("x")))
    with pytest.raises(LoaderError):
        loader.reset_collection()
