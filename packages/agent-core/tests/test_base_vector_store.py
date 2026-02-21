from typing import Any, Tuple, Optional, List

import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from oai_agent_core.core.base_vector_store import BaseVectorStore


class ConcreteVectorStore(BaseVectorStore):
    def query(self, query_text: Optional[str] = None, filter_metadata: Optional[dict] = None, n_results: int = 4,
              order_by: Optional[str] = None, order: str = "desc", **kwargs: Any) -> List[Document]:
        pass

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[Tuple[Document, float]]:
        pass

    def add_documents(self, documents, ids=None):
        pass
    
    def similarity_search(self, query, k=4, **kwargs):
        return []
    
    def count(self):
        return 0
    
    def delete(self, ids=None, **kwargs):
        return True
    
    def reset_collection(self):
        pass
        
    def add_texts(self, texts, metadatas=None, **kwargs):
        pass
        
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        pass


def test_init():
    store = ConcreteVectorStore(
        collection_name="test_coll",
        embedding_function="embed_func",
        persist_directory="/tmp"
    )
    
    assert store.collection_name == "test_coll"
    assert store.embedding_function == "embed_func"
    assert store.persist_directory == "/tmp"


def test_init_defaults():
    store = ConcreteVectorStore()
    assert store.collection_name is None
    assert store.embedding_function is None
    assert store.persist_directory is None


def test_abstract_methods():
    store = ConcreteVectorStore()
    assert store.count() == 0
    assert store.similarity_search("query") == []
    assert store.delete() is True
    store.reset_collection()  # Should not raise
    store.add_documents([])  # Should not raise
    store.add_texts([])  # Should not raise
    ConcreteVectorStore.from_texts([], None)  # Should not raise