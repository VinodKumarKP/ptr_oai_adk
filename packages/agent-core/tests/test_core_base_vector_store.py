from typing import Optional, Any, List, Tuple

from langchain_core.documents import Document

from oai_agent_core.core.base_vector_store import BaseVectorStore


class ConcreteVectorStore(BaseVectorStore):
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[Tuple[Document, float]]:
        pass

    def query(self, query_text: Optional[str] = None, filter_metadata: Optional[dict] = None, n_results: int = 4,
              order_by: Optional[str] = None, order: str = "desc", **kwargs: Any) -> List[Document]:
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
        
    # Implement abstract methods from VectorStore (langchain)
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

def test_abstract_methods():
    # Just verifying instantiation works with concrete class
    store = ConcreteVectorStore()
    assert store.count() == 0
