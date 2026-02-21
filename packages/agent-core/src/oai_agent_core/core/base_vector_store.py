"""Base class for Vector Store implementations."""
import logging
from abc import abstractmethod
from typing import List, Any, Optional, Iterable, Tuple

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError:
    logging.warn("Please install langchain-core: pip install langchain-core")
    Document = Any
    VectorStore = Any


class BaseVectorStore(VectorStore):
    """Abstract base class for vector store wrappers.
    
    This class defines the interface that any vector store implementation
    must follow to be compatible with the DocumentLoader and KnowledgeBaseFactory.
    """

    def __init__(self,
                 **kwargs):
        """Initialize the vector store.

        Args:
            collection_name: Name of the collection/index
            embedding_function: Embedding function to use
            persist_directory: Directory for local persistence (optional)
            **kwargs: Additional store-specific arguments
        """
        self.collection_name = kwargs.get('collection_name')
        self.embedding_function = kwargs.get('embedding_function')
        self.persist_directory = kwargs.get('persist_directory')

    @abstractmethod
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Search for similar documents.

        Args:
            query: The query string
            k: Number of results to return

        Returns:
            List of matching Document objects
        """
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[Tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query: The query string
            k: Number of results to return

        Returns:
            List of matching Document objects
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store.

        Returns:
            Count of documents
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete the entire collection/index."""
        pass

    @abstractmethod
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        pass

    @abstractmethod
    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store."""
        pass

    @abstractmethod
    def query(
            self,
            query_text: Optional[str] = None,
            filter_metadata: Optional[dict] = None,
            n_results: int = 4,
            order_by: Optional[str] = None,
            order: str = "desc",
            **kwargs: Any
    ) -> List[Document]:
        """
        Query the vector store using text, metadata filters, or both.
        """
        pass

    def reinitialize_database(self) -> bool:
        """
        Check if the database needs to be reinitialized.

        Returns:
            bool: True if the database needs to be reinitialized, False otherwise.
        """
        return self.count() == 0