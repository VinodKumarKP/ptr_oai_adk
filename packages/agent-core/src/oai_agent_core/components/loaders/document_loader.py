import logging
import os

# try:
#     from langchain_chroma import Chroma
#     langchain_chroma_available = True
# except ImportError:
#     langchain_chroma_available = False

try:
    import chromadb
    from oai_agent_core.components.vector_store.chroma_vector_store import ChromaVectorStore
    chromadb_available = True
except ImportError:
    chromadb_available = False

from langchain_core.embeddings import Embeddings

from oai_agent_core.core.base_document_loader import BaseDocumentLoader

logging.basicConfig(level=logging.INFO)


class DocumentLoader(BaseDocumentLoader):
    """Document loader implementation using LangChain Chroma."""

    def __init__(self,
                 db_name: str = 'default_db',
                 vector_store=None,
                 embedding: Embeddings = None,
                 persist_directory: str = None,
                 collection_name: str = 'default_collection'):
        """Initialize the DocumentLoader.

        Args:
            db_name: Name of the database.
            vector_store: Optional existing vector store instance.
            embedding: Embedding function to use.
            persist_directory: Directory to persist the vector store.
        """
        if vector_store is None:
            # if langchain_chroma_available:
            #     vector_store = Chroma(embedding_function=embedding,
            #                           persist_directory=persist_directory,
            #                           collection_name=collection_name)
            if chromadb_available:
                vector_store = ChromaVectorStore(
                    collection_name=collection_name,
                    embedding_function=embedding,
                    persist_directory=persist_directory
                )
        super().__init__(db_name=db_name,
                         vector_store=vector_store,
                         embedding=embedding,
                         persist_directory=persist_directory,
                         collection_name=collection_name)

    def reinitialize_database(self) -> bool:
        """Check if the database needs to be reinitialized.

        Returns:
            True if the database should be reinitialized (empty or non-existent),
            False otherwise.
        """
        return self.vector_store.reinitialize_database()

