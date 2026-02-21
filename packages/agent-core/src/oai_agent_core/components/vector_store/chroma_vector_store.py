"""ChromaDB implementation of BaseVectorStore."""

import logging
import os
import tempfile
from typing import List, Optional, Any, Iterable, Tuple
from uuid import uuid4

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST

from oai_agent_core.core.base_vector_store import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of the vector store interface."""

    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        self.delete_collection()

    @classmethod
    def from_texts(cls: type[VST], texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None, *,
                   ids: list[str] | None = None, **kwargs: Any) -> VST:
        pass

    def __init__(self, **kwargs):
        """Initialize ChromaDB vector store.

        Args:
            **kwargs: Arguments including collection_name, embedding_function, persist_directory,
                     host, port, ssl, settings
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        # Validate required arguments
        if not self.collection_name:
            raise ValueError("collection_name is required")
        if not self.embedding_function:
            raise ValueError("embedding_function is required")

        host = kwargs.get('host')
        port = kwargs.get('port')
        ssl = kwargs.get('ssl', False)
        settings = kwargs.get('settings')

        # Initialize Client
        try:
            if host and port:
                self.logger.info(f"Connecting to remote ChromaDB at {host}:{port} (SSL: {ssl})")
                self.client = chromadb.HttpClient(
                    host=host,
                    port=str(port),
                    ssl=ssl,
                    settings=settings
                )
            else:
                persist_dir = self.persist_directory
                if not persist_dir:
                    persist_dir = os.path.join(tempfile.gettempdir(), 'default_db')
                self.logger.info(f"Using local ChromaDB at {persist_dir}")
                self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        except Exception as e:
            raise ValueError(f"Failed to initialize ChromaDB client: {str(e)}")

        # Initialize Collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} )
        except Exception as e:
            raise ValueError(f"Failed to get or create collection: {str(e)}")

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> None:
        """Add texts to ChromaDB."""
        if not texts:
            return

        embeddings = self.embedding_function.embed_documents(texts)
        texts = [text for text in texts]
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts))]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Add documents to ChromaDB."""
        if not documents:
            return

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]

        doc_texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_function.embed_documents(doc_texts)

        self.collection.add(
            ids=ids,
            documents=doc_texts,
            embeddings=embeddings,
            metadatas=metadatas
        )


    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Run similarity search in ChromaDB."""
        query_embedding = self.embedding_function.embed_query(query)

        filter_metadata = kwargs.get('filter')
        where_clause = None
        if filter_metadata:
            filters = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    if len(value) > 1:
                        # If value is a list with multiple items, use $or operator
                        filters.append({"$or": [{key: v} for v in value]})
                    elif len(value) == 1:
                        # If value is a list with single item, use direct equality
                        filters.append({key: value[0]})
                else:
                    filters.append({key: value})
            
            if len(filters) > 1:
                where_clause = {"$and": filters}
            elif filters:
                where_clause = filters[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )

        documents = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Run similarity search in ChromaDB."""
        query_embedding = self.embedding_function.embed_query(query)

        filter_metadata = kwargs.get('filter')
        where_clause = None
        if filter_metadata:
            filters = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    if len(value) > 1:
                        # If value is a list with multiple items, use $or operator
                        filters.append({"$or": [{key: v} for v in value]})
                    elif len(value) == 1:
                        # If value is a list with single item, use direct equality
                        filters.append({key: value[0]})
                else:
                    filters.append({key: value})
            
            if len(filters) > 1:
                where_clause = {"$and": filters}
            elif filters:
                where_clause = filters[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )

        documents = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                score = results['distances'][0][i]
                documents.append((Document(page_content=doc_content, metadata=metadata), score))

        return documents

    def count(self) -> int:
        """Return document count."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete the entire collection/index."""
        self.delete_collection()
        return True

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

        Query the vector store with multi-filter support and manual ordering.
        """
        # 1. Prepare Chroma-specific 'where' clause
        where_clause = None
        if filter_metadata:
            filters = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    if len(value) > 1:
                        # If value is a list with multiple items, use $or operator
                        filters.append({"$or": [{key: v} for v in value]})
                    elif len(value) == 1:
                        # If value is a list with single item, use direct equality
                        filters.append({key: value[0]})
                else:
                    filters.append({key: value})
            
            if len(filters) > 1:
                where_clause = {"$and": filters}
            elif filters:
                where_clause = filters[0]

        documents = []

        # 2. Fetch Data
        if query_text:
            # Semantic Search: Fetch more candidates if we need to sort manually
            fetch_limit = n_results * 10 if order_by else n_results
            query_embedding = self.embedding_function.embed_query(query_text)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_limit,
                where=where_clause,
                **kwargs
            )

            if results.get('documents'):
                for i in range(len(results['documents'][0])):
                    documents.append(Document(
                        page_content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ))
        else:
            # Metadata-only Fetch: Use .get() for efficiency
            # We don't limit here if order_by is present to ensure we get the absolute latest
            results = self.collection.get(
                where=where_clause,
                limit=n_results if not order_by else None,
                include=["documents", "metadatas"],
                **kwargs
            )

            if results.get('documents'):
                for i in range(len(results['documents'])):
                    documents.append(Document(
                        page_content=results['documents'][i],
                        metadata=results['metadatas'][i] if results['metadatas'] else {}
                    ))

        # 3. Manual Sorting & Final Slicing
        if order_by and documents:
            reverse = True if order.lower() == "desc" else False
            # Sort by the metadata key (handles strings, ints, or floats)
            documents.sort(key=lambda x: x.metadata.get(order_by, 0), reverse=reverse)
            # Return exactly n_results after sorting
            return documents[:n_results]

        return documents
