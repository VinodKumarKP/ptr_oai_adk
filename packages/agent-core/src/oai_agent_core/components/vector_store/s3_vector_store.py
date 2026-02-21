"""S3-based vector store implementation."""

import json
import logging
import os
from typing import List, Optional, Any, Iterable, Tuple, TypeVar
from uuid import uuid4

import boto3
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from oai_agent_core.core.base_vector_store import BaseVectorStore

VST = TypeVar("VST", bound="S3VectorStore")


class S3VectorStore(BaseVectorStore):
    """
    S3-based vector store implementation.
    This is a simple implementation that stores vectors and metadata in S3.
    It performs brute-force similarity search (cosine similarity) in memory after downloading the index.
    Suitable for small to medium datasets where a dedicated vector DB is overkill.
    """

    def __init__(self, **kwargs):
        """Initialize S3 vector store.

        Args:
            **kwargs: Arguments including collection_name, embedding_function, bucket_name, prefix
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        # Validate required arguments
        if not self.collection_name:
            raise ValueError("collection_name is required")
        if not self.embedding_function:
            raise ValueError("embedding_function is required")

        self.bucket_name = kwargs.get('bucket_name')
        self.region = kwargs.get('region', 'us-east-1')
        if not self.bucket_name:
            self.bucket_name = os.environ.get(f"{self.collection_name.upper()}_VECTOR_BUCKET") or os.environ.get(
                'S3_VECTOR_BUCKET')
            if not self.bucket_name:
                raise ValueError("bucket_name is required or S3_VECTOR_BUCKET env var must be set")

        self.prefix = kwargs.get('prefix', 'vector_store')
        self.index_key = f"{self.prefix}/{self.collection_name}/index.json"

        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=self.region)

        # In-memory storage
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []
        self.ids: List[str] = []

        self._load_index()

    def _load_index(self):
        """Load index from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.index_key)
            data = json.loads(response['Body'].read().decode('utf-8'))

            self.ids = data.get('ids', [])
            self.vectors = data.get('vectors', [])

            doc_dicts = data.get('documents', [])
            self.documents = [
                Document(page_content=d['page_content'], metadata=d['metadata'])
                for d in doc_dicts
            ]
            self.logger.info(f"Loaded {len(self.documents)} documents from S3: {self.bucket_name}/{self.index_key}")
        except self.s3_client.exceptions.NoSuchKey:
            self.logger.info(f"No existing index found at {self.bucket_name}/{self.index_key}. Starting fresh.")
        except Exception as e:
            self.logger.warning(f"Failed to load index from S3: {e}. Starting fresh.")

    def _save_index(self):
        """Save index to S3."""
        data = {
            'ids': self.ids,
            'vectors': self.vectors,
            'documents': [
                {'page_content': doc.page_content, 'metadata': doc.metadata}
                for doc in self.documents
            ]
        }

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.index_key,
                Body=json.dumps(data)
            )
            self.logger.info(f"Saved {len(self.documents)} documents to S3: {self.bucket_name}/{self.index_key}")
        except Exception as e:
            self.logger.error(f"Failed to save index to S3: {e}")
            raise

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> list[str]:
        """Add texts to S3 vector store.
        
        Args:
            texts: Iterable of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            **kwargs: Additional arguments
            
        Returns:
            List of IDs for the added texts
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts_list))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts_list))]

        embeddings = self.embedding_function.embed_documents(texts_list)

        self.ids.extend(ids)
        self.vectors.extend(embeddings)
        for text, metadata in zip(texts_list, metadatas):
            self.documents.append(Document(page_content=text, metadata=metadata))

        self._save_index()
        return ids

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Add documents to S3 vector store.
        
        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.add_texts(texts, metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Run similarity search.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of matching Document objects
        """
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[Tuple[Document, float]]:
        """Run similarity search with score.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing Document objects and their similarity scores
        """
        if not self.vectors:
            return []

        filter_metadata = kwargs.get('filter')
        
        # Filter indices first if metadata filter is provided
        valid_indices = []
        if filter_metadata:
            for i, doc in enumerate(self.documents):
                match = True
                for key, value in filter_metadata.items():
                    if isinstance(value, list):
                        if doc.metadata.get(key) not in value:
                            match = False
                            break
                    elif doc.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
        else:
            valid_indices = list(range(len(self.documents)))

        if not valid_indices:
            return []

        query_embedding = self.embedding_function.embed_query(query)

        # Calculate cosine similarity
        # Note: This is a simple in-memory implementation. 
        # For large datasets, use a proper vector DB or optimized library like Faiss.

        query_vec = np.array(query_embedding)
        
        # Only use vectors corresponding to valid indices
        doc_vecs = np.array([self.vectors[i] for i in valid_indices])

        # Normalize vectors for cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_docs = np.linalg.norm(doc_vecs, axis=1)

        # Avoid division by zero
        if norm_query == 0:
            return []

        # Cosine similarity = (A . B) / (||A|| * ||B||)
        dot_products = np.dot(doc_vecs, query_vec)

        # Handle zero norm docs
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dot_products / (norm_docs * norm_query)
            similarities = np.nan_to_num(similarities)  # Replace NaNs with 0

        # Get top k indices
        # argsort sorts in ascending order, so we take the last k
        # We need to map back to original indices if we filtered
        
        # If we have fewer results than k, return all of them
        k = min(k, len(similarities))
        
        top_k_local_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for local_idx in top_k_local_indices:
            original_idx = valid_indices[local_idx]
            # Convert similarity to distance-like score if needed, or just return similarity
            # LangChain usually expects distance (lower is better) or similarity (higher is better) depending on store
            # Here we return similarity score
            score = float(similarities[local_idx])
            results.append((self.documents[original_idx], score))

        return results

    def count(self) -> int:
        """Return document count.
        
        Returns:
            Number of documents in the store
        """
        return len(self.documents)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID or delete entire collection if no IDs provided.
        
        Args:
            ids: Optional list of document IDs to delete. If None, deletes entire collection.
            **kwargs: Additional arguments
            
        Returns:
            True if documents were deleted, False otherwise
        """
        if ids is None:
            # Delete entire collection
            self.reset_collection()
            return True

        if not ids:
            return False

        new_ids = []
        new_vectors = []
        new_documents = []

        deleted = False
        for i, doc_id in enumerate(self.ids):
            if doc_id not in ids:
                new_ids.append(doc_id)
                new_vectors.append(self.vectors[i])
                new_documents.append(self.documents[i])
            else:
                deleted = True

        if deleted:
            self.ids = new_ids
            self.vectors = new_vectors
            self.documents = new_documents
            self._save_index()
            return True

        return False

    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=self.index_key)
            self.logger.info(f"Deleted index at {self.bucket_name}/{self.index_key}")
        except Exception as e:
            self.logger.warning(f"Failed to delete collection from S3: {e}")
        finally:
            # Reset in-memory storage regardless of S3 deletion success
            self.ids = []
            self.vectors = []
            self.documents = []

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
        
        Args:
            query_text: Optional text query for similarity search
            filter_metadata: Optional metadata filters to apply
            n_results: Number of results to return
            order_by: Optional metadata field to order results by
            order: Sort order ("desc" or "asc")
            **kwargs: Additional arguments
            
        Returns:
            List of matching Document objects
        """
        # 1. Filter candidates
        candidates = []
        candidate_indices = []

        for i, doc in enumerate(self.documents):
            match = True
            if filter_metadata:
                for k, v in filter_metadata.items():
                    if isinstance(v, list):
                        if doc.metadata.get(k) not in v:
                            match = False
                            break
                    elif doc.metadata.get(k) != v:
                        match = False
                        break

            if match:
                candidates.append(doc)
                candidate_indices.append(i)

        # 2. Rank by similarity if query_text is present
        results = []
        if query_text and candidates:
            query_embedding = self.embedding_function.embed_query(query_text)
            query_vec = np.array(query_embedding)

            # Filter vectors corresponding to candidates
            candidate_vectors = np.array([self.vectors[i] for i in candidate_indices])

            norm_query = np.linalg.norm(query_vec)
            norm_docs = np.linalg.norm(candidate_vectors, axis=1)

            if norm_query > 0:
                dot_products = np.dot(candidate_vectors, query_vec)

                with np.errstate(divide='ignore', invalid='ignore'):
                    similarities = dot_products / (norm_docs * norm_query)
                    similarities = np.nan_to_num(similarities)

                # Sort by similarity
                sorted_indices = np.argsort(similarities)[::-1]

                for idx in sorted_indices:
                    results.append(candidates[idx])
            else:
                results = candidates
        else:
            results = candidates

        # 3. Manual Sorting & Final Slicing
        if order_by and results:
            reverse = True if order.lower() == "desc" else False
            results.sort(key=lambda x: x.metadata.get(order_by, 0), reverse=reverse)

        return results[:n_results]

    @classmethod
    def from_texts(
            cls: type[VST],
            texts: list[str],
            embedding: Embeddings,
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any
    ) -> VST:
        """Create a new S3VectorStore from texts.
        
        Args:
            texts: List of text strings
            embedding: Embeddings function to use
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            **kwargs: Additional arguments for store initialization
            
        Returns:
            New S3VectorStore instance
        """
        store = cls(embedding_function=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
