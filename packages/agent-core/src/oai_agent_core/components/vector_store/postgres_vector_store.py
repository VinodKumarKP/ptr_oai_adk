"""Postgres vector store implementation using pgvector directly."""

import json
import logging
import os
from typing import List, Optional, Any, Iterable, Tuple
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST

try:
    import psycopg2
    from psycopg2.extras import execute_values
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None
    execute_values = None
    register_vector = None

from oai_agent_core.core.base_vector_store import BaseVectorStore


class PostgresVectorStore(BaseVectorStore):
    """Postgres implementation of the vector store interface using pgvector directly."""

    def __init__(self, **kwargs):
        """Initialize Postgres vector store.

        Args:
            **kwargs: Arguments including collection_name, embedding_function, connection_string, vector_dimensions
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        if psycopg2 is None:
            raise ImportError(
                "Could not import psycopg2 or pgvector. "
                "Please install them with `pip install psycopg2-binary pgvector`."
            )

        # Validate required arguments
        if not self.collection_name:
            raise ValueError("collection_name is required")
        if not self.embedding_function:
            raise ValueError("embedding_function is required")

        # Get or build connection string
        self.connection_string = kwargs.get('connection_string')

        if not self.connection_string:
            # Fallback to environment variables if not provided
            db_name = kwargs.get('db_name') or os.environ.get('DB_NAME')
            db_host = kwargs.get("db_host") or os.environ.get(f"DB_HOST_{db_name.upper()}") or os.environ.get(
                'DB_HOST', 'localhost')
            db_port = kwargs.get("db_port") or os.environ.get(f"DB_PORT_{db_name.upper()}") or os.environ.get(
                'DB_PORT', '5432')
            db_user = kwargs.get("db_user") or os.environ.get(f"DB_USER_{db_name.upper()}") or os.environ.get(
                'DB_USER', 'postgres')
            db_password = kwargs.get("db_password") or os.environ.get(f"DB_PASSWORD_{db_name.upper()}") or os.environ.get('DB_PASSWORD',
                                                                                                  'postgres')
            if not db_name or not db_host or not db_port or not db_user or not db_password:
                raise ValueError(
                    "Missing database connection parameters. Please provide connection_string or set DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD environment variables.")

            self.connection_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"

        # Vector dimensions (will be auto-detected from first embedding)
        self.vector_dimensions = kwargs.get('vector_dimensions')

        # Table name based on collection
        self.table_name = f"embeddings_{self.collection_name}"

        # Initialize connection and create tables
        try:
            self._initialize_database()
            self.logger.info(f"Successfully initialized PostgresVectorStore with collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Postgres vector store: {str(e)}")
            raise ValueError(f"Failed to initialize Postgres vector store: {str(e)}")

    def _get_connection(self):
        """Get a database connection."""
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        return conn

    def _initialize_database(self):
        """Initialize database: create extension and tables."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Create pgvector extension if not exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()

                # Detect vector dimensions if not provided
                if self.vector_dimensions is None:
                    self.logger.info("Auto-detecting vector dimensions from embedding function...")
                    test_embedding = self.embedding_function.embed_query("test")
                    self.vector_dimensions = len(test_embedding)
                    self.logger.info(f"Detected vector dimensions: {self.vector_dimensions}")

                # Create table for embeddings
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({self.vector_dimensions})
                )
                """
                cur.execute(create_table_query)

                # Create index for vector similarity search (using HNSW or IVFFlat)
                # HNSW is generally better for most use cases
                index_name = f"{self.table_name}_embedding_idx"
                try:
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.table_name} 
                        USING hnsw (embedding vector_cosine_ops)
                    """)
                except Exception as e:
                    # If HNSW fails, try IVFFlat
                    self.logger.warning(f"HNSW index creation failed, trying IVFFlat: {e}")
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)

                conn.commit()
                self.logger.debug(f"Database tables and indexes created for {self.table_name}")
        finally:
            conn.close()

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
    ) -> list[Any] | None:
        """Add texts to Postgres.
        
        Args:
            texts: Iterable of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the texts
            **kwargs: Additional arguments
            
        Returns:
            List of IDs of added texts
        """
        if not texts:
            return []

        texts_list = list(texts)
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts_list))]

        if metadatas is None:
            metadatas = [{} for _ in range(len(texts_list))]

        # Generate embeddings
        try:
            self.logger.debug(f"Generating embeddings for {len(texts_list)} texts...")
            embeddings = self.embedding_function.embed_documents(texts_list)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

        # Insert into database
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data for insertion
                data = [
                    (ids[i], texts_list[i], json.dumps(metadatas[i]), embeddings[i])
                    for i in range(len(texts_list))
                ]

                # Use execute_values for efficient batch insert
                insert_query = f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                """
                execute_values(cur, insert_query, data)
                conn.commit()

                self.logger.debug(f"Successfully added {len(texts_list)} texts to {self.table_name}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error adding texts to Postgres: {str(e)}")
            raise
        finally:
            conn.close()

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Add documents to Postgres.
        
        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for the documents
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.add_texts(texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Run similarity search in Postgres.
        
        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search arguments (filter for metadata filtering)
            
        Returns:
            List of matching Document objects
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_function.embed_query(query)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

        # Build query
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Base query
                sql_query = f"""
                    SELECT id, content, metadata, 
                           embedding <=> %s::vector as distance
                    FROM {self.table_name}
                """

                params = [query_embedding]

                # Add metadata filter if provided
                filter_metadata = kwargs.get('filter')
                if filter_metadata:
                    conditions = []
                    for key, value in filter_metadata.items():
                        if isinstance(value, list):
                            # Handle list values with OR logic (IN clause)
                            placeholders = ','.join(['%s'] * len(value))
                            conditions.append(f"metadata->>%s IN ({placeholders})")
                            params.append(key)
                            params.extend([str(v) for v in value])
                        else:
                            conditions.append("metadata->>%s = %s")
                            params.extend([key, str(value)])

                    if conditions:
                        sql_query += " WHERE " + " AND ".join(conditions)

                sql_query += f" ORDER BY distance LIMIT {k}"

                cur.execute(sql_query, params)
                results = cur.fetchall()

                # Convert to Document objects
                documents = []
                for row in results:
                    doc = Document(
                        page_content=row[1],
                        metadata=row[2] if row[2] else {}
                    )
                    documents.append(doc)

                return documents
        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            raise
        finally:
            conn.close()

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Run similarity search in Postgres with score.
        
        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search arguments
            
        Returns:
            List of tuples containing Document objects and their similarity scores
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_function.embed_query(query)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Base query
                sql_query = f"""
                    SELECT id, content, metadata, 
                           embedding <=> %s::vector as distance
                    FROM {self.table_name}
                """

                params = [query_embedding]

                # Add metadata filter if provided
                filter_metadata = kwargs.get('filter')
                if filter_metadata:
                    conditions = []
                    for key, value in filter_metadata.items():
                        if isinstance(value, list):
                            # Handle list values with OR logic (IN clause)
                            placeholders = ','.join(['%s'] * len(value))
                            conditions.append(f"metadata->>%s IN ({placeholders})")
                            params.append(key)
                            params.extend([str(v) for v in value])
                        else:
                            conditions.append("metadata->>%s = %s")
                            params.extend([key, str(value)])

                    if conditions:
                        sql_query += " WHERE " + " AND ".join(conditions)

                sql_query += f" ORDER BY distance LIMIT {k}"

                cur.execute(sql_query, params)
                results = cur.fetchall()

                # Convert to Document objects with scores
                documents_with_scores = []
                for row in results:
                    doc = Document(
                        page_content=row[1],
                        metadata=row[2] if row[2] else {}
                    )
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = float(row[3])
                    documents_with_scores.append((doc, score))

                return documents_with_scores
        except Exception as e:
            self.logger.error(f"Error during similarity search with score: {str(e)}")
            raise
        finally:
            conn.close()

    def count(self) -> int:
        """Return document count.
        
        Returns:
            Number of documents in the collection
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cur.fetchone()[0]
                return count
        except Exception as e:
            self.logger.error(f"Error getting document count: {str(e)}")
            return 0
        finally:
            conn.close()

    def delete_collection(self) -> None:
        """Delete the collection (drop table)."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE")
                conn.commit()
                self.logger.info(f"Successfully deleted collection: {self.collection_name}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error deleting collection: {str(e)}")
            raise
        finally:
            conn.close()

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            True if deletion was successful, False/None otherwise
        """
        if not ids:
            self.logger.warning("No IDs provided for deletion")
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Delete documents with given IDs
                delete_query = f"DELETE FROM {self.table_name} WHERE id = ANY(%s)"
                cur.execute(delete_query, (ids,))
                deleted_count = cur.rowcount
                conn.commit()

                self.logger.debug(f"Successfully deleted {deleted_count} documents")
                return True
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error deleting documents: {str(e)}")
            return False
        finally:
            conn.close()

    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        try:
            self.delete_collection()
            self.logger.info(f"Deleted collection: {self.collection_name}")

            # Recreate tables
            self._initialize_database()
            self.logger.info(f"Recreated collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error resetting collection: {str(e)}")
            raise

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
            query_text: Optional query string for similarity search
            filter_metadata: Optional metadata filters
            n_results: Number of results to return
            order_by: Optional field name to order by
            order: Sort order ('asc' or 'desc')
            **kwargs: Additional query arguments
            
        Returns:
            List of matching Document objects
        """
        try:
            if query_text:
                # Perform similarity search with optional metadata filter
                docs = self.similarity_search(
                    query=query_text,
                    k=n_results,
                    filter=filter_metadata,
                    **kwargs
                )
            elif filter_metadata:
                # If no query text but filter provided, get documents matching filter
                self.logger.warning(
                    "Querying with filter_metadata but no query_text - returning filtered results without ranking")
                docs = self._get_by_metadata(filter_metadata, n_results)
            else:
                # No query text and no filter - return empty list
                self.logger.warning("Query called without query_text or filter_metadata")
                docs = []

            # Apply manual ordering if specified
            if order_by and docs:
                reverse = True if order.lower() == "desc" else False
                try:
                    docs.sort(key=lambda x: x.metadata.get(order_by, 0), reverse=reverse)
                    docs = docs[:n_results]
                except Exception as e:
                    self.logger.warning(f"Error sorting by {order_by}: {str(e)}")

            return docs
        except Exception as e:
            self.logger.error(f"Error during query: {str(e)}")
            raise

    def _get_by_metadata(self, filter_metadata: dict, limit: int = 10) -> List[Document]:
        """Get documents by metadata filter without vector search.
        
        Args:
            filter_metadata: Metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching Document objects
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Build query with metadata filters
                sql_query = f"SELECT id, content, metadata FROM {self.table_name}"

                conditions = []
                params = []
                for key, value in filter_metadata.items():
                    if isinstance(value, list):
                        # Handle list values with OR logic (IN clause)
                        placeholders = ','.join(['%s'] * len(value))
                        conditions.append(f"metadata->>%s IN ({placeholders})")
                        params.append(key)
                        params.extend([str(v) for v in value])
                    else:
                        conditions.append("metadata->>%s = %s")
                        params.extend([key, str(value)])

                if conditions:
                    sql_query += " WHERE " + " AND ".join(conditions)

                sql_query += f" LIMIT {limit}"

                cur.execute(sql_query, params)
                results = cur.fetchall()

                # Convert to Document objects
                documents = []
                for row in results:
                    doc = Document(
                        page_content=row[1],
                        metadata=row[2] if row[2] else {}
                    )
                    documents.append(doc)

                return documents
        except Exception as e:
            self.logger.error(f"Error getting documents by metadata: {str(e)}")
            return []
        finally:
            conn.close()

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
        """Create a new PostgresVectorStore from texts.
        
        Args:
            texts: List of text strings
            embedding: Embedding function
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            **kwargs: Additional arguments including collection_name
            
        Returns:
            New PostgresVectorStore instance
        """
        store = cls(embedding_function=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
