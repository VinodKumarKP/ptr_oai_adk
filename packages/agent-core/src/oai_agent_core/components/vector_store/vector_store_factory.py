"""Factory for creating vector store instances."""

from typing import Optional, Any

from oai_agent_core.core.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Factory class to create vector store instances based on type."""

    @staticmethod
    def create_vector_store(
            vector_store_type: str,
            **kwargs: Any
    ) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            vector_store_type: Type of vector store ('chroma', 'postgres', 's3')
            **kwargs: Arguments to pass to the vector store constructor

        Returns:
            An instance of a class inheriting from BaseVectorStore

        Raises:
            ValueError: If the vector store type is unknown
        """
        vector_store_type = vector_store_type.lower()

        if vector_store_type == 'chroma':
            from oai_agent_core.components.vector_store.chroma_vector_store import ChromaVectorStore
            return ChromaVectorStore(**kwargs)
        elif vector_store_type == 'postgres':
            from oai_agent_core.components.vector_store.postgres_vector_store import PostgresVectorStore
            return PostgresVectorStore(**kwargs)
        elif vector_store_type == 's3':
            from oai_agent_core.components.vector_store.s3_vector_store import S3VectorStore
            return S3VectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown vector store type: {vector_store_type}")
