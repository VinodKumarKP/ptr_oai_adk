"""
Vector store provider factory for KB Registry.

Embeddings are created via LiteLLM, which provides a single unified
interface to every major embedding provider:

    bedrock/amazon.titan-embed-text-v1  (default)
    openai/text-embedding-3-small
    openai/text-embedding-3-large
    azure/<deployment-name>
    huggingface/sentence-transformers/all-MiniLM-L6-v2
    …

Credentials are read from environment variables by LiteLLM
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for Bedrock,
OPENAI_API_KEY for OpenAI, etc.) — they are never passed through
the API request body.

Supported vector DB types
-------------------------
chroma    — ChromaDB (HTTP or persistent local)
postgres  — PostgreSQL with pgvector extension
s3        — AWS S3 JSON index with in-memory cosine similarity
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from oai_agent_core.core.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Default model used when no model_id is specified.
DEFAULT_EMBEDDING_MODEL = "bedrock/amazon.titan-embed-text-v1"


# ---------------------------------------------------------------------------
# LiteLLM embedding wrapper  (mirrors base_knowledge_base_factory.py)
# ---------------------------------------------------------------------------

def _make_litellm_embeddings(
    model_id: Optional[str] = None,
    region_name: Optional[str] = None,
):
    """Return a LangChain-compatible Embeddings instance backed by LiteLLM.

    Args:
        model_id:    LiteLLM model identifier.  Defaults to
                     ``bedrock/amazon.titan-embed-text-v1``.
                     Plain Bedrock IDs (e.g. ``amazon.titan-embed-text-v1``)
                     are automatically prefixed with ``bedrock/``.
        region_name: AWS region for Bedrock models.  Falls back to the
                     ``AWS_DEFAULT_REGION`` environment variable when omitted.

    Returns:
        A ``LangChain Embeddings`` instance.

    Raises:
        RuntimeError: If ``litellm`` is not installed.
    """
    try:
        from litellm import embedding as litellm_embedding  # noqa: PLC0415
        from langchain_core.embeddings import Embeddings    # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "litellm and langchain-core are required for embeddings. "
            "Install them with: pip install litellm langchain-core"
        ) from exc

    # Resolve model ID — auto-prefix Bedrock model IDs that lack it.
    resolved_model = model_id or DEFAULT_EMBEDDING_MODEL
    if not resolved_model.startswith("bedrock/") and "amazon" in resolved_model:
        resolved_model = f"bedrock/{resolved_model}"

    logger.debug("Creating LiteLLM embeddings: model=%s region=%s", resolved_model, region_name)

    class LiteLLMEmbeddings(Embeddings):
        """LangChain Embeddings adapter for LiteLLM."""

        def __init__(self, _model_id: str, _region: Optional[str]) -> None:
            self.model_id    = _model_id
            self.region_name = _region

        def _extra_kwargs(self) -> dict:
            kwargs: dict = {}
            if self.region_name:
                kwargs["aws_region_name"] = self.region_name
            return kwargs

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            response = litellm_embedding(
                model=self.model_id,
                input=texts,
                **self._extra_kwargs(),
            )
            return [item["embedding"] for item in response["data"]]

        def embed_query(self, text: str) -> List[float]:
            response = litellm_embedding(
                model=self.model_id,
                input=[text],
                **self._extra_kwargs(),
            )
            return response["data"][0]["embedding"]

    return LiteLLMEmbeddings(resolved_model, region_name)


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _decrypt(value: Optional[str]) -> Optional[str]:
    """Base64-decode a stored config value."""
    if not value:
        return None
    try:
        return base64.b64decode(value.encode()).decode()
    except Exception:
        return value  # already plain text


def _encrypt(value: str) -> str:
    """Base64-encode a config value for storage."""
    return base64.b64encode(value.encode()).decode()


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

class VectorStoreProviderFactory:
    """Creates vector store instances from KB registration data."""

    @staticmethod
    def create_vector_store(
        kb_name: str,
        vector_db_type: str,
        deployment_mode: str,
        embedding_model_id: Optional[str] = None,
        embedding_region: Optional[str] = None,
        configs: Optional[Dict[str, str]] = None,
    ) -> "BaseVectorStore":
        """Build and return a fully configured vector store.

        Args:
            kb_name:           Used as the ``collection_name`` in the vector store.
            vector_db_type:    ``"chroma"`` | ``"postgres"`` | ``"s3"``
            deployment_mode:   ``"builtin"`` | ``"external"``
            embedding_model_id: LiteLLM model identifier.
                                Defaults to ``bedrock/amazon.titan-embed-text-v1``.
            embedding_region:  AWS region for Bedrock models (optional).
            configs:           Decrypted config key-value pairs from the DB
                               (connection details for external deployments).

        Returns:
            A ready-to-use ``BaseVectorStore`` instance.
        """
        from oai_agent_core.components.vector_store.vector_store_factory import (  # noqa: PLC0415
            VectorStoreFactory,
        )

        configs = configs or {}

        vdb = vector_db_type.lower()

        # ---- Pinecone — must be handled BEFORE embedding_fn is created ----
        # Pinecone uses integrated server-side inference; no local embeddings.
        if vdb == "pinecone":
            api_key    = configs.get("pinecone_api_key")
            index_name = configs.get("pinecone_index_name")
            namespace  = configs.get("pinecone_namespace") or kb_name
            if not api_key:
                raise ValueError(
                    "pinecone_api_key is required in vector_db_config for Pinecone"
                )
            if not index_name:
                raise ValueError(
                    "pinecone_index_name is required in vector_db_config for Pinecone"
                )
            return VectorStoreFactory.create_vector_store(
                "pinecone",
                collection_name=kb_name,
                api_key=api_key,
                index_name=index_name,
                namespace=namespace,
            )

        # For all other backends, create the LiteLLM embedding function first.
        embedding_fn = _make_litellm_embeddings(
            model_id=embedding_model_id,
            region_name=embedding_region,
        )

        # ---- Chroma -------------------------------------------------------
        if vdb == "chroma":
            kwargs: Dict[str, Any] = {
                "collection_name": kb_name,
                "embedding_function": embedding_fn,
            }
            if deployment_mode == "external":
                kwargs.update(
                    host=configs.get("chroma_host", "localhost"),
                    port=int(configs.get("chroma_port", "8000")),
                    ssl=configs.get("chroma_ssl", "false").lower() == "true",
                )
            else:
                data_dir = os.environ.get("KB_DATA_DIR", "/tmp/kb_data")  # noqa: S108
                kwargs["persist_directory"] = f"{data_dir}/chroma/{kb_name}"
            return VectorStoreFactory.create_vector_store("chroma", **kwargs)

        # ---- Postgres (pgvector) ------------------------------------------
        if vdb == "postgres":
            if deployment_mode == "external":
                # psycopg2.connect() accepts plain postgresql:// URLs (no +psycopg2 dialect)
                # or key=value DSN strings.  Build from stored config fields.
                pg_user     = configs.get("pg_user", "postgres")
                pg_password = configs.get("pg_password", "")
                pg_host     = configs.get("pg_host", "localhost")
                pg_port     = configs.get("pg_port", "5432")
                pg_database = configs.get("pg_database", "kb_vectors")
                connection_string = (
                    f"host={pg_host} port={pg_port} "
                    f"dbname={pg_database} user={pg_user} password={pg_password}"
                )
            else:
                # Builtin pgvector container — use env var if set, else default DSN.
                # Accepts either a plain postgresql:// URL or key=value DSN.
                raw_url = os.environ.get("KB_POSTGRES_URL", "")
                if raw_url:
                    # Strip SQLAlchemy dialect prefix if someone set it that way
                    connection_string = raw_url.replace("postgresql+psycopg2://", "postgresql://")
                else:
                    pg_host = os.environ.get("KB_PGVECTOR_HOST", "localhost")
                    pg_port = os.environ.get("KB_PGVECTOR_PORT", "5435")
                    pg_user = os.environ.get("KB_PGVECTOR_USER", "postgres")
                    pg_password = os.environ.get("KB_PGVECTOR_PASSWORD", "postgres")
                    pg_database = os.environ.get("KB_PGVECTOR_DB", "kb_vectors")
                    connection_string = (
                        f"host={pg_host} port={pg_port} "
                        f"dbname={pg_database} user={pg_user} password={pg_password}"
                    )
            return VectorStoreFactory.create_vector_store(
                "postgres",
                collection_name=kb_name,
                embedding_function=embedding_fn,
                connection_string=connection_string,
            )

        # ---- S3 vector store ----------------------------------------------
        if vdb == "s3":
            kwargs = {
                "collection_name": kb_name,
                "embedding_function": embedding_fn,
                "prefix": configs.get("s3_prefix", "kb_vector_store"),
                "region": configs.get("s3_region", "us-east-1"),
            }
            if deployment_mode == "external":
                kwargs["bucket_name"] = configs.get("s3_bucket")
                if configs.get("aws_access_key_id"):
                    kwargs["aws_access_key_id"]    = configs["aws_access_key_id"]
                    kwargs["aws_secret_access_key"] = configs.get("aws_secret_access_key", "")
            else:
                kwargs["bucket_name"] = os.environ.get("KB_S3_BUCKET")
            return VectorStoreFactory.create_vector_store("s3", **kwargs)

        raise ValueError(f"Unsupported vector_db_type: {vector_db_type!r}")

    @staticmethod
    def encrypt_configs(vector_db_config: Dict[str, Any]) -> Dict[str, str]:
        """Return a flat ``{key: encrypted_value}`` dict for DB storage."""
        return {k: _encrypt(str(v)) for k, v in vector_db_config.items() if v is not None}

    @staticmethod
    def decrypt_configs(raw: Dict[str, str]) -> Dict[str, str]:
        """Decrypt a flat ``{key: encrypted_value}`` dict loaded from the DB."""
        return {k: (_decrypt(v) or "") for k, v in raw.items()}
