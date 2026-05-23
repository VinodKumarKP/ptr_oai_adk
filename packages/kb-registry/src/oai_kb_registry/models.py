"""
Pydantic models for Knowledge Base Registry API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VectorDBType(str, Enum):
    CHROMA = "chroma"
    POSTGRES = "postgres"
    S3 = "s3"


class DeploymentMode(str, Enum):
    BUILTIN = "builtin"    # Platform provisions the vector DB automatically
    EXTERNAL = "external"  # User supplies their own connection details


class DocumentSourceType(str, Enum):
    UPLOAD = "upload"      # File uploaded via the API
    S3 = "s3"              # Reference to an object already in S3
    URL = "url"            # Publicly accessible URL


class KBStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROVISIONING = "provisioning"
    ERROR = "error"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Knowledge Base registration
# ---------------------------------------------------------------------------

class VectorDBConfig(BaseModel):
    """Provider-specific connection details for EXTERNAL deployment mode."""
    # Chroma
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = 8000
    chroma_ssl: bool = False
    # Postgres (pgvector)
    pg_host: Optional[str] = None
    pg_port: Optional[int] = 5432
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_database: Optional[str] = None
    # S3 Vector
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = "kb_vector_store"
    s3_region: Optional[str] = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class EmbeddingConfig(BaseModel):
    """Embedding model configuration.

    Uses LiteLLM under the hood — ``model_id`` is any LiteLLM-compatible
    model identifier, e.g.:

    * ``bedrock/amazon.titan-embed-text-v1``   (default — AWS Bedrock)
    * ``openai/text-embedding-3-small``
    * ``openai/text-embedding-3-large``
    * ``huggingface/sentence-transformers/all-MiniLM-L6-v2``
    * ``azure/my-ada-deployment``

    Credentials are read from environment variables by LiteLLM
    (``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` for Bedrock,
    ``OPENAI_API_KEY`` for OpenAI, etc.).
    """
    model_id: str = "bedrock/amazon.titan-embed-text-v1"
    region_name: Optional[str] = Field(
        default=None,
        description="AWS region for Bedrock models (e.g. 'us-east-1'). "
                    "Falls back to AWS_DEFAULT_REGION env var when omitted.",
    )


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)
    separators: Optional[List[str]] = None


class KBRegistration(BaseModel):
    """Request body for registering a new knowledge base."""
    name: str = Field(..., min_length=1, max_length=255,
                      description="Unique name for this knowledge base")
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    # Vector store settings
    vector_db_type: VectorDBType = VectorDBType.CHROMA
    deployment_mode: DeploymentMode = DeploymentMode.BUILTIN
    vector_db_config: Optional[VectorDBConfig] = None
    # Embedding settings
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # Chunking settings
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class KBDetails(BaseModel):
    """Full knowledge base record returned by the API."""
    id: int
    name: str
    description: Optional[str]
    tags: List[str]
    vector_db_type: VectorDBType
    deployment_mode: DeploymentMode
    embedding_model_id: str
    embedding_region: Optional[str]
    chunk_size: int
    chunk_overlap: int
    status: KBStatus
    document_count: int = 0
    total_chunks: int = 0
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

class S3DocumentSource(BaseModel):
    """Reference to an S3 object to be indexed."""
    bucket: str
    key: str
    region: str = "us-east-1"
    # Optional: provide credentials if different from KB-level AWS config
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


class URLDocumentSource(BaseModel):
    """Public URL to be fetched and indexed."""
    url: str
    mime_type: Optional[str] = None


class DocumentDetails(BaseModel):
    """Document record returned by the API."""
    id: int
    kb_id: int
    name: str
    source_type: DocumentSourceType
    source_uri: Optional[str]
    file_size_bytes: Optional[int]
    chunk_count: int = 0
    status: DocumentStatus
    error_message: Optional[str]
    indexed_at: Optional[datetime]
    created_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Query / search
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum similarity score (0-1). Results below this are excluded."
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value metadata filters applied before similarity ranking"
    )
    include_metadata: bool = True


class QueryResult(BaseModel):
    """Single result from a semantic search."""
    content: str
    score: Optional[float]
    metadata: Optional[Dict[str, Any]]
    document_name: Optional[str]
    document_id: Optional[int]


class QueryResponse(BaseModel):
    """Response from a semantic search."""
    kb_name: str
    query: str
    results: List[QueryResult]
    total_results: int


# ---------------------------------------------------------------------------
# Action history
# ---------------------------------------------------------------------------

class KBAction(BaseModel):
    """Knowledge base action log entry."""
    id: int
    kb_id: int
    action: str
    performed_by: Optional[str]
    message: Optional[str]
    created_at: Optional[datetime]


# ---------------------------------------------------------------------------
# Misc response models
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None


class ReindexResponse(BaseModel):
    status: str
    kb_name: str
    documents_queued: int
    message: Optional[str] = None
