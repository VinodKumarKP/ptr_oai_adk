"""
Pydantic models for Knowledge Base Registry API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VectorDBType(str, Enum):
    CHROMA = "chroma"
    POSTGRES = "postgres"
    S3 = "s3"
    PINECONE = "pinecone"
    NEO4J = "neo4j_graph"   # Neo4j knowledge graph (GraphRAG) — always external, read-only


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
    # Pinecone (always external — no builtin container)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    pinecone_cloud: Optional[str] = "aws"
    pinecone_region: Optional[str] = "us-east-1"
    # Neo4j knowledge graph (always external, read-only — the graph is loaded
    # by an external pipeline; the registry only queries it).
    neo4j_url: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = "neo4j"
    # traversal (default) | text2cypher. NOTE: text2cypher and the
    # entity_linking entry strategy need an LLM, which the registry runtime does
    # not supply — registry-managed graphs should use traversal.
    neo4j_retrieval_mode: Optional[str] = "traversal"
    neo4j_entry_strategy: Optional[str] = "fulltext"   # fulltext | entity_linking | vector
    neo4j_fulltext_index: Optional[str] = None
    neo4j_max_hops: Optional[int] = 2
    neo4j_rel_limit: Optional[int] = 50
    # Hybrid (entry_strategy='vector'): a node-level vector index as a JSON
    # string, e.g. '{"type":"postgres","settings":{"collection_name":"kg_nodes",
    # "connection_string":"..."}}', plus the shared-id mapping.
    neo4j_vector_entry: Optional[str] = None
    neo4j_entry_id_field: Optional[str] = "node_id"
    neo4j_entry_id_property: Optional[str] = "id"
    # LLM for text2cypher / entity_linking. A LiteLLM model id only — the
    # registry reads credentials from its environment (like the embedding model).
    neo4j_llm_model_id: Optional[str] = None
    neo4j_llm_region: Optional[str] = None


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


class RetrievalConfig(BaseModel):
    """Default retrieval settings stored with the knowledge base."""
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve per query")
    score_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum similarity score (0-1). Results below this are excluded.",
    )


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
    # Retrieval defaults (stored and returned via /config endpoint for agent-core)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @model_validator(mode="after")
    def force_external_for_unmanaged_backends(self) -> "KBRegistration":
        """Pinecone and Neo4j have no builtin container — force external mode."""
        if self.vector_db_type in (VectorDBType.PINECONE, VectorDBType.NEO4J):
            self.deployment_mode = DeploymentMode.EXTERNAL
        return self

    @model_validator(mode="after")
    def neo4j_text2cypher_requires_llm(self) -> "KBRegistration":
        """``text2cypher`` generates Cypher with an LLM, so it needs a model id.

        Provide ``neo4j_llm_model_id`` (a LiteLLM model identifier); the registry
        reads the credentials from its environment. ``traversal`` with
        ``fulltext``/``vector`` entry needs no LLM; ``entity_linking`` optionally
        uses one and degrades to the raw query without it.
        """
        if self.vector_db_type == VectorDBType.NEO4J and self.vector_db_config:
            mode = (self.vector_db_config.neo4j_retrieval_mode or "traversal").lower()
            if mode == "text2cypher" and not self.vector_db_config.neo4j_llm_model_id:
                raise ValueError(
                    "retrieval_mode 'text2cypher' requires 'neo4j_llm_model_id' "
                    "(a LiteLLM model id). Credentials come from the registry's "
                    "environment."
                )
        return self


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


# ---------------------------------------------------------------------------
# Agent-core integration
# ---------------------------------------------------------------------------

class KBVectorStoreSettings(BaseModel):
    """Vector store connection settings in the format oai-agent-core expects."""
    collection_name: str
    # Chroma builtin
    persist_directory: Optional[str] = None
    # Chroma external
    host: Optional[str] = None
    port: Optional[int] = None
    ssl: Optional[bool] = None
    # Postgres
    connection_string: Optional[str] = None
    # S3
    bucket_name: Optional[str] = None
    prefix: Optional[str] = None
    region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    # Pinecone
    api_key: Optional[str] = None
    index_name: Optional[str] = None
    namespace: Optional[str] = None


# ---------------------------------------------------------------------------
# Data sources (external loader-based ingestion)
# ---------------------------------------------------------------------------

class LoaderFieldType(str, Enum):
    TEXT     = "text"
    URL      = "url"
    SECRET   = "secret"
    BOOLEAN  = "boolean"
    NUMBER   = "number"
    TEXTAREA = "textarea"
    SELECT   = "select"


class LoaderField(BaseModel):
    name:         str
    type:         LoaderFieldType
    label:        str
    required:     bool = False
    default:      Optional[Any] = None
    placeholder:  Optional[str] = None
    hint:         Optional[str] = None
    help:         Optional[str] = None
    env_var_hint: Optional[str] = None
    options:      Optional[List[Dict[str, str]]] = None  # for select


class LoaderCatalogEntry(BaseModel):
    id:           str
    display_name: str
    category:     str
    description:  str
    pip_extra:    Optional[str] = None
    fields:       List[LoaderField]


class SyncStatus(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    NEVER   = "never"


class DataSourceCreate(BaseModel):
    source_type:   str
    display_name:  str
    config:        Dict[str, Any]
    sync_schedule: Optional[str] = None   # "manual" or a cron expression


class DataSourceResponse(BaseModel):
    id:             str
    kb_name:        str
    source_type:    str
    display_name:   str
    config_public:  Dict[str, Any]        # secrets masked
    sync_schedule:  Optional[str]
    sync_status:    SyncStatus
    last_sync_at:   Optional[datetime]
    last_sync_error: Optional[str]
    document_count: int
    created_at:     datetime


class SyncRunResponse(BaseModel):
    id:             str
    source_id:      str
    started_at:     Optional[datetime]
    completed_at:   Optional[datetime]
    status:         SyncStatus
    document_count: int
    error_message:  Optional[str]


class TestConnectionRequest(BaseModel):
    source_type: str
    config:      Dict[str, Any]


class TestConnectionResponse(BaseModel):
    status:       str               # "ok" or "error"
    message:      Optional[str] = None
    sample_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Agent-core integration
# ---------------------------------------------------------------------------

class KBAgentConfig(BaseModel):
    """KB config in the format consumed by oai-agent-core BaseKnowledgeBaseFactory.

    Returned by ``GET /knowledge-bases/{kb_name}/config``.  An agent YAML can
    reference a registered KB by name instead of inlining the full config:

    .. code-block:: yaml

        knowledge_base:
          - registry_name: insurance_policies
            # optional local overrides:
            description: "Search insurance docs"
            retrieval_settings:
              top_k: 3
    """
    name: str
    description: str = ""
    vector_store: Dict[str, Any]
    embedding: Dict[str, Any]
    text_splitter: Dict[str, Any]
    retrieval_settings: Dict[str, Any]
    data_sources: List[Any] = Field(default_factory=list)
