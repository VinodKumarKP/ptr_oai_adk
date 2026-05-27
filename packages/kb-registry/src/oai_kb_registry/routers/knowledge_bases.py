"""
Knowledge base CRUD routes.

  GET  /                                     — API overview
  GET  /health                               — Health check
  GET  /knowledge-bases                      — List all knowledge bases
  GET  /knowledge-bases/{kb_name}            — Get a single KB (with doc counts)
  GET  /knowledge-bases/{kb_name}/config     — Agent-core compatible config for KB references
  POST /knowledge-bases                      — Register a new KB
  DELETE /knowledge-bases/{kb_name}          — Delete KB + all documents
  GET  /knowledge-bases/{kb_name}/history    — Audit log
"""

import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from oai_kb_registry.dependencies import get_registry, verify_bearer_token, get_auth_user
from oai_kb_registry.models import KBAgentConfig, KBDetails, KBRegistration, KBStatus, StatusResponse
from oai_kb_registry.services.kb_registry import KBRegistry
from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def root():
    """API overview — all available endpoints."""
    return JSONResponse({
        "message": "Knowledge Base Registry API",
        "description": (
            "Manage vector-backed knowledge bases with document ingestion, "
            "embedding, and semantic search."
        ),
        "endpoints": {
            "GET  /health":                                   "Health check",
            "GET  /knowledge-bases":                          "List all registered knowledge bases",
            "GET  /knowledge-bases/{kb_name}":                "Get KB details with document counts",
            "GET  /knowledge-bases/{kb_name}/config":         "Agent-core compatible config (for registry_name references in agent YAML)",
            "POST /knowledge-bases":                          "Register a new knowledge base",
            "DELETE /knowledge-bases/{kb_name}":              "Delete a knowledge base",
            "GET  /knowledge-bases/{kb_name}/history":        "Audit log for a knowledge base",
            "POST /knowledge-bases/{kb_name}/documents/upload": "Upload a document (PDF, DOCX, TXT, MD)",
            "POST /knowledge-bases/{kb_name}/documents/s3":   "Index an S3 object",
            "GET  /knowledge-bases/{kb_name}/documents":      "List documents in a KB",
            "DELETE /knowledge-bases/{kb_name}/documents/{doc_id}": "Remove a document",
            "POST /knowledge-bases/{kb_name}/query":          "Semantic search",
            "POST /knowledge-bases/{kb_name}/reindex":        "Reindex all documents",
            "POST   /tokens/generate":                        "Generate an API token",
            "GET    /tokens":                                 "List active tokens",
            "DELETE /tokens/{token}":                         "Revoke a token",
        },
    })


@router.get("/health")
async def health_check():
    return {"status": "ok", "registry": "kb"}


# ---------------------------------------------------------------------------
# List / get
# ---------------------------------------------------------------------------

@router.get("/knowledge-bases", response_model=List[Dict])
async def list_knowledge_bases(
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """List all registered knowledge bases."""
    return await registry.db.get_all_knowledge_bases()


@router.get("/knowledge-bases/{kb_name}")
async def get_knowledge_base(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Return full details for a single knowledge base."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_name}")
    return kb


# ---------------------------------------------------------------------------
# Agent-core config endpoint
# ---------------------------------------------------------------------------

@router.get("/knowledge-bases/{kb_name}/config", response_model=KBAgentConfig)
async def get_knowledge_base_config(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Return the KB config in the format consumed by oai-agent-core BaseKnowledgeBaseFactory.

    Agents can reference a registered KB by name in their YAML instead of
    inlining the full vector-store configuration::

        knowledge_base:
          - registry_name: insurance_policies
            description: "Search insurance policy docs"   # optional override
            retrieval_settings:                           # optional override
              top_k: 3
    """
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_name}")

    # Decrypt stored connection details
    raw_configs = await registry.db.get_kb_configs(kb["id"])
    configs = VectorStoreProviderFactory.decrypt_configs(raw_configs)

    vdb = kb["vector_db_type"].lower()
    deployment = (kb.get("deployment_mode") or "builtin").lower()

    # ------------------------------------------------------------------
    # Build vector_store.settings in the format VectorStoreFactory expects
    # ------------------------------------------------------------------
    if vdb == "chroma":
        if deployment == "builtin":
            data_dir = os.environ.get("KB_DATA_DIR", "/tmp/kb_data")
            settings: Dict[str, Any] = {
                "collection_name": kb_name,
                "persist_directory": f"{data_dir}/chroma/{kb_name}",
            }
        else:
            settings = {
                "collection_name": kb_name,
                "host": configs.get("chroma_host", "localhost"),
                "port": int(configs.get("chroma_port") or "8000"),
                "ssl": (configs.get("chroma_ssl") or "false").lower() == "true",
            }

    elif vdb == "postgres":
        if deployment == "builtin":
            pg_host = os.environ.get("KB_PGVECTOR_HOST", "localhost")
            pg_port = os.environ.get("KB_PGVECTOR_PORT", "5435")
            pg_user = os.environ.get("KB_PGVECTOR_USER", "postgres")
            pg_password = os.environ.get("KB_PGVECTOR_PASSWORD", "postgres")
            pg_database = os.environ.get("KB_PGVECTOR_DB", "kb_vectors")
        else:
            pg_host = configs.get("pg_host", "localhost")
            pg_port = configs.get("pg_port", "5432")
            pg_user = configs.get("pg_user", "postgres")
            pg_password = configs.get("pg_password", "")
            pg_database = configs.get("pg_database", "kb_vectors")
        settings = {
            "collection_name": kb_name,
            "connection_string": (
                f"host={pg_host} port={pg_port} "
                f"dbname={pg_database} user={pg_user} password={pg_password}"
            ),
        }

    elif vdb == "s3":
        settings = {
            "collection_name": kb_name,
            "bucket_name": configs.get("s3_bucket") or os.environ.get("KB_S3_BUCKET", ""),
            "prefix": configs.get("s3_prefix", "kb_vector_store"),
            "region": configs.get("s3_region", "us-east-1"),
        }
        if configs.get("aws_access_key_id"):
            settings["aws_access_key_id"] = configs["aws_access_key_id"]
            settings["aws_secret_access_key"] = configs.get("aws_secret_access_key", "")

    elif vdb == "pinecone":
        settings = {
            "collection_name": kb_name,
            "api_key": configs.get("pinecone_api_key"),
            "index_name": configs.get("pinecone_index_name"),
            "namespace": configs.get("pinecone_namespace") or kb_name,
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported vector_db_type: {vdb}")

    # ------------------------------------------------------------------
    # Parse retrieval config (stored as JSON string)
    # ------------------------------------------------------------------
    retrieval_raw = kb.get("retrieval_config") or "{}"
    if isinstance(retrieval_raw, str):
        try:
            retrieval_raw = json.loads(retrieval_raw)
        except json.JSONDecodeError:
            retrieval_raw = {}
    retrieval = retrieval_raw if isinstance(retrieval_raw, dict) else {}

    return KBAgentConfig(
        name=kb_name,
        description=kb.get("description") or "",
        vector_store={"type": vdb, "settings": settings},
        embedding={
            "model_id": kb.get("embedding_model_id", "bedrock/amazon.titan-embed-text-v1"),
            "region_name": kb.get("embedding_region"),
        },
        text_splitter={
            "type": "recursive_character",
            "chunk_size": kb.get("chunk_size", 1000),
            "chunk_overlap": kb.get("chunk_overlap", 200),
        },
        retrieval_settings={
            "top_k": retrieval.get("top_k", 5),
            "score_threshold": retrieval.get("score_threshold", 0.7),
        },
        data_sources=[],  # KB already indexed in the registry — no local sources needed
    )


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@router.post("/knowledge-bases", status_code=201)
async def register_knowledge_base(
    registration: KBRegistration,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Register a new knowledge base (or reconfigure an existing one)."""
    user = "authenticated_user"
    vector_db_config: Dict = {}
    if registration.vector_db_config:
        vector_db_config = registration.vector_db_config.model_dump(exclude_none=True)

    try:
        kb = await registry.register_knowledge_base(
            name=registration.name,
            description=registration.description,
            tags=registration.tags,
            vector_db_type=registration.vector_db_type.value,
            deployment_mode=registration.deployment_mode.value,
            embedding_model_id=registration.embedding.model_id,
            embedding_region=registration.embedding.region_name,
            chunk_size=registration.chunking.chunk_size,
            chunk_overlap=registration.chunking.chunk_overlap,
            vector_db_config=vector_db_config,
            retrieval_config=registration.retrieval.model_dump(),
            performed_by=user,
        )
        return kb
    except Exception as exc:
        logger.error("Failed to register KB %s: %s", registration.name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/knowledge-bases/{kb_name}", response_model=StatusResponse)
async def delete_knowledge_base(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Delete a knowledge base and all its vector data."""
    try:
        await registry.delete_knowledge_base(kb_name)
        return StatusResponse(status="deleted", message=f"Knowledge base '{kb_name}' deleted")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to delete KB %s: %s", kb_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@router.get("/knowledge-bases/{kb_name}/history")
async def get_kb_history(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Return the action audit log for a knowledge base."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_name}")
    actions = await registry.db.get_kb_actions(kb["id"])
    return {"kb_name": kb_name, "actions": actions}
