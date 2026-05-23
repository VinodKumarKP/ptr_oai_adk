"""
Knowledge base CRUD routes.

  GET  /                            — API overview
  GET  /health                      — Health check
  GET  /knowledge-bases             — List all knowledge bases
  GET  /knowledge-bases/{kb_name}   — Get a single KB (with doc counts)
  POST /knowledge-bases             — Register a new KB
  DELETE /knowledge-bases/{kb_name} — Delete KB + all documents
  GET  /knowledge-bases/{kb_name}/history  — Audit log
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from oai_kb_registry.dependencies import get_registry, verify_bearer_token, get_auth_user
from oai_kb_registry.models import KBDetails, KBRegistration, KBStatus, StatusResponse
from oai_kb_registry.services.kb_registry import KBRegistry

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
