"""
Document management routes.

  POST /knowledge-bases/{kb_name}/documents/upload  — Upload a file for indexing
  POST /knowledge-bases/{kb_name}/documents/s3      — Index an S3 object
  GET  /knowledge-bases/{kb_name}/documents         — List documents
  DELETE /knowledge-bases/{kb_name}/documents/{doc_id} — Remove a document
  POST /knowledge-bases/{kb_name}/reindex           — Reindex all documents
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from oai_kb_registry.dependencies import get_registry, verify_bearer_token
from oai_kb_registry.models import ReindexResponse, S3DocumentSource, StatusResponse
from oai_kb_registry.services.kb_registry import KBRegistry

router = APIRouter()
logger = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "application/octet-stream",  # generic — accepted and we detect by extension
}

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".csv"}


def _validate_upload(upload: UploadFile) -> None:
    import os  # noqa: PLC0415
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {ext!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            ),
        )


# ---------------------------------------------------------------------------
# Upload (multipart file)
# ---------------------------------------------------------------------------

@router.post("/knowledge-bases/{kb_name}/documents/upload", status_code=201)
async def upload_document(
    kb_name: str,
    file: UploadFile = File(...),
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Upload a file and index it into the knowledge base.

    Supported formats: PDF, DOCX, TXT, MD, CSV.
    The file is parsed, chunked, embedded, and stored in the configured vector DB.
    """
    _validate_upload(file)

    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        doc = await registry.index_document_from_upload(
            kb_name=kb_name,
            filename=file.filename or "unknown",
            data=data,
        )
        return {"status": "indexed", "document": doc}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Upload indexing failed for KB %s: %s", kb_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# S3 source indexing
# ---------------------------------------------------------------------------

@router.post("/knowledge-bases/{kb_name}/documents/s3", status_code=201)
async def index_s3_document(
    kb_name: str,
    source: S3DocumentSource,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Index an S3 object into the knowledge base.

    Supports PDF, DOCX, TXT, and Markdown objects.
    Credentials are optional — falls back to the KB's stored AWS config or
    the instance IAM role / environment AWS_* vars.
    """
    try:
        doc = await registry.index_document_from_s3(
            kb_name=kb_name,
            bucket=source.bucket,
            key=source.key,
            region=source.region,
            aws_access_key_id=source.aws_access_key_id,
            aws_secret_access_key=source.aws_secret_access_key,
        )
        return {"status": "indexed", "document": doc}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("S3 indexing failed for KB %s: %s", kb_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

@router.get("/knowledge-bases/{kb_name}/documents", response_model=List[Dict[str, Any]])
async def list_documents(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """List all documents registered in the knowledge base."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_name}")
    return await registry.db.get_kb_documents(kb["id"])


# ---------------------------------------------------------------------------
# Remove document
# ---------------------------------------------------------------------------

@router.delete("/knowledge-bases/{kb_name}/documents/{doc_id}", response_model=StatusResponse)
async def remove_document(
    kb_name: str,
    doc_id: int,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Remove a document from the knowledge base.

    Note: Vector embeddings for the document are not immediately deleted from the
    vector store.  Run ``POST /knowledge-bases/{kb_name}/reindex`` after removals
    to rebuild a clean vector store without the deleted document's chunks.
    """
    try:
        await registry.remove_document(kb_name, doc_id)
        return StatusResponse(
            status="removed",
            message=f"Document {doc_id} removed from '{kb_name}'. Reindex to clean vector store.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Remove document failed for KB %s doc %s: %s", kb_name, doc_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------

@router.post("/knowledge-bases/{kb_name}/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Drop and rebuild the vector store from all indexed documents.

    Upload-sourced documents cannot be re-processed automatically (the original
    bytes are not stored).  They will be marked ``failed`` — re-upload them to
    re-index.  S3-sourced documents are re-fetched and re-indexed automatically.
    """
    try:
        queued = await registry.reindex_knowledge_base(kb_name)
        return ReindexResponse(
            status="started",
            kb_name=kb_name,
            documents_queued=queued,
            message=f"Reindex started for {queued} document(s). Check document statuses for progress.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Reindex failed for KB %s: %s", kb_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))
