"""
Semantic search / query route.

  POST /knowledge-bases/{kb_name}/query  — Run a similarity search
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from oai_kb_registry.dependencies import get_registry, verify_bearer_token
from oai_kb_registry.models import QueryRequest, QueryResponse, QueryResult
from oai_kb_registry.services.kb_registry import KBRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/knowledge-bases/{kb_name}/query", response_model=QueryResponse)
async def query_knowledge_base(
    kb_name: str,
    request: QueryRequest,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
):
    """Run a semantic similarity search against the specified knowledge base.

    Returns the top-k most relevant document chunks with similarity scores.
    You can optionally filter by ``score_threshold`` (0.0–1.0) to discard
    low-relevance results, and by ``filter_metadata`` to restrict the search
    to documents matching specific metadata key-value pairs.

    Example request body::

        {
            "query": "What are the data retention policies?",
            "k": 5,
            "score_threshold": 0.6,
            "filter_metadata": {"doc_name": "policy_v2.pdf"}
        }
    """
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base not found: {kb_name}")

    if kb.get("status") == "error":
        raise HTTPException(
            status_code=503,
            detail=(
                f"Knowledge base '{kb_name}' is in error state. "
                "Check the audit log for details."
            ),
        )

    try:
        raw_results = await registry.query(
            kb_name=kb_name,
            query_text=request.query,
            k=request.k,
            score_threshold=request.score_threshold,
            filter_metadata=request.filter_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Query failed for KB %s: %s", kb_name, exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    results = [
        QueryResult(
            content=r["content"],
            score=r.get("score"),
            metadata=r.get("metadata") if request.include_metadata else None,
            document_name=r.get("document_name"),
            document_id=r.get("document_id"),
        )
        for r in raw_results
    ]

    return QueryResponse(
        kb_name=kb_name,
        query=request.query,
        results=results,
        total_results=len(results),
    )
