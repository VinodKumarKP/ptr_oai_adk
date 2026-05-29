"""Routers package for KB Registry."""

from fastapi import APIRouter

from oai_kb_registry.routers.knowledge_bases import router as kb_router
from oai_kb_registry.routers.documents import router as doc_router
from oai_kb_registry.routers.query import router as query_router
from oai_kb_registry.routers.sources import router as sources_router
from oai_kb_registry.routers.token import router as token_router

router = APIRouter()
router.include_router(kb_router,      tags=["Knowledge Bases"])
router.include_router(doc_router,     tags=["Documents"])
router.include_router(query_router,   tags=["Query"])
router.include_router(sources_router, tags=["Data Sources"])
router.include_router(token_router,   tags=["Tokens"])
