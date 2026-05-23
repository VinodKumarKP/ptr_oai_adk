"""Routers package for KB Registry."""

from fastapi import APIRouter

from oai_kb_registry.routers.knowledge_bases import router as kb_router
from oai_kb_registry.routers.documents import router as doc_router
from oai_kb_registry.routers.query import router as query_router
from oai_kb_registry.routers.token import router as token_router

router = APIRouter()
router.include_router(kb_router)
router.include_router(doc_router)
router.include_router(query_router)
router.include_router(token_router)
