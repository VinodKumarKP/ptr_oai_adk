"""
Main FastAPI application for Knowledge Base Registry.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oai_kb_registry.routers import router
from oai_kb_registry.dependencies import initialize_registry, close_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Knowledge Base Registry…")
    try:
        await initialize_registry(logger)
        logger.info("Knowledge Base Registry initialised successfully")
    except Exception as exc:
        logger.error("Failed to initialise KB Registry: %s", exc)
        raise
    yield
    logger.info("Shutting down Knowledge Base Registry…")
    try:
        await close_registry()
        logger.info("Knowledge Base Registry closed")
    except Exception as exc:
        logger.error("Error during KB Registry shutdown: %s", exc)


app = FastAPI(
    title="Knowledge Base Registry",
    description=(
        "Manage vector-backed knowledge bases. "
        "Register KBs, ingest documents (PDF, DOCX, TXT, S3), "
        "and query via semantic similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1/kb-registry")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8085"))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "oai_kb_registry.main:app",
        host=host,
        port=port,
        reload=os.environ.get("ENV", "development") == "development",
    )
