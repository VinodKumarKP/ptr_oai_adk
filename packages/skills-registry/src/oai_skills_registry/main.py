"""
Main FastAPI application for Skills Registry.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oai_skills_registry.routers import router
from oai_skills_registry.dependencies import initialize_registry, close_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting Skills Registry...")
    try:
        await initialize_registry(logger)
        logger.info("Skills Registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Skills Registry: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Skills Registry...")
    try:
        await close_registry()
        logger.info("Skills Registry closed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Skills Registry",
    description="Manage skill lifecycle with Git integration",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = os.environ.get(
    "CORS_ORIGINS",
    "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1/skills-registry")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8083"))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        "oai_skills_registry.main:app",
        host=host,
        port=port,
        reload=os.environ.get("ENV", "development") == "development",
    )
