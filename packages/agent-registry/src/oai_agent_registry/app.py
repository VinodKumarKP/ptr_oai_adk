import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from oai_agent_registry.api import router
from oai_agent_registry.dependencies import registry_instance
from oai_agent_registry.util import get_public_ip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the application's lifespan."""
    logger.info("Initializing HTTP client...")
    registry_instance.client = httpx.AsyncClient(
        timeout=httpx.Timeout(registry_instance.registry_config.default_timeout, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    registry_instance.public_ip = get_public_ip()
    
    if registry_instance.registry_config.enable_auto_discovery:
        await registry_instance.discover_agents()

    yield
    logger.info("Closing HTTP client...")
    await registry_instance.client.aclose()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agent Registry",
        description="A registry for routing requests to multiple agent containers.",
        version="1.0.0",
        lifespan=lifespan
    )

    if registry_instance.registry_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    app.include_router(router)
    return app

app = create_app()