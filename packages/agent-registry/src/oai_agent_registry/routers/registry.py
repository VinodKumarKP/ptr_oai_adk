from fastapi import APIRouter, Request, Depends
from oai_agent_registry.services.registry import AgentRegistry
from oai_agent_registry.dependencies import get_registry

router = APIRouter()

@router.get("/")
async def root():
    """Returns a list of available endpoints."""
    return {
        "message": "Agent Registry",
        "endpoints": {
            "GET /info": "Available agent endpoints.",
            "GET /health": "Health check endpoint.",
            "POST /reload-config": "Reload configuration from file."
        }
    }

@router.get("/info")
async def get_info(registry: AgentRegistry = Depends(get_registry)):
    """Returns information about the registry and its agents."""
    return await registry.get_info()

@router.get("/health")
async def health_check(registry: AgentRegistry = Depends(get_registry)):
    """Performs a health check on all enabled agents."""
    return await registry.health_check()

@router.post("/reload-config")
async def reload_config(registry: AgentRegistry = Depends(get_registry)):
    """Reloads the configuration from the config file."""
    return await registry.reload_config()

@router.api_route("/{agent_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(agent_name: str, path: str, request: Request, registry: AgentRegistry = Depends(get_registry)):
    """Proxies a request to the specified agent."""
    return await registry.proxy_request(agent_name, path, request)
