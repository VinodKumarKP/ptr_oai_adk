from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from oai_mcp_registry.services.registry import MCPRegistry
from oai_mcp_registry.dependencies import get_registry

router = APIRouter()

@router.get("/")
async def root():
    """Returns a list of all available endpoints."""
    info = {
        "message": "MCP Proxy Server",
        "endpoints": {
            "GET /health": "Health check endpoint",
            "GET /info": "Available proxied MCP Server endpoints",
            "POST /reload-config": "Reload configuration",
        },
    }
    return JSONResponse(info)

@router.get("/health")
async def health_check(registry: MCPRegistry = Depends(get_registry)):
    return {
        "status": "ok",
        "configured_servers": len(registry.sub_apps),
        "host_ip": registry.public_ip
    }

@router.get("/info")
async def get_config(registry: MCPRegistry = Depends(get_registry)):
    servers_info = {}
    if registry.config:
        for name, config in registry.config.servers.items():
            servers_info[name] = {
                "description": config.description,
                "endpoint": f"http://{registry.public_ip}:8081/{name}/mcp",
                "status": "active" if name in registry.sub_apps else "inactive"
            }
    return {
        "proxy_info": {
            "host_ip": registry.host_ip,
            "total_servers": len(registry.config.servers) if registry.config else 0,
            "active_servers": len(registry.sub_apps),
            "config_path": registry.config_path
        },
        "mcp_servers": servers_info
    }

@router.post("/reload-config")
async def reload_config(registry: MCPRegistry = Depends(get_registry)):
    return registry.reload_config()
