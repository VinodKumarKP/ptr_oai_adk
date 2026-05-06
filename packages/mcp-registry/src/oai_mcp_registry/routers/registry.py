import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from oai_mcp_registry.services.registry import MCPRegistry
from oai_mcp_registry.dependencies import get_registry
from oai_mcp_registry.models import ServerRegistration, ServerDeregistration

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
            "POST /register": "Register a new MCP server dynamically",
            "POST /deregister": "Deregister an MCP server",
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
        endpoint = os.environ.get('AGENT_BASE_URL', "localhost")
        for name, config in registry.config.servers.items():
            servers_info[name] = {
                "description": config.description,
                "endpoint": f"{endpoint}:8081/{name}/mcp",
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
    return await registry.reload_config()

@router.post("/register")
async def register_server(request: Request, server_registration: ServerRegistration, registry: MCPRegistry = Depends(get_registry)):
    """Registers a new MCP server."""
    return await registry.register_server(request.app, server_registration)

@router.post("/deregister")
async def deregister_server(server_deregistration: ServerDeregistration, registry: MCPRegistry = Depends(get_registry)):
    """Deregisters an MCP server."""
    return await registry.deregister_server(server_deregistration)
