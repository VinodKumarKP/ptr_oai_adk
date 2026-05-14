import os
from typing import Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse

from oai_mcp_registry.dependencies import get_registry
from oai_mcp_registry.models import ServerRegistration, ServerDeregistration, McpServerLifecycleAction, ServerAction, ServerActionHistory
from oai_mcp_registry.services.registry import MCPRegistry

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
            "POST /lifecycle/{mcp_server_name}": "Execute a lifecycle action on an MCP server",
            "GET /history/{mcp_server_name}": "Get action history for an MCP server",
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
    endpoint = f"{os.environ.get('MCP_BASE_URL')}:{os.environ.get('MCP_BASE_URL_PORT', registry.registry_config.port)}"
    if registry.config:
        for name, config in registry.config.servers.items():
            servers_info[name] = {
                "description": config.description,
                "endpoint": f"{endpoint}/{name}/mcp",
                "enabled": config.enabled,
                "status": "active" if config.enabled else 'inactive',
                "current_version": getattr(config, "current_version", None),
                "available_versions": getattr(config, "available_versions", []),
                "deployment_mode": getattr(config, "deployment_mode", "docker"),
                "available_actions": [
                    "start" if not config.enabled else "stop",
                    "restart",
                    "rebuild",
                    "redeploy",
                    "update",
                    "upgrade",
                    "downgrade",
                    "delete"
                ]
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
async def register_server(request: Request, server_registration: ServerRegistration,
                          stream_output: bool = False,
                          registry: MCPRegistry = Depends(get_registry)):
    """Registers a new MCP server."""
    return await registry.register_server(request.app, server_registration, stream_output=stream_output)


@router.post("/deregister")
async def deregister_server(server_deregistration: ServerDeregistration, registry: MCPRegistry = Depends(get_registry)):
    """Deregisters an MCP server."""
    return await registry.deregister_server(server_deregistration)


@router.post("/lifecycle/{mcp_server_name}")
async def execute_lifecycle_action(
        mcp_server_name: str,
        action_payload: McpServerLifecycleAction,
        registry: MCPRegistry = Depends(get_registry)
):
    """Executes a lifecycle action on a mcp server."""
    return await registry.execute_lifecycle_action(mcp_server_name, action_payload.action, action_payload.version,
                                                   action_payload.stream_output)


@router.get("/history/{mcp_server_name}", response_model=ServerActionHistory)
async def get_server_history(
    mcp_server_name: str,
    action: Optional[str] = None,
    limit: int = 100,
    registry: MCPRegistry = Depends(get_registry)
):
    """
    Retrieves action history for a specific MCP server.

    Query Parameters:
    - action: Optional action type to filter by (e.g., 'start', 'stop', 'restart')
    - limit: Maximum number of actions to return (default: 100)

    Returns:
    - ServerActionHistory with server_name, total_count, and list of actions
    """
    try:
        actions = await registry.db_logger.get_server_actions(mcp_server_name, action, limit)
        count = await registry.db_logger.get_server_action_count(mcp_server_name)

        # Convert raw action dicts to ServerAction models
        action_objects = [ServerAction(**a) for a in actions]

        return ServerActionHistory(
            server_name=mcp_server_name,
            total_count=count,
            actions=action_objects
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve action history: {str(e)}")
