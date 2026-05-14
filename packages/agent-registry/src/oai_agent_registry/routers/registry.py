from fastapi import APIRouter, Request, Depends, HTTPException
from typing import Optional
from oai_agent_registry.services.registry import AgentRegistry
from oai_agent_registry.dependencies import get_registry
from oai_agent_registry.models import AgentRegistration, AgentDeregistration, AgentLifecycleAction, AgentAction, AgentActionHistory

router = APIRouter()

@router.get("/")
async def root():
    """Returns a list of available endpoints."""
    return {
        "message": "Agent Registry",
        "endpoints": {
            "GET /info": "Available agent endpoints.",
            "GET /health": "Health check endpoint.",
            "POST /reload-config": "Reload configuration from file.",
            "POST /register": "Register a new agent.",
            "POST /deregister": "Deregister an agent.",
            "POST /lifecycle/{agent_name}": "Execute a lifecycle action on an agent.",
            "GET /history/{agent_name}": "Get action history for an agent."
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

@router.post("/register")
async def register_agent(
    agent_registration: AgentRegistration, 
    stream_output: bool = False, 
    registry: AgentRegistry = Depends(get_registry)
):
    """Registers a new agent."""
    return await registry.register_agent(agent_registration, stream_output)

@router.post("/deregister")
async def deregister_agent(agent_deregistration: AgentDeregistration, registry: AgentRegistry = Depends(get_registry)):
    """Deregisters an agent."""
    return await registry.deregister_agent(agent_deregistration)

@router.post("/lifecycle/{agent_name}")
async def execute_lifecycle_action(
    agent_name: str,
    action_payload: AgentLifecycleAction,
    registry: AgentRegistry = Depends(get_registry)
):
    """Executes a lifecycle action on an agent."""
    return await registry.execute_lifecycle_action(agent_name, action_payload.action, action_payload.version, action_payload.stream_output)

@router.get("/history/{agent_name}", response_model=AgentActionHistory)
async def get_agent_history(
    agent_name: str,
    action: Optional[str] = None,
    limit: int = 100,
    registry: AgentRegistry = Depends(get_registry)
):
    """
    Retrieves action history for a specific agent.

    Query Parameters:
    - action: Optional action type to filter by (e.g., 'start', 'stop', 'rebuild')
    - limit: Maximum number of actions to return (default: 100)

    Returns:
    - AgentActionHistory with agent_name, total_count, and list of actions
    """
    try:
        actions = await registry.db_logger.get_agent_actions(agent_name, action, limit)
        count = await registry.db_logger.get_agent_action_count(agent_name)

        # Convert raw action dicts to AgentAction models
        action_objects = [AgentAction(**a) for a in actions]

        return AgentActionHistory(
            agent_name=agent_name,
            total_count=count,
            actions=action_objects
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve action history: {str(e)}")

@router.api_route("/{agent_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(agent_name: str, path: str, request: Request, registry: AgentRegistry = Depends(get_registry)):
    """Proxies a request to the specified agent."""
    return await registry.proxy_request(agent_name, path, request)
