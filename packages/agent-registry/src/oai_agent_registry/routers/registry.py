import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from oai_agent_registry.dependencies import get_registry
from oai_agent_registry.models import (
    AgentAction,
    AgentActionHistory,
    AgentDeregistration,
    AgentDiscoveryResult,
    AgentLifecycleAction,
    AgentRegistration,
    BulkAgentRegistrationRequest,
    BulkAgentRegistrationResult,
)
from oai_agent_registry.security.dependencies import verify_api_key
from oai_agent_registry.services.registry import AgentRegistry

router = APIRouter()
logger = logging.getLogger(__name__)

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
            "GET /history/{agent_name}": "Get action history for an agent.",
            "POST /agents/discover": "Discover agents from a GitHub repository.",
            "POST /agents/register-bulk": "Bulk-register discovered agents.",
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

@router.post("/agents/discover", response_model=AgentDiscoveryResult)
async def discover_agents(
    request_data: Dict,
    registry: AgentRegistry = Depends(get_registry),
):
    """
    Discover all agent config YAMLs in a GitHub repository.

    Scans ``agentic_registry_agents/agents_config/`` (and ``agents_config/`` as
    fallback) for ``.yaml`` files and returns parsed metadata for each agent
    found.  No authentication required for public repositories.
    """
    from oai_agent_registry.services.agent_discovery import AgentDiscovery

    git_repository_url = request_data.get("git_repository_url")
    if not git_repository_url:
        raise HTTPException(status_code=400, detail="git_repository_url is required")

    auth_token = request_data.get("auth_token")
    config_path = request_data.get("config_path")  # optional explicit YAML directory

    try:
        discovery = AgentDiscovery(logger=logger)
        existing_names = list(registry.agents.keys())
        result = await discovery.discover(
            git_repository_url=git_repository_url,
            existing_agent_names=existing_names,
            auth_token=auth_token,
            config_path=config_path,
        )
        return AgentDiscoveryResult(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to discover agents: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/agents/register-bulk", response_model=BulkAgentRegistrationResult)
async def register_agents_bulk(
    request: BulkAgentRegistrationRequest,
    api_key: bool = Depends(verify_api_key),
    registry: AgentRegistry = Depends(get_registry),
):
    """
    Register and deploy multiple agents from a GitHub repository in one operation.

    Fetches each selected agent's config YAML, then calls the same
    ``register_agent`` path used by the individual /register endpoint so that
    deployment is triggered automatically according to ``deployment_mode``.

    Requires a valid API token.
    """
    from oai_agent_registry.services.agent_discovery import AgentDiscovery

    if not request.agent_names:
        raise HTTPException(status_code=400, detail="agent_names must not be empty")

    try:
        # Re-discover to get the YAML metadata for the selected names
        discovery = AgentDiscovery(logger=logger)
        discovery_result = await discovery.discover(
            git_repository_url=request.git_repository_url,
            auth_token=request.auth_token,
            config_path=request.config_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Discovery failed during bulk register: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build a lookup by agent name  (discover() returns a plain dict, not a model)
    agents_by_name: Dict[str, Dict] = {
        a["name"]: a for a in discovery_result["agents"]
    }

    successful: list[str] = []
    failed: list[Dict[str, str]] = []

    for agent_name in request.agent_names:
        agent_data = agents_by_name.get(agent_name)
        if agent_data is None:
            failed.append({"agent_name": agent_name, "error": "Not found in repository"})
            continue

        if agent_data.get("error"):
            failed.append({"agent_name": agent_name, "error": agent_data["error"]})
            continue

        try:
            # Apply overrides from request; fall back to values parsed from YAML
            framework = request.framework or agent_data.get("framework")
            deployment_mode = request.deployment_mode  # always set (default: docker)

            agent_registration = AgentRegistration(
                name=agent_name,
                description=agent_data.get("description", ""),
                endpoint=agent_data.get("endpoint") or "",
                port=agent_data.get("port"),
                source=agent_data.get("source") or request.git_repository_url,
                active=True,
                registered_via="registry",
                framework=framework,
                prompts=agent_data.get("prompts") or [],
                tags=agent_data.get("tags") or [],
                current_version=None,
                available_versions=[],
                deployment_mode=deployment_mode,
            )

            # Delegate to register_agent so the deployer is invoked based on
            # deployment_mode (docker / kubernetes / python_package).
            await registry.register_agent(agent_registration, stream_output=False)

            successful.append(agent_name)
            logger.info(
                "Bulk-registered and deploying agent '%s' (mode=%s) from %s",
                agent_name, deployment_mode, request.git_repository_url,
            )
        except Exception as exc:
            logger.error("Failed to register agent '%s': %s", agent_name, exc)
            failed.append({"agent_name": agent_name, "error": str(exc)})

    return BulkAgentRegistrationResult(
        total_registered=len(successful),
        successful=successful,
        failed=failed,
    )


@router.api_route("/{agent_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(agent_name: str, path: str, request: Request, registry: AgentRegistry = Depends(get_registry)):
    """Proxies a request to the specified agent."""
    return await registry.proxy_request(agent_name, path, request)
