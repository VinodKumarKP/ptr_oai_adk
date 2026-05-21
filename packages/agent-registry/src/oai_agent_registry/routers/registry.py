import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

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


def _build_agent_registrations(
    request: BulkAgentRegistrationRequest,
    agents_by_name: Dict[str, Dict],
) -> Tuple[List[Tuple[str, "AgentRegistration"]], List[Dict[str, str]]]:
    """
    Build AgentRegistration objects for each requested agent name.
    Returns (to_deploy, pre_failed) where pre_failed contains agents that
    couldn't be found or had parse errors before deployment even starts.
    """
    to_deploy: List[Tuple[str, AgentRegistration]] = []
    pre_failed: List[Dict[str, str]] = []

    for agent_name in request.agent_names:
        agent_data = agents_by_name.get(agent_name)
        if agent_data is None:
            pre_failed.append({"agent_name": agent_name, "error": "Not found in repository"})
            continue
        if agent_data.get("error"):
            pre_failed.append({"agent_name": agent_name, "error": agent_data["error"]})
            continue

        framework = request.framework or agent_data.get("framework")
        to_deploy.append((
            agent_name,
            AgentRegistration(
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
                deployment_mode=request.deployment_mode,
                env_vars=request.env_vars or None,
            ),
        ))

    return to_deploy, pre_failed


async def _stream_bulk_deployment(
    agents_to_deploy: List[Tuple[str, "AgentRegistration"]],
    pre_failed: List[Dict[str, str]],
    registry: AgentRegistry,
) -> AsyncGenerator[str, None]:
    """
    Async generator that deploys all agents concurrently and yields SSE events.

    Each event is a JSON-encoded line:
      data: {"agent": <name>, "status": "queued|building|done|error", "line": <text>}

    A final summary event signals completion:
      data: {"type": "complete", "total_registered": N, "successful": [...], "failed": [...]}
    """
    from oai_agent_registry.models import AgentConfig

    queue: asyncio.Queue = asyncio.Queue()
    successful: List[str] = []
    failed: List[Dict[str, str]] = list(pre_failed)  # seed with pre-discovery failures

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # Announce every agent immediately so the UI can render panels upfront
    for agent_name, _ in agents_to_deploy:
        yield _sse({"agent": agent_name, "status": "queued", "line": "Queued for deployment…"})

    async def _deploy_one(agent_name: str, agent_reg: AgentRegistration) -> None:
        """Deploy a single agent and push all output lines to the shared queue."""
        try:
            deployer = registry._get_deployer(agent_reg.deployment_mode)
            if not deployer:
                raise RuntimeError(f"No deployer available for mode '{agent_reg.deployment_mode}'")

            # Assign a port if none was specified in the YAML
            if not agent_reg.port:
                agent_reg.port = deployer.find_available_port()

            await queue.put({
                "agent": agent_name,
                "status": "building",
                "line": f"Starting {agent_reg.deployment_mode} deployment on port {agent_reg.port}…",
            })

            async for raw_line in deployer.stream_deploy_agent(
                agent_name=agent_name,
                source_url=agent_reg.source,
                framework=agent_reg.framework,
                env=agent_reg.env_vars or {},
                description=agent_reg.description or "",
                tags=agent_reg.tags or [],
                port=agent_reg.port,
                current_version=agent_reg.current_version,
                refresh_repo=False,
            ):
                line = raw_line.strip()
                if line:
                    await queue.put({"agent": agent_name, "status": "building", "line": line})

            # Persist to DB and in-memory registry after a successful build
            db_values = await registry._get_merged_agent_values(agent_name, agent_reg, "registry")
            if agent_reg.port:
                db_values["port"] = agent_reg.port

            registry.agents[agent_name] = AgentConfig(**db_values)

            await registry.db_logger.log_agent_registration(
                agent_name=agent_name,
                endpoint_url=db_values["endpoint"],
                port=db_values["port"],
                source=db_values["source"],
                active=db_values["active"],
                registered_via="registry",
                framework=db_values["framework"],
                prompts=db_values["prompts"],
                tags=db_values["tags"],
                description=db_values["description"],
                current_version=db_values["current_version"],
                available_versions=db_values["available_versions"],
                deployment_mode=db_values["deployment_mode"],
                env_vars=db_values.get("env_vars"),
            )

            successful.append(agent_name)
            await queue.put({
                "agent": agent_name,
                "status": "done",
                "line": f"✅ Agent '{agent_name}' deployed successfully.",
            })

        except Exception as exc:
            logger.error("Bulk deploy failed for '%s': %s", agent_name, exc)
            failed.append({"agent_name": agent_name, "error": str(exc)})
            await queue.put({
                "agent": agent_name,
                "status": "error",
                "line": f"❌ {exc}",
            })
        finally:
            await queue.put(None)  # sentinel — this agent is finished

    # Launch all deployments concurrently
    for agent_name, agent_reg in agents_to_deploy:
        asyncio.create_task(_deploy_one(agent_name, agent_reg))

    # Drain the queue until every task has sent its sentinel
    pending = len(agents_to_deploy)
    while pending > 0:
        event = await queue.get()
        if event is None:
            pending -= 1
        else:
            yield _sse(event)

    # Final summary — frontend uses this to render the completion card
    yield _sse({
        "type": "complete",
        "total_registered": len(successful),
        "successful": successful,
        "failed": failed,
    })


@router.post("/agents/register-bulk")
async def register_agents_bulk(
    request: BulkAgentRegistrationRequest,
    stream_output: bool = False,
    api_key: bool = Depends(verify_api_key),
    registry: AgentRegistry = Depends(get_registry),
):
    """
    Register and deploy multiple agents from a GitHub repository in one operation.

    When ``stream_output=true`` (query param) returns an SSE stream where each
    event is a JSON object tagged with the agent name, allowing the UI to show
    live build logs for every agent in parallel.

    When ``stream_output=false`` (default) deployment runs in background tasks
    and a JSON summary is returned immediately.

    Requires a valid API token.
    """
    from oai_agent_registry.services.agent_discovery import AgentDiscovery

    if not request.agent_names:
        raise HTTPException(status_code=400, detail="agent_names must not be empty")

    try:
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

    agents_by_name: Dict[str, Dict] = {
        a["name"]: a for a in discovery_result["agents"]
    }

    agents_to_deploy, pre_failed = _build_agent_registrations(request, agents_by_name)

    # ── Streaming path ────────────────────────────────────────────────────────
    if stream_output:
        return StreamingResponse(
            _stream_bulk_deployment(agents_to_deploy, pre_failed, registry),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            },
        )

    # ── Non-streaming path (background tasks, immediate JSON response) ────────
    successful: List[str] = []
    failed: List[Dict[str, str]] = list(pre_failed)

    for agent_name, agent_reg in agents_to_deploy:
        try:
            await registry.register_agent(agent_reg, stream_output=False)
            successful.append(agent_name)
            logger.info("Bulk-registered agent '%s' (mode=%s)", agent_name, agent_reg.deployment_mode)
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
