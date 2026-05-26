import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from oai_mcp_registry.dependencies import get_registry
from oai_platform_core.readme_fetcher import fetch_readme, read_local_readme, invalidate_readme_cache, readme_cache_stats

logger = logging.getLogger(__name__)
from oai_mcp_registry.models import (
    ServerRegistration,
    ServerDeregistration,
    UpdateServerEnvVarsRequest,
    McpServerLifecycleAction,
    ServerAction,
    ServerActionHistory,
    MCPServerDiscoveryResult,
    BulkMCPServerRegistrationRequest,
    BulkMCPServerRegistrationResult,
)
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
                "source": getattr(config, "source", None),
                "tags": getattr(config, "tags", []),
                "env_vars": getattr(config, "env_vars", None) or {},
                "sensitive_vars": getattr(config, "sensitive_vars", []) or [],
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
    return await registry.register_server(server_registration, stream_output=stream_output)


@router.post("/deregister")
async def deregister_server(server_deregistration: ServerDeregistration, registry: MCPRegistry = Depends(get_registry)):
    """Deregisters an MCP server."""
    return await registry.deregister_server(server_deregistration)


@router.patch("/servers/{mcp_server_name}/env-vars")
async def update_server_env_vars(
    mcp_server_name: str,
    payload: UpdateServerEnvVarsRequest,
    registry: MCPRegistry = Depends(get_registry),
):
    """Update environment variables for a registered MCP server."""
    return await registry.update_server_env_vars(mcp_server_name, payload.env_vars, payload.sensitive_vars)


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


# ---------------------------------------------------------------------------
# Bulk registration helpers
# ---------------------------------------------------------------------------

logger = __import__("logging").getLogger(__name__)


def _build_server_registrations(
    request: BulkMCPServerRegistrationRequest,
    servers_by_name: Dict[str, Dict],
) -> Tuple[List[Tuple[str, ServerRegistration]], List[Dict[str, str]]]:
    """Build ServerRegistration objects for each requested server name."""
    to_deploy: List[Tuple[str, ServerRegistration]] = []
    pre_failed: List[Dict[str, str]] = []

    user_overrides: Dict[str, Dict] = request.server_env_overrides or {}

    for server_name in request.server_names:
        server_data = servers_by_name.get(server_name)
        if server_data is None:
            pre_failed.append({"server_name": server_name, "error": "Not found in repository"})
            continue
        if server_data.get("error"):
            pre_failed.append({"server_name": server_name, "error": server_data["error"]})
            continue

        # Merge YAML-discovered env vars with any user-supplied overrides.
        # User overrides take precedence so real secret values replace placeholders.
        base_env: Dict = server_data.get("env_vars") or {}
        per_server_overrides: Dict = user_overrides.get(server_name, {})
        merged_env: Optional[Dict] = {**base_env, **per_server_overrides} if (base_env or per_server_overrides) else None

        to_deploy.append((
            server_name,
            ServerRegistration(
                name=server_name,
                description=server_data.get("description", ""),
                endpoint=server_data.get("endpoint") or "",
                port=server_data.get("port"),
                source=server_data.get("source") or request.git_repository_url,
                active=True,
                registered_via="registry",
                tags=server_data.get("tags") or [],
                current_version=None,
                available_versions=[],
                deployment_mode=request.deployment_mode,
                env_vars=merged_env,
                sensitive_vars=server_data.get("sensitive_vars") or [],
            ),
        ))

    return to_deploy, pre_failed


async def _stream_bulk_mcp_deployment(
    agents_to_deploy: List[Tuple[str, ServerRegistration]],
    pre_failed: List[Dict[str, str]],
    registry: MCPRegistry,
) -> AsyncGenerator[str, None]:
    """
    Async generator that deploys all MCP servers concurrently and yields SSE events.

    Each event is a JSON-encoded line:
      data: {"server": <name>, "status": "queued|building|done|error", "line": <text>}

    Final summary event:
      data: {"type": "complete", "total_registered": N, "successful": [...], "failed": [...]}
    """
    from oai_mcp_registry.models import ServerConfig

    queue: asyncio.Queue = asyncio.Queue()
    successful: List[str] = []
    failed: List[Dict[str, str]] = list(pre_failed)

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # Announce every server immediately so the UI can render panels upfront
    for server_name, _ in agents_to_deploy:
        yield _sse({"server": server_name, "status": "queued", "line": "Queued for deployment…"})

    async def _deploy_one(server_name: str, server_reg: ServerRegistration) -> None:
        try:
            deployer = registry._get_deployer(server_reg.deployment_mode)
            if not deployer:
                raise RuntimeError(f"No deployer available for mode '{server_reg.deployment_mode}'")

            if not server_reg.port:
                server_reg.port = deployer.find_available_port()

            await queue.put({
                "server": server_name,
                "status": "building",
                "line": f"Starting {server_reg.deployment_mode} deployment on port {server_reg.port}…",
            })

            async for raw_line in deployer.stream_deploy_server(
                server_name=server_name,
                source_url=server_reg.source,
                framework=None,
                env=server_reg.env_vars or {},
                description=server_reg.description or "",
                tags=server_reg.tags or [],
                port=server_reg.port,
                current_version=server_reg.current_version,
                refresh_repo=False,
            ):
                line = raw_line.strip()
                if line:
                    await queue.put({"server": server_name, "status": "building", "line": line})

            # Persist to DB and mount proxy after successful build
            db_values = await registry._get_merged_server_values(server_name, server_reg, "registry")
            if server_reg.port:
                db_values["port"] = server_reg.port

            server_config = ServerConfig(**db_values)
            registry.config.servers[server_name] = server_config

            # Mount the MCP proxy
            try:
                url = registry._build_upstream_url(server_config)
                if url:
                    from fastmcp import FastMCP
                    mcp = FastMCP.as_proxy(url, name=server_name)
                    sub_app = mcp.http_app()
                    await registry.stop_sub_app(server_name)
                    registry.sub_apps[server_name] = sub_app
                    await registry.start_sub_app(server_name, sub_app)
            except Exception as mount_exc:
                logger.warning("Proxy mount failed for '%s': %s", server_name, mount_exc)

            await registry.db_logger.log_server_registration(
                server_name=server_name,
                endpoint_url=db_values["endpoint"],
                port=db_values["port"],
                source=db_values["source"],
                active=db_values["active"],
                registered_via="registry",
                tags=db_values["tags"],
                description=db_values["description"],
                current_version=db_values["current_version"],
                available_versions=db_values["available_versions"],
                deployment_mode=db_values["deployment_mode"],
                env_vars=db_values.get("env_vars"),
                sensitive_vars=db_values.get("sensitive_vars"),
            )

            successful.append(server_name)
            await queue.put({
                "server": server_name,
                "status": "done",
                "line": f"✅ Server '{server_name}' deployed successfully.",
            })

        except Exception as exc:
            logger.error("Bulk deploy failed for MCP server '%s': %s", server_name, exc)
            failed.append({"server_name": server_name, "error": str(exc)})
            await queue.put({"server": server_name, "status": "error", "line": f"❌ {exc}"})
        finally:
            await queue.put(None)  # sentinel

    # Launch all deployments concurrently
    for server_name, server_reg in agents_to_deploy:
        asyncio.create_task(_deploy_one(server_name, server_reg))

    # Drain the queue until every task has sent its sentinel
    pending = len(agents_to_deploy)
    while pending > 0:
        event = await queue.get()
        if event is None:
            pending -= 1
        else:
            yield _sse(event)

    yield _sse({
        "type": "complete",
        "total_registered": len(successful),
        "successful": successful,
        "failed": failed,
    })


# ---------------------------------------------------------------------------
# Bulk registration endpoints
# ---------------------------------------------------------------------------

@router.post("/servers/discover", response_model=MCPServerDiscoveryResult)
async def discover_servers(
    request_data: Dict,
    registry: MCPRegistry = Depends(get_registry),
):
    """
    Discover all MCP server config YAMLs in a GitHub repository.

    Scans common config directories (``mcp_registry_servers/servers_config/``,
    ``servers_config/``, etc.) for ``.yaml`` files and returns parsed metadata.
    Pass ``config_path`` in the body to override auto-detection.
    """
    from oai_mcp_registry.services.mcp_discovery import MCPDiscovery

    git_repository_url = request_data.get("git_repository_url")
    if not git_repository_url:
        raise HTTPException(status_code=400, detail="git_repository_url is required")

    auth_token = request_data.get("auth_token")
    config_path = request_data.get("config_path")

    try:
        discovery = MCPDiscovery(logger=logger)
        existing_names = list(registry.config.servers.keys()) if registry.config else []
        result = await discovery.discover(
            git_repository_url=git_repository_url,
            existing_server_names=existing_names,
            auth_token=auth_token,
            config_path=config_path,
        )
        return MCPServerDiscoveryResult(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to discover MCP servers: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/servers/register-bulk")
async def register_servers_bulk(
    request: Request,
    bulk_request: BulkMCPServerRegistrationRequest,
    stream_output: bool = False,
    registry: MCPRegistry = Depends(get_registry),
):
    """
    Register and deploy multiple MCP servers from a GitHub repository.

    When ``stream_output=true`` returns a multiplexed SSE stream with live
    build logs for all servers running concurrently.
    When ``stream_output=false`` deployment runs as background tasks and a
    JSON summary is returned immediately.
    """
    from oai_mcp_registry.services.mcp_discovery import MCPDiscovery

    if not bulk_request.server_names:
        raise HTTPException(status_code=400, detail="server_names must not be empty")

    try:
        discovery = MCPDiscovery(logger=logger)
        discovery_result = await discovery.discover(
            git_repository_url=bulk_request.git_repository_url,
            auth_token=bulk_request.auth_token,
            config_path=bulk_request.config_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Discovery failed during bulk MCP register: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    servers_by_name: Dict[str, Dict] = {
        s["name"]: s for s in discovery_result["servers"]
    }

    to_deploy, pre_failed = _build_server_registrations(bulk_request, servers_by_name)

    # ── Streaming path ────────────────────────────────────────────────────────
    if stream_output:
        return StreamingResponse(
            _stream_bulk_mcp_deployment(to_deploy, pre_failed, registry),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming path ────────────────────────────────────────────────────
    successful: List[str] = []
    failed: List[Dict[str, str]] = list(pre_failed)

    for server_name, server_reg in to_deploy:
        try:
            await registry.register_server(server_reg, stream_output=False)
            successful.append(server_name)
            logger.info("Bulk-registered MCP server '%s' (mode=%s)", server_name, server_reg.deployment_mode)
        except Exception as exc:
            logger.error("Failed to register MCP server '%s': %s", server_name, exc)
            failed.append({"server_name": server_name, "error": str(exc)})

    return BulkMCPServerRegistrationResult(
        total_registered=len(successful),
        successful=successful,
        failed=failed,
    )


# ---------------------------------------------------------------------------
# README endpoints
# ---------------------------------------------------------------------------

@router.get("/servers/{server_name}/readme")
async def get_server_readme(
    server_name: str,
    registry: MCPRegistry = Depends(get_registry),
):
    """Return the MCP server's README.

    Resolution order:
    1. Local filesystem: ``MCP_LOCAL_DIR/{server_name}/README.md``
       (matches the convention: mcp_registry_servers/servers/{server_name}/README.md)
    2. GitHub: ``source`` field URL, fetched via raw.githubusercontent.com
    """
    server = await registry.db_logger.get_server_details(server_name)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    # ── 1. Local filesystem (preferred) ──────────────────────────────────────
    local_dir = os.getenv("MCP_LOCAL_DIR")
    if local_dir:
        content, meta = read_local_readme(local_dir, server_name)
        if content is not None:
            return {
                "server_name": server_name,
                "content":     content,
                "cached":      meta.get("cached", False),
                "local":       True,
                "file_path":   meta.get("file_path"),
            }

    # ── 2. GitHub fallback ────────────────────────────────────────────────────
    repo_url = server.get("source")
    if not repo_url:
        raise HTTPException(status_code=404, detail=f"No README found: configure MCP_LOCAL_DIR or add a 'source' URL")

    content, meta = await fetch_readme(
        repo_url=repo_url,
        cache_key=f"mcp:{server_name}",
        github_token=os.getenv("GITHUB_TOKEN"),
        filenames=[f"mcp_registry_servers/servers/{server_name}/README.md"]
    )
    if content is None:
        raise HTTPException(status_code=404, detail=meta.get("error", "README not found"))

    return {
        "server_name": server_name,
        "content":     content,
        "cached":      meta.get("cached", False),
        "branch":      meta.get("branch"),
        "source_url":  meta.get("url"),
    }


@router.post("/servers/{server_name}/readme/invalidate-cache")
async def invalidate_server_readme_cache(
    server_name: str,
    registry: MCPRegistry = Depends(get_registry),
):
    """Evict the cached README for this MCP server."""
    local_dir = os.getenv("MCP_LOCAL_DIR")
    if local_dir:
        invalidate_readme_cache(f"local:{local_dir}:{server_name}")
    invalidate_readme_cache(f"mcp:{server_name}")
    return {"server_name": server_name, "cache_cleared": True}


@router.get("/readme/cache-stats")
async def mcp_readme_cache_stats():
    """Return cache statistics for all MCP server README entries."""
    stats = readme_cache_stats()
    stats["entries"] = [e for e in stats["entries"] if e["key"].startswith("mcp:")]
    return stats
