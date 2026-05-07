import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import JSONResponse
import httpx

from oai_mcp_registry.dependencies import registry_instance
from oai_mcp_registry.routers.registry import router
from oai_mcp_registry.security.dependencies import verify_api_key, api_key_header, _validate_token

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mcp")
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MCPRegistry")

# Registry-internal route prefixes — never dispatched to a sub-app or proxied
REGISTRY_ROUTES = frozenset([
    "register", "deregister", "info", "health",
    "reload-config", "lifecycle", "docs", "openapi.json", "redoc"
])


# ---------------------------------------------------------------------------
# FastAPI app (innermost layer)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sub-app lifespan runner
#
# Each FastMCP sub-app uses anyio cancel scopes and contextvars that MUST be
# entered and exited in the same asyncio Task.  Sharing an AsyncExitStack
# across the registry lifespan violates this — the stack tears down contexts
# in a different task than the one that created them, causing:
#   RuntimeError: Attempted to exit cancel scope in a different task
#
# Fix: run every sub-app lifespan inside its own dedicated asyncio Task via
# asyncio.Event-based handshake so the cancel scope never crosses task boundaries.
# ---------------------------------------------------------------------------

# Tracks running lifespan tasks: name -> (task, shutdown_event)
_sub_app_tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}


async def _run_sub_app_lifespan(name: str, sub_app, ready: asyncio.Event, shutdown: asyncio.Event):
    """
    Runs a single sub-app's lifespan entirely within this task.
    Signals `ready` once the lifespan has started, then waits for
    `shutdown` before tearing down.
    """
    try:
        async with sub_app.router.lifespan_context(sub_app):
            ready.set()
            logger.info(f"  [Connected] {name}")
            await shutdown.wait()
    except Exception as e:
        logger.error(f"  [Lifespan Error] {name}: {e}")
        ready.set()  # unblock caller even on failure


async def start_sub_app(name: str, sub_app) -> bool:
    """Start a sub-app lifespan task. Returns True if started successfully."""
    if name in _sub_app_tasks:
        return True  # already running

    ready = asyncio.Event()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        _run_sub_app_lifespan(name, sub_app, ready, shutdown),
        name=f"lifespan:{name}"
    )
    _sub_app_tasks[name] = (task, shutdown)
    await ready.wait()  # block until the sub-app's lifespan context is entered
    return not task.done() or not task.exception() if task.done() else True


async def stop_sub_app(name: str):
    """Signal a sub-app lifespan task to shut down and wait for it."""
    entry = _sub_app_tasks.pop(name, None)
    if entry is None:
        return
    task, shutdown = entry
    shutdown.set()
    try:
        await asyncio.wait_for(task, timeout=10.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"  [Shutdown Warning] {name}: {e}")


async def stop_all_sub_apps():
    """Shut down all running sub-app lifespan tasks concurrently."""
    names = list(_sub_app_tasks.keys())
    await asyncio.gather(*[stop_sub_app(n) for n in names], return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry_instance.initialize()

    if registry_instance.registry_config.enable_auto_discovery:
        host = registry_instance.registry_config.host
        await registry_instance.discover_servers(host)

    # Start each sub-app in its own task so anyio cancel scopes stay within
    # the task that created them — avoids the cross-task context var errors.
    logger.info("--- Starting Upstream Connections ---")
    results = await asyncio.gather(
        *[start_sub_app(name, sub_app) for name, sub_app in registry_instance.sub_apps.items()],
        return_exceptions=True
    )
    active_count = sum(1 for r in results if r is True)
    logger.info(f"--- Proxy Ready: {active_count}/{len(registry_instance.sub_apps)} upstreams active ---")

    # Expose start/stop helpers to registry_instance so register_server can use them
    registry_instance.start_sub_app = start_sub_app
    registry_instance.stop_sub_app = stop_sub_app

    yield

    logger.info("--- Shutting Down ---")
    await stop_all_sub_apps()
    await registry_instance.shutdown()


_fastapi_app = FastAPI(
    title="MCP Gateway",
    description="Unified Proxy for Distributed MCP Servers",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
    security=[{api_key_header.model.name: []}],
)

_fastapi_app.include_router(router)


# ---------------------------------------------------------------------------
# Layer 1 (innermost ASGI wrapper): DynamicMCPDispatcher
#
# FastAPI freezes its router at startup, so app.mount() called later has no
# effect on routing.  This pure-ASGI middleware intercepts every request
# before FastAPI sees it and dispatches MCP/SSE paths to the correct sub-app
# by looking it up in registry_instance.sub_apps at request time.
# ---------------------------------------------------------------------------

class DynamicMCPDispatcher:
    """
    Routes  /{server_name}/mcp  and  /{server_name}/sse  directly to the
    matching FastMCP sub-app ASGI callable.  Everything else is passed
    through to the inner FastAPI app.
    """

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.inner_app(scope, receive, send)
            return

        path = scope.get("path", "")
        path_parts = path.strip("/").split("/")
        server_name = path_parts[0] if path_parts else ""

        # Registry-internal routes go straight to FastAPI
        if not server_name or server_name in REGISTRY_ROUTES:
            await self.inner_app(scope, receive, send)
            return

        sub_app = registry_instance.sub_apps.get(server_name)
        if sub_app is None:
            await self.inner_app(scope, receive, send)
            return

        remaining = "/" + "/".join(path_parts[1:]) if len(path_parts) > 1 else "/"

        # Only hand MCP/SSE sub-paths to the sub-app; other sub-paths fall
        # through to the NonMCPProxyMiddleware (outer layer)
        if not remaining.lstrip("/").startswith(("mcp", "sse")):
            await self.inner_app(scope, receive, send)
            return

        # Rewrite scope so the sub-app sees itself rooted at /
        child_scope = dict(scope)
        child_scope["path"] = remaining
        child_scope["root_path"] = scope.get("root_path", "") + f"/{server_name}"

        logger.debug(f"[Dispatcher] {server_name}{remaining} -> sub_app")
        await sub_app(child_scope, receive, send)


# ---------------------------------------------------------------------------
# Layer 2 (outermost ASGI wrapper): NonMCPProxyMiddleware
#
# Handles non-MCP/SSE sub-paths for known servers by forwarding them via
# httpx to the upstream's base URL (e.g. custom REST endpoints on the
# upstream container).  Everything else is passed inward.
# ---------------------------------------------------------------------------

class NonMCPProxyMiddleware:
    """
    Proxies  /{server_name}/<anything except mcp|sse>  to the upstream server
    via httpx.  MCP/SSE paths are intentionally skipped here and handled by
    DynamicMCPDispatcher (the inner layer).
    """

    HOP_BY_HOP = frozenset([
        "connection", "keep-alive", "transfer-encoding", "te",
        "trailer", "proxy-authorization", "proxy-authenticate", "upgrade",
    ])

    def __init__(self, inner_app):
        self.inner_app = inner_app

    def _resolve_base_url(self, server_config) -> str | None:
        if server_config.endpoint:
            base = server_config.endpoint.rstrip("/")
            for suffix in ("/mcp", "/sse"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            return base
        elif server_config.port:
            return f"http://{registry_instance.host_ip}:{server_config.port}"
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.inner_app(scope, receive, send)
            return

        path = scope.get("path", "")
        path_parts = path.strip("/").split("/")
        server_name = path_parts[0] if path_parts else ""

        if not server_name or server_name in REGISTRY_ROUTES:
            await self.inner_app(scope, receive, send)
            return

        remaining = "/".join(path_parts[1:])

        # MCP/SSE paths are handled by the inner DynamicMCPDispatcher
        if remaining.startswith(("mcp", "sse")):
            await self.inner_app(scope, receive, send)
            return

        if not registry_instance.config:
            await self.inner_app(scope, receive, send)
            return

        server_config = registry_instance.config.servers.get(server_name)
        if not server_config:
            await self.inner_app(scope, receive, send)
            return

        base_url = self._resolve_base_url(server_config)
        if not base_url:
            resp = Response(status_code=503, content="Server configuration invalid")
            await resp(scope, receive, send)
            return

        query = scope.get("query_string", b"").decode()
        target_url = f"{base_url}/{remaining}" if remaining else base_url
        if query:
            target_url = f"{target_url}?{query}"

        raw_headers = {
            k.decode(): v.decode()
            for k, v in scope.get("headers", [])
            if k.decode().lower() not in self.HOP_BY_HOP
        }

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        method = scope.get("method", "GET")
        logger.debug(f"[NonMCPProxy] {method} {path} -> {target_url}")

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            try:
                upstream = await client.request(
                    method=method,
                    url=target_url,
                    headers=raw_headers,
                    content=body,
                )
                resp = Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    headers=dict(upstream.headers),
                )
            except httpx.RequestError as e:
                logger.error(f"[NonMCPProxy] Error -> {target_url}: {e}")
                resp = Response(status_code=502, content=f"Proxy error: {str(e)}")

        await resp(scope, receive, send)


# ---------------------------------------------------------------------------
# Final ASGI app exported to uvicorn
#
# Request flow:
#   uvicorn
#     -> NonMCPProxyMiddleware   (handles /{server}/non-mcp-path via httpx)
#     -> DynamicMCPDispatcher    (handles /{server}/mcp|sse via sub-app ASGI)
#     -> FastAPI                 (handles /health /register /info etc.)
# ---------------------------------------------------------------------------
app = NonMCPProxyMiddleware(DynamicMCPDispatcher(_fastapi_app))