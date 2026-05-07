import logging
import warnings
from contextlib import asynccontextmanager, AsyncExitStack
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry_instance.initialize()

    if registry_instance.registry_config.enable_auto_discovery:
        host = registry_instance.registry_config.host
        await registry_instance.discover_servers(host)

    # Mount servers known at startup so FastAPI's static router covers them.
    # Servers that register dynamically later are handled by DynamicMCPDispatcher.
    for name, sub_app in registry_instance.sub_apps.items():
        app.mount(f"/{name}", sub_app)

    async with AsyncExitStack() as stack:
        registry_instance.exit_stack = stack
        logger.info("--- Starting Upstream Connections ---")
        active_count = 0
        for name, sub_app in registry_instance.sub_apps.items():
            try:
                await stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
                logger.info(f"  [Connected] {name}")
                active_count += 1
            except Exception as e:
                logger.error(f"  [Connection Error] {name}: {e}")
        logger.info(f"--- Proxy Ready: {active_count}/{len(registry_instance.sub_apps)} upstreams active ---")
        yield
        logger.info("--- Shutting Down ---")
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