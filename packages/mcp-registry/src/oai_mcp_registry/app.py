import logging
import warnings
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI, Request, Response
import httpx

from oai_mcp_registry.dependencies import registry_instance
from oai_mcp_registry.routers.registry import router

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mcp")
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MCPRegistry")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if registry_instance.registry_config.enable_auto_discovery:
        host = registry_instance.registry_config.host
        await registry_instance.discover_servers(host)

    for name, sub_app in registry_instance.sub_apps.items():
        app.mount(f"/{name}", sub_app)

    async with AsyncExitStack() as stack:
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

app = FastAPI(
    title="MCP Gateway",
    description="Unified Proxy for Distributed MCP Servers",
    lifespan=lifespan
)

app.include_router(router)

@app.middleware("http")
async def proxy_middleware(request: Request, call_next):
    path_parts = request.url.path.strip("/").split("/")
    if not path_parts or not path_parts[0]:
        return await call_next(request)

    server_name = path_parts[0]
    remaining_path = "/".join(path_parts[1:])

    if server_name not in registry_instance.sub_apps or remaining_path.startswith(("mcp", "sse")):
        return await call_next(request)

    server_config = registry_instance.config.servers[server_name]
    if server_config.endpoint:
        base_url = server_config.endpoint.rstrip("/mcp").rstrip("/sse").rstrip("/")
    elif server_config.port:
        base_url = f"http://{registry_instance.host_ip}:{server_config.port}"
    else:
        return Response(status_code=503, content="Server configuration invalid")

    target_url = f"{base_url}/{remaining_path}" if remaining_path else base_url
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=dict(request.headers),
                content=await request.body(),
                timeout=30
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        except httpx.RequestError as e:
            return Response(status_code=502, content=f"Proxy error: {str(e)}")
