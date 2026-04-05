import os
from typing import List, Optional

from fastapi import APIRouter
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.utils.logger import get_logger

try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from oai_agent_server.a2a.agent_card import build_agent_card
    from oai_agent_server.a2a.agent_executor import BaseAgentExecutor
    A2A_SDK_AVAILABLE = True
except ImportError:
    A2AStarletteApplication = None
    DefaultRequestHandler = None
    InMemoryTaskStore = None
    build_agent_card = None
    BaseAgentExecutor = None
    A2A_SDK_AVAILABLE = False


def create_a2a_router(
    agent: BaseAgent,
    agent_name: str,
    allowed_modes: List[str],
    a2a_base_url: Optional[str] = None,
    a2a_streaming: bool = True,
    a2a_push_notifications: bool = True,
) -> Optional[APIRouter]:
    logger = get_logger()

    if "a2a" not in allowed_modes:
        logger.info("A2A mode not enabled — skipping SDK integration")
        return None

    if not A2A_SDK_AVAILABLE:
        logger.warning(
            "A2A mode is enabled, but 'a2a-sdk' is not installed. "
            "Skipping A2A endpoint setup. "
            "Install with: pip install \"a2a-sdk[http-server]\""
        )
        return None

    # 1. Build the agent card
    agent_card = build_agent_card(
        agent_name=agent_name,
        base_url=(
            a2a_base_url
            or os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
        ) + "/a2a",
        streaming=a2a_streaming,
        push_notifications=a2a_push_notifications,
    )

    # 2. Create your executor
    executor = BaseAgentExecutor(
        agent=agent,
        use_streaming=a2a_streaming,
    )

    # 3. Wire up the SDK's request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # 4. Build the a2a ASGI app. It's a Starlette app with the routes.
    a2a_asgi_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    # 5. Instead of mounting (which hides routes from OpenAPI), create a
    #    router and copy the routes from the generated a2a app.
    a2a_router = APIRouter()
    for route in a2a_asgi_app.routes:
        if hasattr(route, "methods"):  # Exclude websockets, mounts, etc.
            a2a_router.add_api_route(
                path=route.path,
                endpoint=route.endpoint,
                methods=list(route.methods),
                tags=["a2a"],  # Group in Swagger UI
                include_in_schema=True,
            )

    logger.info(
        f"A2A SDK routes included at /a2a "
        f"(streaming={a2a_streaming}, "
        f"pushNotifications={a2a_push_notifications})"
    )
    return a2a_router