from typing import List, Optional, Dict, Any, Tuple, Literal, Union

from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes, create_rest_routes
from fastapi import APIRouter, Query
from starlette.applications import Starlette
from starlette.routing import Route

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.utils.logger import get_logger
from pydantic import BaseModel, Field

from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.services.logging_service import LoggingService

try:
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from oai_agent_server.a2a.agent_card import build_agent_card, AgentCard
    from oai_agent_server.a2a.agent_executor import BaseAgentExecutor
    A2A_SDK_AVAILABLE = True
except ImportError:
    DefaultRequestHandler = None
    InMemoryTaskStore = None
    build_agent_card = None
    BaseAgentExecutor = None
    AgentCard = None
    A2A_SDK_AVAILABLE = False

# Pydantic model for a JSON-RPC request to improve OpenAPI schema
class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = Field(default="2.0", description="JSON-RPC version")
    method: str = Field(..., description="The name of the method to be invoked.")
    params: Optional[Union[List[Any], Dict[str, Any]]] = Field(None, description="Parameters for the method.")
    id: Optional[Union[str, int]] = Field(None, description="Request identifier.")

    class Config:
        json_schema_extra = {
            "example": {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "tool_name": "my_tool",
                    "prompt": "What is the weather in SF?",
                },
                "id": "1",
            }
        }


def create_a2a_router(
    agent: BaseAgent,
    agent_name: str,
    allowed_modes: List[str],
    db_logger,
    llm_judge_service: LLMJudgeService,
    logging_service: LoggingService,
    a2a_base_url: Optional[str] = None,
    a2a_streaming: bool = True,
    a2a_push_notifications: bool = True,
    agent_config: Optional[Dict[str, Any]] = None,
    task_store: Any = None,
) -> Optional[Tuple[APIRouter, AgentCard]]:
    logger = get_logger()

    if "a2a" not in allowed_modes:
        logger.info("A2A mode not enabled — skipping SDK integration")
        return None, None

    if not A2A_SDK_AVAILABLE:
        logger.warning(
            "A2A mode is enabled, but 'a2a-sdk' is not installed. "
            "Skipping A2A endpoint setup. "
            "Install with: pip install \"a2a-sdk[http-server]\""
        )
        return None, None

    agent_card = build_agent_card(
        agent_name=agent_name,
        base_url=a2a_base_url,
        streaming=a2a_streaming,
        push_notifications=a2a_push_notifications,
        agent_config=agent_config,
    )

    executor = BaseAgentExecutor(
        agent=agent,
        db_logger=db_logger,
        llm_judge_service=llm_judge_service,
        allowed_modes=allowed_modes,
        use_streaming=a2a_streaming,
        agent_card=agent_card,
    )

    # Use injected task store (DatabaseTaskStore or InMemoryTaskStore).
    # Falls back to InMemoryTaskStore if none was provided (e.g. tests).
    effective_task_store = task_store if task_store is not None else InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=effective_task_store,
        agent_card=agent_card,
    )

    a2a_router = APIRouter()

    # Manually add the routes from the SDK to the FastAPI router
    # This provides better integration with FastAPI's docs and context
    card_routes = create_agent_card_routes(agent_card)
    for route in card_routes:
        a2a_router.add_api_route(route.path, route.endpoint, methods=set(route.methods), tags=["a2a"])

    # Set rpc_url to "/" because the prefix is handled by the router inclusion in main.py
    jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url="/", enable_v0_3_compat=True)
    for route in jsonrpc_routes:
        if hasattr(route, "methods"):  # Exclude websockets, mounts, etc.
            openapi_extra = None
            # The JSON-RPC endpoint is a POST to the root of the mounted app.
            # We manually add the request body to the OpenAPI schema here.
            if route.path == '/' and 'POST' in route.methods:
                openapi_extra = {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": JsonRpcRequest.model_json_schema()
                            }
                        },
                        "required": True
                    }
                }

            a2a_router.add_api_route(
                path=route.path,
                endpoint=route.endpoint,
                methods=list(route.methods),
                tags=["a2a"],  # Group in Swagger UI
                include_in_schema=True,
                openapi_extra=openapi_extra,
            )

    if "logs" in allowed_modes:
        @a2a_router.get("/logs", tags=["a2a", "logs"])
        async def get_logs(
                session_id: Optional[str] = Query(None),
                user_id: Optional[str] = Query(None),
                endpoint: Optional[str] = Query(None),
                status: Optional[str] = Query(None),
                limit: int = Query(100, ge=1, le=1000),
                offset: int = Query(0, ge=0)
        ):
            return await logging_service.get_logs(session_id, user_id, endpoint, status, limit, offset)

        @a2a_router.get("/logs/interaction/{interaction_id}", tags=["a2a", "logs"])
        async def get_chat_log_by_interaction_id(interaction_id: str):
            return await logging_service.get_chat_log_by_interaction_id(interaction_id)

        @a2a_router.get("/logs/activity/interaction/{interaction_id}", tags=["a2a", "logs"])
        async def get_activity_logs_by_interaction_id(interaction_id: str):
            return await logging_service.get_activity_logs_by_interaction_id(interaction_id)

        @a2a_router.get("/logs/sessions/{session_id}", tags=["a2a", "logs"])
        async def get_session_logs(session_id: str):
            return await logging_service.get_session_logs(session_id)

        @a2a_router.get("/logs/stats", tags=["a2a", "logs"])
        async def get_log_stats(user_id: Optional[str] = Query(None)):
            return await logging_service.get_log_stats(user_id=user_id)

        @a2a_router.get("/logs/stats/users", tags=["a2a", "logs"])
        async def get_user_stats():
            return await logging_service.get_user_stats()

    if "monitoring" in allowed_modes:
        @a2a_router.get("/evaluations/agent", tags=["a2a", "monitoring"])
        async def get_evaluations_by_agent_name():
            return await logging_service.get_evaluations_by_agent_name()

        @a2a_router.get("/evaluations/session/{session_id}", tags=["a2a", "monitoring"])
        async def get_evaluations_by_session_id(session_id: str):
            return await logging_service.get_evaluations_by_session_id(session_id)

        @a2a_router.get("/evaluations/{interaction_id}", tags=["a2a", "monitoring"])
        async def get_evaluation(interaction_id: str):
            return await logging_service.get_evaluation(interaction_id)

    logger.info(
        f"A2A SDK routes included at /a2a "
        f"(streaming={a2a_streaming}, "
        f"pushNotifications={a2a_push_notifications})"
    )
    return a2a_router, agent_card
