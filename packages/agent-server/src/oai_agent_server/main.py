import argparse
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, APIRouter

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

from oai_agent_core.components.configuration.model_config import ConfigManager
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.utils.logger import get_logger

from oai_agent_server.middleware.logging import LoggingMiddleware
from oai_agent_server.middleware.request_context import (
    setup_request_isolation, HeaderCaptureMiddleware,
)
from oai_agent_server.middleware.request_tracking import RequestTrackingMiddleware
from oai_agent_server.routers.agent import create_agent_router
from oai_agent_server.routers.chat import create_chat_router
from oai_agent_server.routers.health import create_health_router
from oai_agent_server.routers.logs import create_logs_router
from oai_agent_server.routers.tokens import create_token_router
from oai_agent_server.services.agent_service import AgentService
from oai_agent_server.services.chat_service import ChatService
from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.services.logging_service import LoggingService
from oai_agent_server.services.token_service import TokenService
from oai_agent_server.utils.database_logger import DatabaseLogger

file_root = os.path.dirname(os.path.abspath(__file__))
for path in [
    file_root,
    os.path.dirname(file_root),
    os.path.dirname(os.path.dirname(file_root)),
    os.path.dirname(os.path.dirname(os.path.dirname(file_root))),
]:
    if path not in sys.path:
        sys.path.append(path)


class ServerState:
    def __init__(self):
        self.active_requests = 0
        self.is_shutting_down = False
        self.shutdown_timeout = 30
        self.request_lock = threading.Lock()
        self.start_time = time.time()


class AgentHTTPServer:
    """
    HTTP Server for hosting an AI Agent.

    Exposes two surfaces:
      /chat, /health, /agent, /logs, /tokens  — existing REST API (unchanged)
      /a2a                                     — A2A protocol (via a2a-sdk)
    """

    def __init__(
        self,
        agent: BaseAgent,
        agent_name: str,
        config_root: str = None,
        enable_request_isolation: bool = True,
        allowed_modes: Optional[List[str]] = None,
        # A2A-specific options
        a2a_base_url: Optional[str] = None,
        a2a_streaming: bool = True,
        a2a_push_notifications: bool = True,
    ):
        self.agent = agent
        self.agent_name = agent.agent_name
        self.agent.session_id = str(uuid.uuid4())
        self.config_root = config_root
        self.logger = get_logger()
        self.base_config_manager = ConfigManager(config_root=config_root)
        self.enable_request_isolation = enable_request_isolation
        self.server_state = ServerState()

        ALWAYS_ACTIVE_MODES = {"health", "agent", "chat", "logs", "a2a"}
        self.allowed_modes = (
            list(ALWAYS_ACTIVE_MODES)
            if allowed_modes is None
            else list(set(allowed_modes).union(ALWAYS_ACTIVE_MODES))
        )

        self.a2a_base_url = a2a_base_url
        self.a2a_streaming = a2a_streaming
        self.a2a_push_notifications = a2a_push_notifications

        if enable_request_isolation:
            setup_request_isolation(self.logger)

        self.db_logger = DatabaseLogger(logger=self.logger)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield

        self.app = FastAPI(
            title=f"Agent HTTP Server - {agent_name}",
            description=f"HTTP API for {agent_name} agent",
            version="1.0.0",
            lifespan=lifespan,
        )
        self.app.state.agent_name = self.agent_name

        self._setup_middleware()
        self._setup_services()
        self._setup_routes()
        self._setup_a2a_sdk()   # ← new

    # ------------------------------------------------------------------
    # Middleware (unchanged)
    # ------------------------------------------------------------------

    def _setup_middleware(self):
        self.app.add_middleware(RequestTrackingMiddleware, server_state=self.server_state)
        self.app.add_middleware(LoggingMiddleware, logger=self.logger)
        self.app.add_middleware(
            HeaderCaptureMiddleware,
            logger=self.logger,
            enable_request_isolation=self.enable_request_isolation,
        )

    # ------------------------------------------------------------------
    # Services (unchanged)
    # ------------------------------------------------------------------

    def _setup_services(self):
        self.llm_judge_service = LLMJudgeService(
            agent_class=type(self.agent),
            config_root=self.config_root,
            logger=self.logger,
            db_logger=self.db_logger,
            judge_model_id=os.environ.get(
                "LLM_JUDGE_MODEL_ID", "bedrock/us.amazon.nova-micro-v1:0"
            ),
        )
        self.chat_service = ChatService(
            self.agent, self.db_logger, self.logger,
            self.llm_judge_service, self.allowed_modes,
        )
        self.agent_service = AgentService(
            self.agent, self.server_state, self.logger
        )
        self.logging_service = LoggingService(self.db_logger, self.agent_name)
        self.token_service = TokenService()

    # ------------------------------------------------------------------
    # Existing REST routes (unchanged)
    # ------------------------------------------------------------------

    def _setup_routes(self):
        self.app.include_router(
            create_chat_router(self.chat_service, self.allowed_modes)
        )
        self.app.include_router(
            create_agent_router(
                self.agent_service, self.enable_request_isolation, self.allowed_modes
            )
        )
        self.app.include_router(
            create_logs_router(self.logging_service, self.allowed_modes)
        )
        self.app.include_router(
            create_health_router(
                self.agent_name, self.server_state,
                self.enable_request_isolation, self.allowed_modes,
            )
        )
        self.app.include_router(
            create_token_router(self.token_service, self.allowed_modes)
        )

    # ------------------------------------------------------------------
    # A2A SDK  ← the only new method
    # ------------------------------------------------------------------

    def _setup_a2a_sdk(self):
        """
        Integrates the a2a-sdk by including its routes directly into FastAPI,
        making them visible in the OpenAPI (Swagger) documentation.
        """
        if "a2a" not in self.allowed_modes:
            self.logger.info("A2A mode not enabled — skipping SDK integration")
            return

        if not A2A_SDK_AVAILABLE:
            self.logger.warning(
                "A2A mode is enabled, but 'a2a-sdk' is not installed. "
                "Skipping A2A endpoint setup. "
                "Install with: pip install \"a2a-sdk[http-server]\""
            )
            return

        # 1. Build the agent card
        agent_card = build_agent_card(
            agent_name=self.agent_name,
            base_url=(
                self.a2a_base_url
                or os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
            ) + "/a2a",
            streaming=self.a2a_streaming,
            push_notifications=self.a2a_push_notifications,
        )

        # 2. Create your executor
        executor = BaseAgentExecutor(
            agent=self.agent,
            use_streaming=self.a2a_streaming,
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

        self.app.include_router(a2a_router, prefix="/a2a")
        self.logger.info(
            f"A2A SDK routes included at /a2a "
            f"(streaming={self.a2a_streaming}, "
            f"pushNotifications={self.a2a_push_notifications})"
        )

    # ------------------------------------------------------------------
    # Startup / run (unchanged)
    # ------------------------------------------------------------------

    async def startup(self):
        self.server_state.start_time = time.time()
        try:
            await self.agent.initialize()
            self.logger.info(f"Agent '{self.agent_name}' initialized successfully")
            await self.db_logger.initialize()
        except Exception as e:
            self.logger.info(
                f"Warning: Failed to initialize agent during startup: {e}"
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        self.logger.info(
            f"Starting Agent HTTP Server for '{self.app.title}' on {host}:{port}"
        )
        uvicorn.run(self.app, host=host, port=port)


def parse_args(optional_agent_name_flag=False):
    parser = argparse.ArgumentParser(description="Start Agent HTTP Server")
    if optional_agent_name_flag:
        parser.add_argument(
            "agent_name", default=None, nargs="?",
            help="Name of the agent to initialize",
        )
    else:
        parser.add_argument("agent_name", help="Name of the agent to initialize")
    parser.add_argument("--port", "-p", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--temperature", "-t", type=float)
    parser.add_argument("--max-tokens", "-m", type=int)
    parser.add_argument("--allowed-modes", nargs="+", default=None)
    return parser.parse_args()


def main(server: AgentHTTPServer):
    args = parse_args(optional_agent_name_flag=True)
    agent_name = server.agent_name
    config = server.base_config_manager.load_agent_config(
        agent_name=agent_name, abort_if_not_found=False
    )
    port = args.port or config.get("port", 8000)
    server.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()