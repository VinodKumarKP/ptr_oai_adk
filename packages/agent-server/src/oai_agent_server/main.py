import argparse
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List

import httpx
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

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
from oai_agent_server.routers.a2a import create_a2a_router
from oai_agent_server.routers.scheduler import create_schedule_router
from oai_agent_server.security.dependencies import verify_api_key, api_key_header
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

    ALWAYS_ACTIVE_MODES = {"health", "agent", "chat", "logs", "a2a"}

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
        self.a2a_agent_card = None

        self.allowed_modes = (
            list(self.ALWAYS_ACTIVE_MODES)
            if allowed_modes is None
            else list(set(allowed_modes).union(self.ALWAYS_ACTIVE_MODES))
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
            await self.shutdown()

        self.app = FastAPI(
            title=f"Agent HTTP Server - {agent_name}",
            description=f"HTTP API for {agent_name} agent",
            version="1.0.0",
            lifespan=lifespan,
            dependencies=[Depends(verify_api_key)],
            security=[{api_key_header.model.name: []}],
        )
        self.app.state.agent_name = self.agent_name

        self._setup_middleware()
        self._setup_services()
        self._setup_routes()

    # ------------------------------------------------------------------
    # Middleware (unchanged)
    # ------------------------------------------------------------------

    def set_allowed_modes(self, allowed_modes: List[str]=None):
        self.allowed_modes = (
            list(self.ALWAYS_ACTIVE_MODES)
            if allowed_modes is None
            else list(set(allowed_modes).union(self.ALWAYS_ACTIVE_MODES))
        )
        self._setup_routes()


    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
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
        a2a_router, agent_card = create_a2a_router(
            agent=self.agent,
            agent_name=self.agent_name,
            allowed_modes=self.allowed_modes,
            db_logger=self.db_logger,
            llm_judge_service=self.llm_judge_service,
            a2a_base_url=self.a2a_base_url,
            a2a_streaming=self.a2a_streaming,
            a2a_push_notifications=self.a2a_push_notifications,
            agent_config=self.agent.agent_config,
        )
        if a2a_router:
            self.app.include_router(a2a_router, prefix="/a2a")
            self.a2a_agent_card = agent_card

        schedule_router = create_schedule_router(self.agent, self.db_logger, self.allowed_modes)
        if schedule_router:
            self.app.include_router(
                schedule_router,
                prefix="/schedule",
            )

    # ------------------------------------------------------------------
    # Startup / run (unchanged)
    # ------------------------------------------------------------------
    async def _register_with_registry(self):
        """Register the agent with the agent registry if AGENT_BASE_URL is set."""
        agent_base_url = os.environ.get("AGENT_BASE_URL")
        if not agent_base_url:
            self.logger.info("AGENT_BASE_URL not set, skipping registration.")
            return

        registry_url = f"{agent_base_url.rstrip('/')}/register"
        agent_info = {
            "name": self.agent_name,
            "description": self.agent.agent_config.get("description", "No description available"),
            "endpoint": f"http://localhost:{self.agent.agent_config.get('port', 8000)}",
            "agent_class": self.agent.agent_config.get("agent_class"),
            "tools": [tool for tool in self.agent.agent_config.get("tools", [])],
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(registry_url, json=agent_info)
                if response.status_code == 200:
                    self.logger.info(f"Successfully registered agent '{self.agent_name}' with registry at {agent_base_url}")
                else:
                    self.logger.error(f"Failed to register agent. Status: {response.status_code}, Response: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to agent registry at {registry_url}: {e}")

    async def _deregister_from_registry(self):
        """Deregister the agent from the agent registry."""
        agent_base_url = os.environ.get("AGENT_BASE_URL")
        if not agent_base_url:
            return

        registry_url = f"{agent_base_url.rstrip('/')}/deregister"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(registry_url, json={"name": self.agent_name})
                if response.status_code == 200:
                    self.logger.info(f"Successfully deregistered agent '{self.agent_name}' from registry.")
                else:
                    self.logger.error(f"Failed to deregister agent. Status: {response.status_code}, Response: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to agent registry at {registry_url}: {e}")

    async def startup(self):
        self.server_state.start_time = time.time()
        try:
            await self.agent.initialize()
            self.logger.info(f"Agent '{self.agent_name}' initialized successfully")
            await self.db_logger.initialize()
            await self._register_with_registry()
        except Exception as e:
            self.logger.info(
                f"Warning: Failed to initialize agent during startup: {e}"
            )

    async def shutdown(self):
        self.logger.info("Shutting down agent server.")
        await self._deregister_from_registry()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        if self.a2a_agent_card and "placeholder.url" in self.a2a_agent_card.url:
            # URL was not set by env var or config, so build it dynamically
            display_host = "localhost" if host == "0.0.0.0" else host
            base_url = f"http://{display_host}:{port}"
            self.a2a_agent_card.url = f"{base_url}/a2a/"
            self.a2a_agent_card.additional_interfaces[0].url = f"{base_url}/chat"
            self.a2a_agent_card.additional_interfaces[1].url = f"{base_url}/chat/stream"

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
    server.agent.agent_config['port'] = port
    # server.set_allowed_modes(args.allowed_modes)
    server.run(host=args.host, port=port)


if __name__ == "__main__":
    main()
