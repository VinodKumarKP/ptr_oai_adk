import argparse
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List

import uvicorn
from fastapi import FastAPI

from oai_agent_core.components.configuration.model_config import ConfigManager
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.utils.logger import get_logger
from oai_agent_server.middleware.logging import LoggingMiddleware
from oai_agent_server.middleware.request_context import setup_request_isolation, HeaderCaptureMiddleware
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

# Ensure paths are correct
file_root = os.path.dirname(os.path.abspath(__file__))
path_list = [
    file_root,
    os.path.dirname(file_root),
    os.path.dirname(os.path.dirname(file_root)),
    os.path.dirname(os.path.dirname(os.path.dirname(file_root)))
]
for path in path_list:
    if path not in sys.path:
        sys.path.append(path)


class ServerState:
    """Tracks the state of the server, including active requests and shutdown status."""

    def __init__(self):
        self.active_requests = 0
        self.is_shutting_down = False
        self.shutdown_timeout = 30  # seconds
        self.request_lock = threading.Lock()
        self.start_time = time.time()


class AgentHTTPServer:
    """
    HTTP Server for hosting an AI Agent.

    This class wraps a BaseAgent instance with a FastAPI server, providing
    endpoints for chat, health checks, logs, and agent information.
    It handles middleware setup, service initialization, and lifecycle management.
    """

    def __init__(self, agent: BaseAgent,
                 agent_name: str,
                 config_root: str = None,
                 enable_request_isolation: bool = True,
                 allowed_modes: Optional[List[str]] = None):
        """
        Initialize the Agent HTTP Server.

        Args:
            agent: The BaseAgent instance to host.
            agent_name: Name of the agent.
            config_root: Root directory for configuration files.
            enable_request_isolation: Whether to enable request context isolation.
            allowed_modes: List of allowed API modes (e.g., ["chat", "health"]).
        """
        self.agent = agent
        self.agent_name = agent.agent_name
        self.agent.session_id = str(uuid.uuid4())
        self.config_root = config_root

        self.logger = get_logger()
        self.base_config_manager = ConfigManager(config_root=config_root)
        self.enable_request_isolation = enable_request_isolation
        self.server_state = ServerState()

        # Define modes that are always active
        ALWAYS_ACTIVE_MODES = {"health", "agent", "chat", "logs", "monitoring"}

        if allowed_modes is None:
            # If no modes are explicitly provided, use a comprehensive default list
            # including always active modes and other common modes.
            self.allowed_modes = list(ALWAYS_ACTIVE_MODES)
        else:
            # If modes are explicitly provided, ensure always active modes are included.
            self.allowed_modes = list(set(allowed_modes).union(ALWAYS_ACTIVE_MODES))

        # Setup request-aware environment if enabled
        if enable_request_isolation:
            setup_request_isolation(self.logger)

        self.db_logger = DatabaseLogger(logger=self.logger)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown (if needed)
            pass

        self.app = FastAPI(
            title=f"Agent HTTP Server - {agent_name}",
            description=f"HTTP API for {agent_name} agent",
            version="1.0.0",
            lifespan=lifespan
        )

        # Store agent_name in app state for dependencies
        self.app.state.agent_name = self.agent_name

        self._setup_middleware()
        self._setup_services()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup all middleware in the correct order"""
        # Authentication is now handled by Depends() in routers

        # 1. Request Tracking (Outer layer - tracks active requests)
        self.app.add_middleware(RequestTrackingMiddleware, server_state=self.server_state)

        # 2. Logging (Logs request duration)
        self.app.add_middleware(LoggingMiddleware, logger=self.logger)

        # 3. Header Capture (Inner layer - sets up context for the request)
        self.app.add_middleware(HeaderCaptureMiddleware, logger=self.logger,
                                enable_request_isolation=self.enable_request_isolation)

    def _setup_services(self):
        """Initialize internal services."""
        self.llm_judge_service = LLMJudgeService(
            agent_class=type(self.agent),
            config_root=self.config_root,
            logger=self.logger,
            db_logger=self.db_logger,
            judge_model_id=os.environ.get('LLM_JUDGE_MODEL_ID', 'bedrock/us.amazon.nova-micro-v1:0')
        )
        self.chat_service = ChatService(self.agent, self.db_logger, self.logger, self.llm_judge_service,
                                        self.allowed_modes)
        self.agent_service = AgentService(self.agent, self.server_state, self.logger)
        self.logging_service = LoggingService(self.db_logger, self.agent_name)
        self.token_service = TokenService()

    def _setup_routes(self):
        """Setup all API routes"""
        # Pass allowed_modes to each router creator
        self.app.include_router(create_chat_router(self.chat_service, self.allowed_modes))
        self.app.include_router(
            create_agent_router(self.agent_service, self.enable_request_isolation, self.allowed_modes))
        self.app.include_router(create_logs_router(self.logging_service, self.allowed_modes))
        self.app.include_router(
            create_health_router(self.agent_name, self.server_state, self.enable_request_isolation, self.allowed_modes))
        self.app.include_router(
            create_token_router(self.token_service, self.allowed_modes))

    async def startup(self):
        """Initialize the agent on server startup"""
        self.server_state.start_time = time.time()
        try:
            await self.agent.initialize()
            self.logger.info(f"Agent '{self.agent_name}' initialized successfully")

            # Initialize database logger
            await self.db_logger.initialize()
        except Exception as e:
            self.logger.info(f"Warning: Failed to initialize agent during startup: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        self.logger.info(f"Starting Agent HTTP Server for '{self.app.title}' on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def parse_args(optional_agent_name_flag=False):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start Agent HTTP Server")
    if optional_agent_name_flag:
        parser.add_argument("agent_name", help="Name of the agent to initialize", default=None, nargs='?')
    else:
        parser.add_argument("agent_name", help="Name of the agent to initialize")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port to run server on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to (default: 0.0.0.0)")
    parser.add_argument("--temperature", "-t", type=float, help="Temperature for the agent")
    parser.add_argument("--max-tokens", "-m", type=int, help="Maximum tokens for the agent")
    parser.add_argument("--allowed-modes", nargs='+', default=None,
                        help="List of allowed modes (chat, logs, token, monitoring). Health and agent modes are always active.")

    args = parser.parse_args()
    return args


def main(server: AgentHTTPServer):
    """Main entry point for the server."""
    args = parse_args(optional_agent_name_flag=True)
    agent_name = server.agent_name
    host = "0.0.0.0"
    port = args.port

    config = server.base_config_manager.load_agent_config(agent_name=agent_name, abort_if_not_found=False)
    port = port if port else config.get("port", 8000)
    server.run(host=host, port=port)


if __name__ == "__main__":
    main()
