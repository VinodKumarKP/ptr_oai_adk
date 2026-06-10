import argparse
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    Limiter = None
    get_remote_address = None
    RateLimitExceeded = None
    _rate_limit_exceeded_handler = None

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
from oai_agent_server.routers.admin import create_admin_router
from oai_agent_server.security.dependencies import verify_api_key, api_key_header
from oai_agent_server.services.agent_service import AgentService
from oai_agent_server.services.chat_service import ChatService
from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.services.logging_service import LoggingService
from oai_agent_server.services.token_service import TokenService
from oai_agent_server.utils.database_logger import DatabaseLogger

try:
    from oai_agent_server.a2a.database_task_store import (
        _LazyTaskStoreProxy,
        build_task_store_from_env,
    )
    _A2A_TASK_STORE_AVAILABLE = True
except ImportError:
    _LazyTaskStoreProxy = None  # type: ignore[assignment]
    build_task_store_from_env = None  # type: ignore[assignment]
    _A2A_TASK_STORE_AVAILABLE = False

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
        self.is_agent_ready = True


def _configure_structured_logging() -> None:
    """Configure root logger format based on LOG_FORMAT env (text|json).

    Always attaches RequestIdFilter so log lines emitted during request
    handling carry the request_id (matches the X-Request-ID response header).
    """
    from oai_agent_server.utils.logging_filter import RequestIdFilter

    request_id_filter = RequestIdFilter()
    log_format = os.environ.get("LOG_FORMAT", "text").lower()
    root = logging.getLogger()

    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler()
            # Including %(request_id)s in the format string makes JsonFormatter
            # auto-extract the field onto the JSON record.
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
            )
            handler.setFormatter(formatter)
            handler.addFilter(request_id_filter)
            root.handlers = [handler]
            root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
        except ImportError:
            # Fall through to text-style configuration below.
            log_format = "text"

    if log_format != "json":
        # Update existing handlers (or install one) with the request_id format.
        fmt = "%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s"
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt))
            root.addHandler(handler)
        for h in root.handlers:
            try:
                h.setFormatter(logging.Formatter(fmt))
            except Exception:
                pass

    # Attach the filter to the root logger (covers anything that propagates).
    root.addFilter(request_id_filter)
    for h in root.handlers:
        h.addFilter(request_id_filter)

    # Also wire uvicorn's loggers so request lines and errors carry the id.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.addFilter(request_id_filter)
        for h in lg.handlers:
            h.addFilter(request_id_filter)
            try:
                if log_format != "json":
                    h.setFormatter(logging.Formatter(
                        "%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s"
                    ))
            except Exception:
                pass


class AgentHTTPServer:
    """
    HTTP Server for hosting an AI Agent.

    Exposes two surfaces:
      /chat, /health, /agent, /logs, /tokens  — existing REST API (unchanged)
      /a2a                                     — A2A protocol (via a2a-sdk)
    """

    ALWAYS_ACTIVE_MODES = {"health", "agent", "chat", "logs", "a2a", "monitoring", "token", "readme"}

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

        # Lazy: created in startup() so import doesn't break if APScheduler missing.
        self.scheduler = None

        if enable_request_isolation:
            setup_request_isolation(self.logger)

        _configure_structured_logging()

        self.db_logger = DatabaseLogger(logger=self.logger)

        # A2A task store: proxy now, real store bound during startup() so the
        # router can be wired up before the DB backend is initialised.
        self._a2a_task_store_proxy = (
            _LazyTaskStoreProxy() if _A2A_TASK_STORE_AVAILABLE else None
        )
        self._a2a_task_store = None  # populated in startup()

        # Rate limiter (slowapi)
        if _SLOWAPI_AVAILABLE:
            self.limiter = Limiter(key_func=get_remote_address)
        else:
            self.limiter = None

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
        self.app.state.server_state = self.server_state
        self.app.state.db_logger = self.db_logger

        # Attach the limiter to the app state and register handler so endpoints
        # can use the @limiter.limit decorator.
        if self.limiter is not None:
            self.app.state.limiter = self.limiter
            self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
        _origins_env = os.environ.get("ALLOWED_ORIGINS", "")
        if _origins_env.strip() == "*":
            # Explicit opt-in for development only.
            allowed_origins = ["*"]
            allow_credentials = False  # Browsers reject credentials with wildcard
        else:
            allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
            allow_credentials = True

        if not allowed_origins:
            allowed_origins = ["http://localhost:3000"]

        self.logger.info(
            f"[CORS] Allowed origins: {allowed_origins} (allow_credentials={allow_credentials})"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(LoggingMiddleware, logger=self.logger)
        self.app.add_middleware(
            HeaderCaptureMiddleware,
            logger=self.logger,
            enable_request_isolation=self.enable_request_isolation,
        )
        # RequestTracking must be outermost (added LAST in Starlette) so the
        # request_id contextvar is set before any other middleware logs.
        self.app.add_middleware(RequestTrackingMiddleware, server_state=self.server_state)

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
                self.config_root, self.agent_name,
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
            logging_service=self.logging_service,
            a2a_base_url=self.a2a_base_url,
            a2a_streaming=self.a2a_streaming,
            a2a_push_notifications=self.a2a_push_notifications,
            agent_config=self.agent.agent_config,
            task_store=self._a2a_task_store_proxy,
        )
        if a2a_router:
            self.app.include_router(a2a_router, prefix="/a2a")
            self.a2a_agent_card = agent_card

        if os.environ.get("ENABLE_SCHEDULER", "true").lower() != "false":
            try:
                schedule_router = create_schedule_router(
                    self.agent, self.db_logger, self.allowed_modes
                )
                if schedule_router:
                    self.app.include_router(
                        schedule_router,
                        prefix="/schedule",
                    )
                    self.logger.info("Scheduler router enabled")
            except ImportError as e:
                self.logger.warning(
                    "Scheduler router skipped (apscheduler not installed): %s", e
                )
        else:
            self.logger.info(
                "Scheduler router disabled via ENABLE_SCHEDULER=false"
            )

        # Admin router — operator-only endpoints for inspecting the A2A
        # task store. Strict auth (no localhost bypass) is enforced inside
        # the router via Depends(verify_api_key_strict).
        self.app.include_router(create_admin_router())

    # ------------------------------------------------------------------
    # Startup / run (unchanged)
    # ------------------------------------------------------------------
    def _get_local_registry_url(self):
        """
        Get the local registry URL by parsing AGENT_BASE_URL and forcing localhost.
        Returns the local URL or None if the environment variable is not set.
        """
        agent_base_url = os.environ.get('AGENT_REGISTRY_URL') or os.environ.get("AGENT_BASE_URL")
        if not agent_base_url:
            return None

        try:
            parsed_url = urlparse(agent_base_url)
            port = parsed_url.port
            if not port:
                self.logger.warning(f"Could not extract port from AGENT_REGISTRY_URL '{agent_base_url}'. Using original URL.")
                return agent_base_url.rstrip('/')
            
            local_url = os.environ.get('AGENT_REGISTRY_URL') or f"http://localhost:{port}"
            self.logger.info(f"AGENT_REGISTRY_URL is set. Forcing registry connection to {local_url}")
            return local_url
        except Exception as e:
            self.logger.error(f"Failed to parse AGENT_REGISTRY_URL '{agent_base_url}': {e}")
            return None

    async def _register_with_registry(self):
        """Register the agent with the agent registry if AGENT_BASE_URL is set."""
        registry_base_url = self._get_local_registry_url()
        if not registry_base_url:
            self.logger.info("AGENT_BASE_URL not set, skipping registration.")
            return

        registry_url = f"{registry_base_url}/register"
        agent_info = {
            "name": self.agent_name,
            "description": self.agent.agent_config.get("description", "No description available"),
            "endpoint": f"http://localhost:{self.agent.agent_config.get('port', 8000)}",
            "port": self.agent.agent_config.get('port', 8000),
            "registered_via": "dynamic",
            "framework": self.agent.agent_type
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
                response = await client.post(registry_url, json=agent_info)
                if response.status_code == 200:
                    self.logger.info(f"Successfully registered agent '{self.agent_name}' with registry at {registry_base_url}")
                else:
                    self.logger.error(f"Failed to register agent using {registry_url}. Status: {response.status_code}, Response: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to agent registry at {registry_url}: {e}")

    async def _deregister_from_registry(self):
        """Deregister the agent from the agent registry."""
        registry_base_url = self._get_local_registry_url()
        if not registry_base_url:
            return

        registry_url = f"{registry_base_url}/deregister"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
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
            
            # Pre-initialize LLM judge service to eliminate first-request latency
            if "monitoring" in self.allowed_modes:
                try:
                    await self.llm_judge_service.initialize_judge_agent()
                    self.logger.info("LLM Judge service pre-initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to pre-initialize judge service: {e}; will initialize on first request")
            
            # Build the persistent A2A task store now that the DB backend is up.
            if (
                _A2A_TASK_STORE_AVAILABLE
                and self._a2a_task_store_proxy is not None
                and "a2a" in self.allowed_modes
            ):
                try:
                    self._a2a_task_store = await build_task_store_from_env(
                        self.db_logger, self.logger,
                    )
                    self._a2a_task_store_proxy.bind(self._a2a_task_store)
                    # Expose the real task store on app.state so admin
                    # endpoints can introspect it.
                    self.app.state.task_store = self._a2a_task_store
                except Exception:
                    self.logger.error(
                        "Failed to initialise A2A task store; "
                        "falling back to in-memory store.",
                        exc_info=True,
                    )
                    try:
                        from a2a.server.tasks import InMemoryTaskStore
                        self._a2a_task_store = InMemoryTaskStore()
                        self._a2a_task_store_proxy.bind(self._a2a_task_store)
                        self.app.state.task_store = self._a2a_task_store
                    except Exception:
                        self.logger.error(
                            "InMemoryTaskStore fallback also failed", exc_info=True,
                        )
            # Initialise APScheduler now so jobs can run as soon as the
            # server accepts requests (no race with first /schedule call).
            if os.environ.get("ENABLE_SCHEDULER", "true").lower() != "false":
                try:
                    from apscheduler.schedulers.asyncio import AsyncIOScheduler
                    self.scheduler = AsyncIOScheduler()
                    self.scheduler.start()
                    self.app.state.scheduler = self.scheduler
                    self.logger.info("Scheduler started")
                except ImportError:
                    self.logger.warning(
                        "APScheduler not installed; /schedule endpoints will return 503."
                    )
                    self.app.state.scheduler = None
                except Exception:
                    self.logger.error(
                        "Failed to start APScheduler; /schedule endpoints will return 503.",
                        exc_info=True,
                    )
                    self.app.state.scheduler = None
            else:
                self.logger.info(
                    "Scheduler disabled via ENABLE_SCHEDULER=false"
                )
                self.app.state.scheduler = None
            await self._register_with_registry()
        except Exception:
            self.logger.error(
                "Failed to initialize agent during startup", exc_info=True
            )
            self.server_state.is_agent_ready = False

    async def shutdown(self):
        self.logger.info("Shutting down agent server.")
        await self._deregister_from_registry()
        # Stop the A2A task-store cleanup loop before closing DB connections.
        try:
            if self._a2a_task_store is not None and hasattr(
                self._a2a_task_store, "shutdown"
            ):
                await self._a2a_task_store.shutdown()
        except Exception:
            self.logger.error("Failed to shutdown A2A task store", exc_info=True)
        # Stop scheduler before closing the DB so in-flight jobs don't
        # fire against a closed connection pool.
        if self.scheduler is not None:
            try:
                self.scheduler.shutdown(wait=False)
                self.logger.info("Scheduler shut down")
            except Exception:
                self.logger.error("Failed to shut down scheduler", exc_info=True)
        try:
            await self.db_logger.close()
        except Exception:
            self.logger.error("Failed to close database logger", exc_info=True)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        if self.a2a_agent_card and 'placeholder' in self.a2a_agent_card.supported_interfaces[0].url:
            # URL was not set by env var or config, so build it dynamically
            display_host = "localhost" if host == "0.0.0.0" else host
            base_url = f"http://{display_host}:{port}"
            self.a2a_agent_card.supported_interfaces[0].url = f"{base_url}/a2a/"
            self.a2a_agent_card.supported_interfaces[1].url = f"{base_url}/chat"
            self.a2a_agent_card.supported_interfaces[2].url = f"{base_url}/chat/stream"

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

    if os.environ.get('AGENT_REGISTRY_URL', None):
        parsed = urlparse(os.environ.get('AGENT_REGISTRY_URL'))
        os.environ['AGENT_BASE_URL'] = parsed.netloc.split(':')[0]
        os.environ['AGENT_BASE_URL_PORT'] = str(parsed.port)
    elif os.environ.get('AGENT_BASE_URL', None):
        os.environ['AGENT_REGISTRY_URL'] = f"{os.environ.get('AGENT_BASE_URL')}:{os.environ.get('AGENT_BASE_URL_PORT')}"

    server.run(host=args.host, port=port)


if __name__ == "__main__":
    main()