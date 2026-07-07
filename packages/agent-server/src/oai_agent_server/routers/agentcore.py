"""AWS Bedrock AgentCore Runtime HTTP protocol contract endpoints.

Implements the AgentCore Runtime service contract for the HTTP protocol:

  POST /invocations  — primary invocation endpoint (JSON in, JSON or SSE out)
  GET  /ping         — platform health probe (Healthy / HealthyBusy)

The endpoints are thin adapters over the existing ChatService so an agent
deployed to AgentCore behaves identically to one served via /chat and
/chat/stream. Enabled via the ``agentcore`` mode.

Contract reference:
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html
"""
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from oai_agent_core.utils.logger import get_logger

from oai_agent_server.models.requests import ChatRequest, StreamChatRequest


class InvocationRequest(BaseModel):
    """Payload for POST /invocations.

    AgentCore forwards the caller's JSON payload as-is. ``prompt`` is the
    conventional field from the contract examples; ``message``/``input`` are
    accepted as aliases so existing /chat clients can be pointed at the
    AgentCore data plane without payload changes.
    """
    prompt: Optional[Any] = Field(None, description="User input (contract-conventional field)")
    message: Optional[Any] = Field(None, description="Alias for prompt (chat-API compatible)")
    input: Optional[Any] = Field(None, description="Alias for prompt")
    session_id: Optional[str] = Field(
        None,
        description=(
            "Optional session id. On AgentCore the "
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id header (bound to "
            "request.state) takes precedence."
        ),
    )
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    stream: bool = Field(False, description="If true, respond with SSE streaming")

    def resolve_message(self) -> Any:
        for value in (self.prompt, self.message, self.input):
            if value is not None:
                return value
        return None


def create_agentcore_router(
    chat_service,
    server_state,
    allowed_modes: List[str],
) -> Optional[Tuple[APIRouter, Any]]:
    """Build the AgentCore HTTP-contract router, or None when mode disabled."""
    logger = get_logger()

    if "agentcore" not in allowed_modes:
        return None

    router = APIRouter(tags=["agentcore"])

    # Dedicated in-flight counter for /ping status. server_state.active_requests
    # cannot be used: the /ping request itself (and unrelated health probes)
    # increment it, which would make the runtime consider the session busy
    # forever. Only real invocations should hold the session active.
    state_lock = server_state.request_lock

    def _adjust_invocations(delta: int) -> None:
        with state_lock:
            current = getattr(server_state, "active_invocations", 0)
            server_state.active_invocations = max(0, current + delta)

    @router.post("/invocations")
    async def invocations(
        request: Request,
        payload: InvocationRequest,
        background_tasks: BackgroundTasks,
    ):
        """AgentCore primary invocation endpoint (JSON or SSE response)."""
        user_message = payload.resolve_message()
        if user_message is None:
            raise HTTPException(
                status_code=400,
                detail="Payload must include 'prompt' (or 'message'/'input')",
            )

        _adjust_invocations(+1)
        try:
            if payload.stream:
                stream_request = StreamChatRequest(
                    message=user_message,
                    session_id=payload.session_id,
                    user_id=payload.user_id,
                )
                response = await chat_service.process_stream_chat(
                    http_request=request,
                    stream_request=stream_request,
                    background_tasks=background_tasks,
                    headers=request.headers,
                )
            else:
                chat_request = ChatRequest(
                    message=user_message,
                    session_id=payload.session_id,
                    user_id=payload.user_id,
                )
                response = await chat_service.process_chat(
                    http_request=request,
                    chat_request=chat_request,
                    background_tasks=background_tasks,
                    headers=request.headers,
                )
            return response
        finally:
            # NOTE: for streaming responses the generator may outlive this
            # handler; the counter guards the request handling window, while
            # the platform's own session tracking covers stream delivery.
            _adjust_invocations(-1)

    @router.get("/ping")
    async def ping() -> JSONResponse:
        """AgentCore health probe.

        Healthy      — ready for new work
        HealthyBusy  — operational but processing invocations (keeps the
                       runtime session active for long-running work)
        503          — agent failed to initialize
        """
        if not server_state.is_agent_ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "Unhealthy",
                    "time_of_last_update": int(time.time()),
                },
            )

        busy = getattr(server_state, "active_invocations", 0) > 0
        return JSONResponse(
            content={
                "status": "HealthyBusy" if busy else "Healthy",
                "time_of_last_update": int(time.time()),
            }
        )

    logger.info("AgentCore HTTP contract routes enabled (/invocations, /ping)")
    return router
