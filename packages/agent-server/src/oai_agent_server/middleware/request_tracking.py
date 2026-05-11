import re
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

# Module-level contextvar so log records emitted during request handling
# (across async tasks) can pick up the active request_id without plumbing.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# Loose UUID match (accepts UUIDs of any version, hex with optional dashes).
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _coerce_request_id(value: str | None) -> str:
    if value and _UUID_RE.match(value.strip()):
        return value.strip()
    return str(uuid4())


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking active requests, propagating request IDs,
    and handling graceful shutdown."""

    def __init__(self, app, server_state):
        super().__init__(app)
        self.server_state = server_state

    async def dispatch(self, request: Request, call_next):
        """Track active requests, set request_id contextvar, enforce shutdown state."""
        request_id = _coerce_request_id(request.headers.get("X-Request-ID"))
        token = request_id_ctx.set(request_id)
        request.state.request_id = request_id

        try:
            # Check if shutting down
            if self.server_state.is_shutting_down:
                resp = JSONResponse(
                    status_code=503,
                    content={"detail": "Server is shutting down, please try again after restart", "status": "shutting_down"},
                )
                resp.headers["X-Request-ID"] = request_id
                return resp

            # Increment active requests
            with self.server_state.request_lock:
                self.server_state.active_requests += 1

            try:
                response: Response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            finally:
                # Decrement active requests
                with self.server_state.request_lock:
                    self.server_state.active_requests -= 1
        finally:
            request_id_ctx.reset(token)
