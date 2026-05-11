from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking active requests and handling graceful shutdown."""

    def __init__(self, app, server_state):
        super().__init__(app)
        self.server_state = server_state

    async def dispatch(self, request: Request, call_next):
        """Track active requests and enforce shutdown state"""
        # Note: /restart and /kill are tracked too — they are high-value
        # audit events.

        # Check if shutting down
        if self.server_state.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is shutting down, please try again after restart", "status": "shutting_down"}
            )

        # Increment active requests
        with self.server_state.request_lock:
            self.server_state.active_requests += 1

        try:
            response = await call_next(request)
            return response
        finally:
            # Decrement active requests
            with self.server_state.request_lock:
                self.server_state.active_requests -= 1
