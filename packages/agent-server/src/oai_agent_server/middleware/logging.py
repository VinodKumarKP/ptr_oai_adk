import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request duration."""

    def __init__(self, app, logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):
        """Log request method, path, and processing time."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        self.logger.info(f"Request: {request.method} {request.url.path} completed in {process_time:.4f}s")
        response.headers["X-Response-Time"] = f"{process_time:.2f}s"
        return response
