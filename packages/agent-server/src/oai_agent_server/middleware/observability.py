"""Observability middleware for FastAPI.

Integrates OpenTelemetry tracing and Prometheus metrics collection
into the request/response cycle.
"""

import time
import logging
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting observability metrics from HTTP requests.
    
    Automatically instruments:
    - Request duration
    - Error tracking
    - Response status
    - Request context (method, path, client IP)
    """
    
    def __init__(self, app, observability_manager, logger: Optional[logging.Logger] = None):
        """Initialize observability middleware.
        
        Args:
            app: FastAPI app
            observability_manager: ObservabilityManager instance
            logger: Logger instance
        """
        super().__init__(app)
        self.observability_manager = observability_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip instrumentation for observability-internal paths to avoid
        # self-referential metric inflation (Prometheus scrapes, health probes).
        if path in ("/metrics", "/health", "/ready", "/status"):
            return await call_next(request)
        
        # Tracing context
        trace_attributes = {
            "http.method": method,
            "http.path": path,
            "http.client_ip": client_ip,
            "http.scheme": request.url.scheme,
        }
        
        try:
            with self.observability_manager.trace_operation(
                f"http.{method.lower()}",
                attributes=trace_attributes
            ):
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                status_code = response.status_code
                
                self.observability_manager.record_request(
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration=duration
                )
                
                # Log request
                log_level = logging.WARNING if status_code >= 400 else logging.INFO
                self.logger.log(
                    log_level,
                    f"{method} {path} - {status_code} ({duration:.3f}s)",
                    extra={
                        "http.method": method,
                        "http.path": path,
                        "http.status_code": status_code,
                        "http.duration_seconds": duration,
                        "http.client_ip": client_ip,
                    }
                )
                
                # Add performance headers
                response.headers["X-Process-Time"] = str(duration)
                
                return response
        
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            
            self.observability_manager.record_request(
                method=method,
                path=path,
                status_code=500,
                duration=duration
            )
            
            self.logger.error(
                f"{method} {path} - ERROR ({duration:.3f}s): {str(e)}",
                extra={
                    "http.method": method,
                    "http.path": path,
                    "http.status_code": 500,
                    "http.duration_seconds": duration,
                    "http.client_ip": client_ip,
                    "error": str(e),
                },
                exc_info=True
            )
            
            raise


class StreamingMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking streaming response metrics.
    
    Tracks:
    - Number of streaming events
    - Streaming success/failure
    - Bytes transmitted
    """
    
    def __init__(self, app, observability_manager, logger: Optional[logging.Logger] = None):
        """Initialize streaming metrics middleware.
        
        Args:
            app: FastAPI app
            observability_manager: ObservabilityManager instance
            logger: Logger instance
        """
        super().__init__(app)
        self.observability_manager = observability_manager
        self.logger = logger or logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process streaming requests.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Only track streaming endpoints
        if "/stream" not in request.url.path:
            return await call_next(request)
        
        response = await call_next(request)
        
        # For streaming responses, record event
        if response.status_code == 200:
            self.observability_manager.record_streaming_event("stream_start")
        
        return response
