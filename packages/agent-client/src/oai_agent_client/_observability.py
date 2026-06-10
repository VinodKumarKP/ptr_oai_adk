"""Observability hooks for request/response instrumentation and metrics.

This module provides pluggable hooks for integrating observability tools
(logging, metrics, tracing, etc.) without hard dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional


@dataclass
class RequestEvent:
    """Event data emitted when a request starts.
    
    Attributes:
        timestamp: When the request was initiated (UTC).
        request_id: Unique request ID (X-Request-ID header).
        method: HTTP method (GET, POST, etc.).
        endpoint: Target endpoint path or full URL.
        payload_size: Approximate size of request body in bytes.
        user_agent: Optional user agent string from config headers.
        headers: Request headers (sensitive values redacted).
    """

    timestamp: datetime
    request_id: str
    method: str
    endpoint: str
    payload_size: int
    user_agent: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResponseEvent:
    """Event data emitted when a response is received.
    
    Attributes:
        timestamp: When the response was received (UTC).
        request_id: Unique request ID (matches RequestEvent).
        status_code: HTTP status code.
        response_time_ms: Total round-trip time in milliseconds.
        is_stream: True if this was a streaming response.
        response_size: Approximate response size in bytes.
    """

    timestamp: datetime
    request_id: str
    status_code: int
    response_time_ms: float
    is_stream: bool
    response_size: int = 0


@dataclass
class ErrorEvent:
    """Event data emitted when a request fails.
    
    Attributes:
        timestamp: When the error occurred (UTC).
        request_id: Unique request ID.
        method: HTTP method that failed.
        endpoint: Target endpoint.
        exception_type: Class name of the exception.
        exception_message: Exception message.
        response_time_ms: Time elapsed before failure.
        is_retryable: Whether the error can be retried.
    """

    timestamp: datetime
    request_id: str
    method: str
    endpoint: str
    exception_type: str
    exception_message: str
    response_time_ms: float
    is_retryable: bool


@dataclass
class RetryEvent:
    """Event data emitted when a request is retried.
    
    Attributes:
        timestamp: When the retry was initiated (UTC).
        request_id: Unique request ID.
        attempt: Retry attempt number (0-indexed).
        exception_type: Type of exception that triggered retry.
        delay_ms: Delay before retry in milliseconds.
        reason: Human-readable reason for retry.
    """

    timestamp: datetime
    request_id: str
    attempt: int
    exception_type: str
    delay_ms: float
    reason: str


class ObservabilityHooks:
    """Pluggable hooks for request/response instrumentation.
    
    Use this class to integrate observability tools (logging, metrics, tracing)
    without hard dependencies. Pass an instance to ClientConfig.observability_hooks.
    
    Example::
    
        def my_metrics_hook(event: ResponseEvent):
            statsd.timing("agent_client.request_ms", event.response_time_ms)
            statsd.gauge("agent_client.status", event.status_code)
        
        hooks = ObservabilityHooks(on_response=my_metrics_hook)
        config = ClientConfig(url="...", observability_hooks=hooks)
        async with AsyncAgentClient(config=config) as client:
            await client.invoke("hello")
    """

    def __init__(
        self,
        on_request: Optional[Callable[[RequestEvent], None]] = None,
        on_response: Optional[Callable[[ResponseEvent], None]] = None,
        on_error: Optional[Callable[[ErrorEvent], None]] = None,
        on_retry: Optional[Callable[[RetryEvent], None]] = None,
    ):
        """Initialize observability hooks.
        
        Args:
            on_request: Called when a request starts. Hook should complete quickly.
            on_response: Called when a successful response is received.
            on_error: Called when a request fails with an exception.
            on_retry: Called when a request will be retried.
        
        All hooks are optional and should not raise exceptions (failures are logged).
        """
        self.on_request = on_request
        self.on_response = on_response
        self.on_error = on_error
        self.on_retry = on_retry

    def emit_request(self, event: RequestEvent) -> None:
        """Safely emit a request event."""
        if self.on_request:
            try:
                self.on_request(event)
            except Exception:
                # Silently ignore hook failures to avoid disrupting the request
                pass

    def emit_response(self, event: ResponseEvent) -> None:
        """Safely emit a response event."""
        if self.on_response:
            try:
                self.on_response(event)
            except Exception:
                # Silently ignore hook failures
                pass

    def emit_error(self, event: ErrorEvent) -> None:
        """Safely emit an error event."""
        if self.on_error:
            try:
                self.on_error(event)
            except Exception:
                # Silently ignore hook failures
                pass

    def emit_retry(self, event: RetryEvent) -> None:
        """Safely emit a retry event."""
        if self.on_retry:
            try:
                self.on_retry(event)
            except Exception:
                # Silently ignore hook failures
                pass


__all__ = [
    "RequestEvent",
    "ResponseEvent",
    "ErrorEvent",
    "RetryEvent",
    "ObservabilityHooks",
]
