"""Phase 1 Tests: Configuration validation, observability, error handling.

Tests for the production-readiness improvements:
- Configuration validation (port, timeout, endpoints, retry settings)
- Observability hooks (request/response/error/retry events)
- Stream error handling (malformed JSON)
- Enhanced error messages with context
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from oai_agent_client import AsyncAgentClient, ClientConfig, SyncAgentClient
from oai_agent_client.exceptions import (
    APIError,
    AgentConnectionError,
    AgentTimeoutError,
    ServerError,
)
from oai_agent_client._observability import (
    ErrorEvent,
    ObservabilityHooks,
    RequestEvent,
    ResponseEvent,
    RetryEvent,
)


MOCK_URL = "http://localhost:8000"


def make_async_transport(handler):
    return httpx.MockTransport(handler)


def healthy_handler(response_builder):
    """Wrap handler to make /health always return 200."""
    def _h(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return response_builder(request)
    return _h


# ============================================================================
# CONFIGURATION VALIDATION TESTS (Task 1)
# ============================================================================


class TestClientConfigValidation:
    """Test enhanced configuration validation."""

    def test_port_must_be_in_valid_range(self):
        """Port must be between 1 and 65535."""
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ClientConfig(command="test", port=0)

        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ClientConfig(command="test", port=65536)

        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            ClientConfig(command="test", port=-1)

        # Valid ports should work
        config = ClientConfig(command="test", port=1)
        assert config.port == 1
        config = ClientConfig(command="test", port=65535)
        assert config.port == 65535

    def test_request_timeout_must_be_gte_connect_timeout(self):
        """request_timeout must be >= connect_timeout."""
        # This should fail: request_timeout (30) < connect_timeout (60)
        with pytest.raises(ValueError, match="request_timeout.*must be >=.*connect_timeout"):
            ClientConfig(
                command="test",
                connect_timeout=60.0,
                request_timeout=30.0,
            )

        # This should work: request_timeout == connect_timeout
        config = ClientConfig(
            command="test",
            connect_timeout=30.0,
            request_timeout=30.0,
        )
        assert config.request_timeout == config.connect_timeout

        # This should work: request_timeout > connect_timeout
        config = ClientConfig(
            command="test",
            connect_timeout=10.0,
            request_timeout=60.0,
        )
        assert config.request_timeout > config.connect_timeout

    def test_command_must_be_non_empty(self):
        """Command cannot be empty or whitespace-only."""
        with pytest.raises(ValueError, match="command cannot be empty"):
            ClientConfig(command="")

        with pytest.raises(ValueError, match="command cannot be empty"):
            ClientConfig(command="   ")

        # Valid command
        config = ClientConfig(command="python")
        assert config.command == "python"

    def test_endpoint_paths_must_not_contain_slashes(self):
        """Endpoint paths should be relative (not absolute URLs)."""
        # Reject absolute paths (starting with /)
        with pytest.raises(ValueError, match="endpoint.*must be a relative path"):
            ClientConfig(
                command="test",
                health_endpoint="/health",
            )

        # Reject absolute URLs (containing ://)
        with pytest.raises(ValueError, match="endpoint.*must be a relative path"):
            ClientConfig(
                command="test",
                invoke_endpoint="http://example.com/chat",
            )

        # Valid endpoints (including paths with slashes like "api/chat")
        config = ClientConfig(
            command="test",
            health_endpoint="health",
            invoke_endpoint="api/v1/chat",
            stream_endpoint="chat/stream",
        )
        assert config.health_endpoint == "health"
        assert config.invoke_endpoint == "api/v1/chat"
        assert config.stream_endpoint == "chat/stream"

    def test_retry_jitter_must_be_in_range(self):
        """retry_jitter must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="retry_jitter.*between 0.0 and 1.0"):
            ClientConfig(command="test", retry_jitter=-0.1)

        with pytest.raises(ValueError, match="retry_jitter.*between 0.0 and 1.0"):
            ClientConfig(command="test", retry_jitter=1.1)

        # Valid values
        config = ClientConfig(command="test", retry_jitter=0.0)
        assert config.retry_jitter == 0.0
        config = ClientConfig(command="test", retry_jitter=0.5)
        assert config.retry_jitter == 0.5
        config = ClientConfig(command="test", retry_jitter=1.0)
        assert config.retry_jitter == 1.0

    def test_max_retries_must_be_non_negative(self):
        """max_retries must be >= 0."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            ClientConfig(command="test", max_retries=-1)

        # Valid values
        config = ClientConfig(command="test", max_retries=0)
        assert config.max_retries == 0

    def test_retry_factors_must_be_positive(self):
        """retry_backoff_factor and retry_max_backoff must be > 0."""
        with pytest.raises(ValueError, match="retry_backoff_factor must be > 0"):
            ClientConfig(command="test", retry_backoff_factor=0.0)

        with pytest.raises(ValueError, match="retry_backoff_factor must be > 0"):
            ClientConfig(command="test", retry_backoff_factor=-1.0)

        with pytest.raises(ValueError, match="retry_max_backoff must be > 0"):
            ClientConfig(command="test", retry_max_backoff=0.0)

        with pytest.raises(ValueError, match="retry_max_backoff must be > 0"):
            ClientConfig(command="test", retry_max_backoff=-1.0)

    def test_timeout_deprecation_with_conflict_raises(self):
        """Using both timeout and request_timeout should raise."""
        with pytest.raises(ValueError, match="Cannot specify both timeout and request_timeout"):
            ClientConfig(
                command="test",
                timeout=60.0,
                request_timeout=90.0,  # Both explicitly set
            )

    def test_timeout_deprecation_copies_if_no_conflict(self):
        """Using timeout alone should copy to request_timeout."""
        with pytest.warns(DeprecationWarning, match="timeout is deprecated"):
            config = ClientConfig(
                command="test",
                timeout=45.0,
            )
        # Should have copied timeout to request_timeout
        assert config.request_timeout == 45.0


# ============================================================================
# OBSERVABILITY HOOKS TESTS (Tasks 3, 6, 7)
# ============================================================================


class TestObservabilityHooks:
    """Test observability hooks functionality."""

    def test_request_event_emission(self):
        """RequestEvent should be emitted on request start."""
        events: list[RequestEvent] = []

        def capture_request(event: RequestEvent) -> None:
            events.append(event)

        hooks = ObservabilityHooks(on_request=capture_request)
        assert hooks.on_request is not None

        # Emit an event
        event = RequestEvent(
            timestamp=datetime.utcnow(),
            request_id="req-123",
            method="POST",
            endpoint="chat",
            payload_size=256,
        )
        hooks.emit_request(event)

        assert len(events) == 1
        assert events[0].request_id == "req-123"
        assert events[0].method == "POST"

    def test_response_event_emission(self):
        """ResponseEvent should be emitted on success."""
        events: list[ResponseEvent] = []

        def capture_response(event: ResponseEvent) -> None:
            events.append(event)

        hooks = ObservabilityHooks(on_response=capture_response)
        event = ResponseEvent(
            timestamp=datetime.utcnow(),
            request_id="req-456",
            status_code=200,
            response_time_ms=42.5,
            is_stream=False,
        )
        hooks.emit_response(event)

        assert len(events) == 1
        assert events[0].status_code == 200
        assert events[0].response_time_ms == 42.5

    def test_error_event_emission(self):
        """ErrorEvent should be emitted on error."""
        events: list[ErrorEvent] = []

        def capture_error(event: ErrorEvent) -> None:
            events.append(event)

        hooks = ObservabilityHooks(on_error=capture_error)
        event = ErrorEvent(
            timestamp=datetime.utcnow(),
            request_id="req-789",
            method="POST",
            endpoint="chat",
            exception_type="AgentTimeoutError",
            exception_message="Connection timeout",
            response_time_ms=5000.0,
            is_retryable=True,
        )
        hooks.emit_error(event)

        assert len(events) == 1
        assert events[0].exception_type == "AgentTimeoutError"
        assert events[0].is_retryable is True

    def test_retry_event_emission(self):
        """RetryEvent should be emitted on retry."""
        events: list[RetryEvent] = []

        def capture_retry(event: RetryEvent) -> None:
            events.append(event)

        hooks = ObservabilityHooks(on_retry=capture_retry)
        event = RetryEvent(
            timestamp=datetime.utcnow(),
            request_id="req-xyz",
            attempt=1,
            exception_type="ServerError",
            delay_ms=100.0,
            reason="HTTP 503",
        )
        hooks.emit_retry(event)

        assert len(events) == 1
        assert events[0].attempt == 1
        assert events[0].delay_ms == 100.0

    def test_hook_exceptions_are_silently_ignored(self):
        """Hook exceptions should not disrupt requests."""
        def failing_hook(event: Any) -> None:
            raise RuntimeError("hook boom")

        hooks = ObservabilityHooks(on_request=failing_hook)
        # Should not raise
        hooks.emit_request(
            RequestEvent(
                timestamp=datetime.utcnow(),
                request_id="req-test",
                method="GET",
                endpoint="health",
                payload_size=0,
            )
        )

    def test_config_accepts_observability_hooks(self):
        """ClientConfig should accept observability_hooks."""
        hooks = ObservabilityHooks()
        config = ClientConfig(
            command="test",
            observability_hooks=hooks,
        )
        assert config.observability_hooks is hooks


@pytest.mark.asyncio
async def test_async_client_emits_request_event():
    """AsyncAgentClient should emit RequestEvent on invoke."""
    events: list[RequestEvent] = []

    def capture_request(event: RequestEvent) -> None:
        events.append(event)

    hooks = ObservabilityHooks(on_request=capture_request)
    config = ClientConfig(url=MOCK_URL, observability_hooks=hooks)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={"response": "ok"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        await client.invoke("hello")

    assert len(events) == 1
    assert events[0].method == "POST"
    assert events[0].endpoint == "chat"
    assert events[0].payload_size > 0


@pytest.mark.asyncio
async def test_async_client_emits_response_event():
    """AsyncAgentClient should emit ResponseEvent on success."""
    events: list[ResponseEvent] = []

    def capture_response(event: ResponseEvent) -> None:
        events.append(event)

    hooks = ObservabilityHooks(on_response=capture_response)
    config = ClientConfig(url=MOCK_URL, observability_hooks=hooks)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={"response": "ok"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        await client.invoke("hello")

    assert len(events) == 1
    assert events[0].status_code == 200
    assert events[0].response_time_ms >= 0


@pytest.mark.asyncio
async def test_async_client_emits_error_event():
    """AsyncAgentClient should emit ErrorEvent on error."""
    events: list[ErrorEvent] = []

    def capture_error(event: ErrorEvent) -> None:
        events.append(event)

    hooks = ObservabilityHooks(on_error=capture_error)
    config = ClientConfig(url=MOCK_URL, observability_hooks=hooks, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(500, text="server error")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        try:
            await client.invoke("hello")
        except ServerError:
            pass

    assert len(events) == 1
    assert events[0].exception_type == "ServerError"
    assert events[0].is_retryable is False


@pytest.mark.asyncio
async def test_async_client_emits_retry_event():
    """AsyncAgentClient should emit RetryEvent on retry."""
    retry_events: list[RetryEvent] = []

    def capture_retry(event: RetryEvent) -> None:
        retry_events.append(event)

    hooks = ObservabilityHooks(on_retry=capture_retry)
    config = ClientConfig(
        url=MOCK_URL,
        observability_hooks=hooks,
        max_retries=2,
        retry_backoff_factor=0.001,  # Near-instant retries for testing (must be > 0)
        retry_jitter=0.0,
        retry_on_methods={"POST"},  # Enable retry for POST (normally non-idempotent)
    )

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        if call_count["n"] < 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": True})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        await client.invoke("hello")

    assert len(retry_events) == 1
    assert retry_events[0].attempt == 0
    assert retry_events[0].exception_type == "ServerError"


# ============================================================================
# STREAM ERROR HANDLING TESTS (Task 4)
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_malformed_sse_json_raises_error():
    """Malformed JSON in SSE falls back to plain text content."""
    config = ClientConfig(url=MOCK_URL, max_retries=0)
    malformed_body = "data: {invalid json}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(
            200,
            text=malformed_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        chunks = [c async for c in client.stream("test")]
        # Non-JSON SSE data falls back to plain text content
        assert chunks == [{"content": "{invalid json}"}]


@pytest.mark.asyncio
async def test_valid_sse_json_succeeds():
    """Valid JSON in SSE should parse correctly."""
    config = ClientConfig(url=MOCK_URL)
    body = 'data: {"chunk": 1}\n\ndata: {"chunk": 2}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(
            200,
            text=body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        chunks = [c async for c in client.stream("test")]
    assert len(chunks) == 2
    assert chunks[0] == {"chunk": 1}
    assert chunks[1] == {"chunk": 2}


# ============================================================================
# ENHANCED ERROR MESSAGES TESTS (Task 5)
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_error_includes_timeout_value():
    """Timeout errors should include the timeout duration."""
    config = ClientConfig(
        url=MOCK_URL, 
        max_retries=0, 
        connect_timeout=3.0,
        request_timeout=5.0
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        raise httpx.ReadTimeout("simulated")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        try:
            await client.invoke("hello")
        except AgentTimeoutError as e:
            # Error message should mention the timeout value
            assert "5.0" in str(e) or "timeout" in str(e).lower()


@pytest.mark.asyncio
async def test_connection_error_includes_url():
    """Connection errors should include the target URL."""
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        raise httpx.ConnectError("simulated")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        try:
            await client.invoke("hello")
        except AgentConnectionError as e:
            # Error message should include context
            assert "localhost" in str(e) or MOCK_URL in str(e)


@pytest.mark.asyncio
async def test_server_error_includes_status_and_request_id():
    """Server errors should include status code and request ID."""
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(
            500,
            text="Internal Server Error",
            headers={"X-Request-ID": "req-xyz-123"},
        )

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        try:
            await client.invoke("hello")
        except ServerError as e:
            assert "500" in str(e)
            assert e.request_id == "req-xyz-123"


def test_sync_client_config_validation_same_as_async():
    """SyncAgentClient should validate config the same way."""
    # Should fail with invalid port
    with pytest.raises(ValueError, match="port must be between"):
        ClientConfig(url=MOCK_URL, port=99999)

    # Should fail with invalid timeout hierarchy
    with pytest.raises(ValueError, match="request_timeout.*must be >=.*connect_timeout"):
        ClientConfig(
            url=MOCK_URL,
            connect_timeout=100.0,
            request_timeout=10.0,
        )
