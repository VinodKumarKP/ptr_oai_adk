"""Tests for the httpx-based agent client (async + sync)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from oai_agent_client import (
    AgentClient,
    AsyncAgentClient,
    ClientConfig,
    SyncAgentClient,
)
from oai_agent_client.exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    ConfigurationError,
    ConnectionError,
    AgentConnectionError,
    AgentTimeoutError,
    RateLimitError,
    ServerError,
    ServerStartupError,
)
from oai_agent_client.agent_client import _redact_headers
from oai_agent_client._retry import compute_backoff


MOCK_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Mock-transport helpers
# ---------------------------------------------------------------------------


def make_async_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def make_sync_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def healthy_handler(
    response_builder: Callable[[httpx.Request], httpx.Response]
) -> Callable[[httpx.Request], httpx.Response]:
    """Wrap a handler so /health always returns 200 OK."""
    def _h(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health") or request.url.path.endswith("health"):
            return httpx.Response(200, text="ok")
        return response_builder(request)
    return _h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def remote_config():
    return ClientConfig(url=MOCK_URL, headers={"Authorization": "Bearer test"})


@pytest.fixture
def local_config():
    return ClientConfig(
        command="dummy_command",
        args=["--port", "8000"],
        headers={"Authorization": "Bearer test"},
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_valid_remote_config(self):
        config = ClientConfig(url=MOCK_URL, headers={"X-Token": "test"})
        assert str(config.url) == MOCK_URL + "/"
        assert config.headers == {"X-Token": "test"}

    def test_valid_local_config(self):
        config = ClientConfig(command="python", args=["-m", "http.server"])
        assert config.command == "python"
        assert config.args == ["-m", "http.server"]

    def test_missing_url_and_command_fails(self):
        with pytest.raises(ValueError):
            ClientConfig()

    def test_both_url_and_command_fails(self):
        with pytest.raises(ValueError):
            ClientConfig(url=MOCK_URL, command="python")


# ---------------------------------------------------------------------------
# Async happy paths (rewritten on httpx MockTransport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_success(remote_config):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "success"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        response = await client.invoke("hello", config={"session_id": "123"})
    assert response == {"response": "success"}
    assert captured["url"].endswith("/chat")
    assert captured["json"] == {"message": "hello", "session_id": "123"}


@pytest.mark.asyncio
async def test_stream_success(remote_config):
    body = "data: {\"chunk\": 1}\n\ndata: {\"chunk\": 2}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(
            200,
            text=body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        chunks = [c async for c in client.stream("hello stream")]
    assert chunks == [{"chunk": 1}, {"chunk": 2}]


@pytest.mark.asyncio
async def test_api_error_handling(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(500, text="Internal Server Error")

    config = ClientConfig(url=MOCK_URL, max_retries=0)
    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(APIError) as excinfo:
            await client.invoke("test")
    assert excinfo.value.status_code == 500
    assert "Internal Server Error" in excinfo.value.message


@pytest.mark.asyncio
async def test_connection_error_handling():
    config = ClientConfig(
        url="http://localhost:9999",
        headers={"Authorization": "Bearer test"},
        startup_timeout=2,
    )
    with pytest.raises(ConnectionError):
        async with AsyncAgentClient(config=config):
            pass


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_local_server_management(mock_subprocess, local_config):
    mock_process = AsyncMock()
    mock_process.stdout.at_eof.side_effect = [False, True]
    mock_process.stdout.readline.return_value = b"log line"
    mock_subprocess.return_value = mock_process

    with patch.object(AsyncAgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AsyncAgentClient(config=local_config) as client:
            mock_subprocess.assert_called_once_with(
                "dummy_command",
                "--port",
                "8000",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert client._server_process is not None

        mock_process.terminate.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec", side_effect=OSError("File not found"))
async def test_server_startup_error(mock_subprocess, local_config):
    with pytest.raises(ServerStartupError):
        async with AsyncAgentClient(config=local_config):
            pass


# ---------------------------------------------------------------------------
# Batch 1 tests (ported)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_failure_reraises_original(remote_config):
    sentinel = RuntimeError("original boom")
    with patch.object(
        AsyncAgentClient, "_wait_for_server", new_callable=AsyncMock, side_effect=sentinel
    ):
        with pytest.raises(RuntimeError, match="original boom"):
            async with AsyncAgentClient(config=remote_config):
                pass


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_log_tasks_cancelled_on_close(mock_subprocess, local_config):
    mock_process = AsyncMock()
    mock_process.stdout.at_eof.return_value = False
    mock_process.stderr.at_eof.return_value = False

    block_forever = asyncio.Event()

    async def never_returns():
        await block_forever.wait()
        return b""

    mock_process.stdout.readline.side_effect = never_returns
    mock_process.stderr.readline.side_effect = never_returns
    mock_subprocess.return_value = mock_process

    with patch.object(AsyncAgentClient, "_wait_for_server", new_callable=AsyncMock):
        client = AsyncAgentClient(config=local_config)
        async with client:
            assert len(client._log_tasks) == 2
            tasks = list(client._log_tasks)
            assert all(not t.done() for t in tasks)
        assert client._log_tasks == []
        assert all(t.done() for t in tasks)


@pytest.mark.asyncio
async def test_accepts_201(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(201, json={"ok": True})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        result = await client.invoke("hi")
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_accepts_204_no_body(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(204)

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        result = await client.invoke("hi")
    assert result == {}


@pytest.mark.asyncio
async def test_4xx_still_raises_api_error(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(404, text="not found")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        with pytest.raises(APIError) as excinfo:
            await client.invoke("hi")
    assert excinfo.value.status_code == 404
    assert "not found" in excinfo.value.message


def test_redact_headers_helper():
    redacted = _redact_headers(
        {
            "Authorization": "Bearer secret",
            "X-Api-Key": "key123",
            "Cookie": "session=abc",
            "X-Auth-Token": "tok",
            "Content-Type": "application/json",
        }
    )
    assert redacted["Authorization"] == "***REDACTED***"
    assert redacted["X-Api-Key"] == "***REDACTED***"
    assert redacted["Cookie"] == "***REDACTED***"
    assert redacted["X-Auth-Token"] == "***REDACTED***"
    assert redacted["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_redacted_headers_in_logs(caplog):
    config = ClientConfig(
        url=MOCK_URL,
        headers={"Authorization": "Bearer test"},
        log_level="DEBUG",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, json={"x": 1})

    transport = make_async_transport(handler)
    with caplog.at_level(logging.DEBUG, logger="oai_agent_client.async_client"):
        async with AsyncAgentClient(config=config, transport=transport) as client:
            await client.invoke("hi")

    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Bearer test" not in combined
    assert "***REDACTED***" in combined


def test_old_connection_error_alias_works():
    from oai_agent_client import ConnectionError as ImportedConnectionError

    assert ImportedConnectionError is AgentConnectionError
    assert ConnectionError is AgentConnectionError


def test_agent_client_alias_is_async():
    """The legacy ``AgentClient`` name maps to ``AsyncAgentClient``."""
    assert AgentClient is AsyncAgentClient


# ---------------------------------------------------------------------------
# Batch 2 tests (ported)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_401_raises_auth_error(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(401, text="nope")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        with pytest.raises(AuthError) as ei:
            await client.invoke("hi")
    assert ei.value.status_code == 401
    assert isinstance(ei.value, APIError)


@pytest.mark.asyncio
async def test_429_raises_rate_limit_with_retry_after():
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(429, text="slow down", headers={"Retry-After": "7"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(RateLimitError) as ei:
            await client.invoke("hi")
    assert ei.value.retry_after == 7.0


@pytest.mark.asyncio
async def test_500_raises_server_error():
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(500, text="boom")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(ServerError) as ei:
            await client.invoke("hi")
    assert ei.value.status_code == 500
    assert isinstance(ei.value, APIError)


@pytest.mark.asyncio
async def test_timeout_raises_agent_timeout_error():
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        raise httpx.ReadTimeout("simulated")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(AgentTimeoutError):
            await client.invoke("hi")


@pytest.mark.asyncio
async def test_request_id_propagated_to_exception():
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(500, text="boom", headers={"X-Request-ID": "server-id-xyz"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(ServerError) as ei:
            await client.invoke("hi")
    assert ei.value.request_id == "server-id-xyz"


@pytest.mark.asyncio
async def test_request_id_in_outgoing_headers(remote_config):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": 1})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        await client.invoke("hi")
    assert "x-request-id" in captured["headers"] or "X-Request-ID" in captured["headers"]
    rid = captured["headers"].get("x-request-id") or captured["headers"].get("X-Request-ID")
    assert len(rid.split("-")) == 5


@pytest.mark.asyncio
async def test_caller_supplied_request_id_used(remote_config):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": 1})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        await client.invoke("hi", request_id="caller-id-123")
    rid = captured["headers"].get("x-request-id") or captured["headers"].get("X-Request-ID")
    assert rid == "caller-id-123"


def test_config_is_frozen():
    config = ClientConfig(url=MOCK_URL)
    with pytest.raises(Exception):
        config.request_timeout = 99.0


@pytest.mark.asyncio
async def test_update_headers_propagates_to_client(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        client.update_headers(**{"X-Custom": "v1"})
        assert client._headers["X-Custom"] == "v1"
        assert client._client.headers.get("X-Custom") == "v1"


@pytest.mark.asyncio
async def test_retry_on_503_for_GET():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": 1})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._request("GET", "ping")
    assert result == {"ok": 1}
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_no_retry_on_503_for_POST_by_default():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=3,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        return httpx.Response(503, text="busy")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(ServerError):
            await client.invoke("hi")
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_retry_on_429_uses_retry_after_header():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(429, text="", headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": 1})

    sleep_calls: List[float] = []

    async def fake_sleep(d):
        sleep_calls.append(d)

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with patch("asyncio.sleep", side_effect=fake_sleep):
            result = await client.invoke("hi")
    assert result == {"ok": 1}
    assert sleep_calls == [0.0]


@pytest.mark.asyncio
async def test_max_retries_respected():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        return httpx.Response(503, text="busy")

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ServerError):
                await client._request("GET", "ping")
    assert call_count["n"] == 3


def test_backoff_increases_then_caps():
    factor, cap, jitter = 1.0, 8.0, 0.0
    delays = [compute_backoff(i, factor, cap, jitter) for i in range(6)]
    assert delays[0] == 1.0
    assert delays[1] == 2.0
    assert delays[2] == 4.0
    assert delays[3] == 8.0
    assert delays[4] == 8.0
    assert delays[5] == 8.0


def test_jitter_within_range():
    factor, cap, jitter = 1.0, 100.0, 1.0
    for attempt in range(4):
        upper = factor * (2 ** attempt)
        for _ in range(20):
            d = compute_backoff(attempt, factor, cap, jitter)
            assert 0.0 <= d <= upper + 1e-9


@pytest.mark.asyncio
async def test_per_call_retry_override_enables_post_retry():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        if call_count["n"] < 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": 1})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.invoke("hi", retry=True)
    assert result == {"ok": 1}
    assert call_count["n"] == 2


def test_deprecated_timeout_field_maps_to_request_timeout():
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        config = ClientConfig(url=MOCK_URL, timeout=120.0)
    assert config.request_timeout == 120.0
    assert any(issubclass(rec.category, DeprecationWarning) for rec in caught)


# ---------------------------------------------------------------------------
# Batch 3: Sync client tests
# ---------------------------------------------------------------------------


def test_sync_invoke(remote_config):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "success"})

    transport = make_sync_transport(handler)
    with SyncAgentClient(config=remote_config, transport=transport) as client:
        result = client.invoke("hello", config={"session_id": "abc"})
    assert result == {"response": "success"}
    assert captured["url"].endswith("/chat")
    assert captured["json"] == {"message": "hello", "session_id": "abc"}


def test_sync_stream(remote_config):
    body = "data: {\"chunk\": 1}\n\ndata: {\"chunk\": 2}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    transport = make_sync_transport(handler)
    with SyncAgentClient(config=remote_config, transport=transport) as client:
        chunks = list(client.stream("hi"))
    assert chunks == [{"chunk": 1}, {"chunk": 2}]


def test_sync_does_not_support_local_subprocess():
    config = ClientConfig(command="python", args=["-m", "http.server"])
    with pytest.raises(ConfigurationError):
        SyncAgentClient(config=config)


def test_sync_close_idempotent(remote_config):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    transport = make_sync_transport(handler)
    client = SyncAgentClient(config=remote_config, transport=transport)
    with client:
        pass
    client.close()  # second close: no error


def test_sync_429_rate_limit():
    config = ClientConfig(url=MOCK_URL, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(429, text="slow", headers={"Retry-After": "3"})

    transport = make_sync_transport(handler)
    with SyncAgentClient(config=config, transport=transport) as client:
        with pytest.raises(RateLimitError) as ei:
            client.invoke("hi")
    assert ei.value.retry_after == 3.0


def test_sync_retry_on_503_for_GET():
    """Sync client retries on retryable statuses the same way as async."""
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"ok": 1})

    transport = make_sync_transport(handler)
    with SyncAgentClient(config=config, transport=transport) as client:
        with patch("time.sleep") as sleeper:
            result = client._request("GET", "ping")
            assert sleeper.call_count == 2
    assert result == {"ok": 1}
    assert call_count["n"] == 3


def test_async_and_sync_share_retry_semantics():
    """Both clients use the same internal retry helper."""
    from oai_agent_client._base import _is_retryable

    cfg = ClientConfig(url=MOCK_URL, retry_on_methods={"GET"}, retry_on_statuses={503})
    exc = ServerError(503, "")
    assert _is_retryable(exc, "GET", cfg, None) is True
    assert _is_retryable(exc, "POST", cfg, None) is False
    # Retry override forces retry.
    assert _is_retryable(exc, "POST", cfg, True) is True


# ---------------------------------------------------------------------------
# Batch 3: SSE parser correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_handles_multi_line_data(remote_config):
    """Multi-line ``data:`` fields are joined with newlines per the SSE spec."""
    body = "data: line1\ndata: line2\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        chunks = [c async for c in client.stream("x")]
    # Joined non-JSON value falls back to {"content": ...}
    assert chunks == [{"content": "line1\nline2"}]


@pytest.mark.asyncio
async def test_sse_handles_comments_and_event_field(remote_config):
    """Lines starting with ``:`` are SSE comments and must be ignored. ``event:`` is passed through."""
    body = ":heartbeat\ndata: {\"n\": 1}\n\nevent: ping\ndata: {\"n\": 2}\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        chunks = [c async for c in client.stream("x")]
    assert chunks == [{"n": 1}, {"n": 2}]


@pytest.mark.asyncio
async def test_sse_handles_partial_chunks(remote_config):
    """SSE events split across multiple TCP chunks must still parse correctly."""

    # An httpx transport that yields the body in tiny pieces.
    body = "data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\n"
    pieces = [body[i : i + 3].encode("utf-8") for i in range(0, len(body), 3)]

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, text="ok")

        async def gen():
            for p in pieces:
                yield p

        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=gen(),
        )

    transport = httpx.MockTransport(stream_handler)
    async with AsyncAgentClient(config=remote_config, transport=transport) as client:
        chunks = [c async for c in client.stream("x")]
    assert chunks == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# Batch 3: URL and endpoint configurability (M3, M4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", [
    "http://example.com",
    "http://example.com/",
    "http://example.com/api",
    "http://example.com/api/",
])
@pytest.mark.asyncio
async def test_url_join_robustness(base):
    """Base URL with/without trailing slash and with a path prefix all work."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["path"] = path
        return httpx.Response(200, json={"ok": 1})

    config = ClientConfig(url=base)
    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        await client.invoke("hi")

    # The path must contain /chat regardless of whether base had /api or not.
    assert captured["path"].endswith("/chat")


@pytest.mark.asyncio
async def test_configurable_invoke_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, text="ok")
        captured["path"] = path
        return httpx.Response(200, json={"ok": 1})

    config = ClientConfig(url=MOCK_URL, invoke_endpoint="v1/chat")
    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport) as client:
        await client.invoke("hi")
    assert captured["path"].endswith("/v1/chat")


@pytest.mark.asyncio
async def test_configurable_health_endpoint():
    seen_paths: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, text="ok")

    config = ClientConfig(url=MOCK_URL, health_endpoint="ready")
    transport = make_async_transport(handler)
    async with AsyncAgentClient(config=config, transport=transport):
        pass
    assert any(p.endswith("/ready") for p in seen_paths)


# ---------------------------------------------------------------------------
# Batch 3: Keyword-only enforcement (M2)
# ---------------------------------------------------------------------------


def test_keyword_only_params_enforced():
    """``config`` must be passed by keyword to the constructor."""
    cfg = ClientConfig(url=MOCK_URL)
    with pytest.raises(TypeError):
        AsyncAgentClient(cfg)  # type: ignore[misc]
    with pytest.raises(TypeError):
        SyncAgentClient(cfg)  # type: ignore[misc]


def test_invoke_optional_kwargs_are_keyword_only():
    """``request_id`` and ``retry`` must be keyword-only on invoke/stream."""
    import inspect
    sig = inspect.signature(AsyncAgentClient.invoke)
    assert sig.parameters["request_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["retry"].kind is inspect.Parameter.KEYWORD_ONLY
    sig2 = inspect.signature(SyncAgentClient.invoke)
    assert sig2.parameters["request_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig2.parameters["retry"].kind is inspect.Parameter.KEYWORD_ONLY


def test_bad_request_error_for_400():
    """A vanilla 400 yields BadRequestError, not AuthError."""
    from oai_agent_client._base import _raise_for_status
    with pytest.raises(BadRequestError) as ei:
        _raise_for_status(400, "bad", None, None)
    assert ei.value.status_code == 400
