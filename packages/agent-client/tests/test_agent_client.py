import logging
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from oai_agent_client import ClientConfig, AgentClient
from oai_agent_client.exceptions import (
    APIError,
    AuthError,
    BadRequestError,
    ConnectionError,
    AgentConnectionError,
    AgentTimeoutError,
    RateLimitError,
    ServerError,
    ServerStartupError,
)
from oai_agent_client.agent_client import _redact_headers
from oai_agent_client._retry import compute_backoff

# Mock server URL
MOCK_URL = "http://localhost:8000"

@pytest.fixture
def remote_config():
    """Fixture for a valid remote server configuration."""
    return ClientConfig(url=MOCK_URL, headers={"Authorization": "Bearer test"})

@pytest.fixture
def local_config():
    """Fixture for a valid local server configuration."""
    return ClientConfig(
        command="dummy_command",
        args=["--port", "8000"],
        headers={"Authorization": "Bearer test"}
    )

class TestClientConfig:
    """Tests for the ClientConfig model."""
    def test_valid_remote_config(self):
        config = ClientConfig(url=MOCK_URL, headers={"X-Token": "test"})
        # Pydantic's HttpUrl type adds a trailing slash.
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

@pytest.mark.asyncio
async def test_invoke_success(remote_config):
    """Test a successful invoke call."""
    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = {"response": "success"}
                
                response = await client.invoke("hello", config={"session_id": "123"})

                assert response == {"response": "success"}
                args, kwargs = mock_request.call_args
                assert args == ("POST", "chat")
                assert kwargs["data"] == {"message": "hello", "session_id": "123"}

@pytest.mark.asyncio
async def test_stream_success(remote_config):
    """Test a successful stream call."""
    async def mock_stream_gen():
        yield {"chunk": 1}
        yield {"chunk": 2}

    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client, '_stream_request', return_value=mock_stream_gen()) as mock_stream:
                chunks = [chunk async for chunk in client.stream("hello stream")]

                assert len(chunks) == 2
                assert chunks[0] == {"chunk": 1}
                args, kwargs = mock_stream.call_args
                assert args == ("POST", "chat/stream")
                assert kwargs["data"] == {"message": "hello stream"}

@pytest.mark.asyncio
async def test_api_error_handling(remote_config):
    """Test that APIError is raised on server error."""
    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client._session, 'request') as mock_request:
                mock_response = AsyncMock()
                mock_response.status = 500
                mock_response.text.return_value = "Internal Server Error"
                
                mock_request.return_value.__aenter__.return_value = mock_response
                
                with pytest.raises(APIError) as excinfo:
                    await client.invoke("test")
                
                assert excinfo.value.status_code == 500
                assert "Internal Server Error" in excinfo.value.message

@pytest.mark.asyncio
async def test_connection_error_handling():
    """Test that ConnectionError is raised on connection failure."""
    config = ClientConfig(
        url="http://localhost:9999",
        headers={"Authorization": "Bearer test"},
        startup_timeout=2,
    )
    with pytest.raises(ConnectionError):
        async with AgentClient(config=config):
            pass

@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec')
async def test_local_server_management(mock_subprocess, local_config):
    """Test that the client starts and stops a local server process."""
    mock_process = AsyncMock()
    mock_process.stdout.at_eof.side_effect = [False, True]
    mock_process.stdout.readline.return_value = b"log line"
    mock_subprocess.return_value = mock_process

    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=local_config) as client:
            mock_subprocess.assert_called_once_with(
                "dummy_command", "--port", "8000",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            assert client._server_process is not None

        mock_process.terminate.assert_called_once()

@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec', side_effect=OSError("File not found"))
async def test_server_startup_error(mock_subprocess, local_config):
    """Test that ServerStartupError is raised if the command fails."""
    with pytest.raises(ServerStartupError):
        async with AgentClient(config=local_config):
            pass


# ---------------------------------------------------------------------------
# Batch 1 production-readiness fix tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aenter_failure_reraises_original(remote_config):
    """__aenter__ must re-raise the original exception, not a cleanup error."""
    sentinel = RuntimeError("original boom")

    with patch.object(
        AgentClient, "_wait_for_server", new_callable=AsyncMock, side_effect=sentinel
    ):
        with pytest.raises(RuntimeError, match="original boom"):
            async with AgentClient(config=remote_config):
                pass


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_log_tasks_cancelled_on_close(mock_subprocess, local_config):
    """Background log-reader tasks must be tracked and cancelled on cleanup."""
    mock_process = AsyncMock()
    # Make the stream block forever so we can observe cancellation.
    mock_process.stdout.at_eof.return_value = False
    mock_process.stderr.at_eof.return_value = False

    block_forever = asyncio.Event()

    async def never_returns():
        await block_forever.wait()
        return b""

    mock_process.stdout.readline.side_effect = never_returns
    mock_process.stderr.readline.side_effect = never_returns
    mock_subprocess.return_value = mock_process

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        client = AgentClient(config=local_config)
        async with client:
            assert len(client._log_tasks) == 2
            tasks = list(client._log_tasks)
            assert all(not t.done() for t in tasks)

        # After exit, tasks must be drained.
        assert client._log_tasks == []
        assert all(t.done() for t in tasks)


def _mock_response(status, body_text="", body_json=None, content_length=None):
    """Build an async context-manager mock for ``session.request(...)``."""
    response = AsyncMock()
    response.status = status
    if content_length is None:
        content_length = len(body_text.encode("utf-8")) if body_text else 0
    response.content_length = content_length
    response.text = AsyncMock(return_value=body_text)
    response.json = AsyncMock(return_value=body_json if body_json is not None else {})
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_accepts_201(remote_config):
    """A 201 Created response is treated as success."""
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            ctx = _mock_response(201, body_text='{"ok": true}', body_json={"ok": True})
            with patch.object(client._session, "request", return_value=ctx):
                result = await client.invoke("hi")
                assert result == {"ok": True}


@pytest.mark.asyncio
async def test_accepts_204_no_body(remote_config):
    """A 204 No Content response returns an empty dict without parsing JSON."""
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            ctx = _mock_response(204, body_text="", content_length=0)
            # If .json() were called it would still succeed (returns {}), but we
            # want to ensure the empty-body shortcut is taken.
            ctx.__aenter__.return_value.json.side_effect = AssertionError(
                "json() should not be called on 204"
            )
            with patch.object(client._session, "request", return_value=ctx):
                result = await client.invoke("hi")
                assert result == {}


@pytest.mark.asyncio
async def test_4xx_still_raises_api_error(remote_config):
    """A 404 response surfaces as APIError with the original status code."""
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            ctx = _mock_response(404, body_text="not found")
            with patch.object(client._session, "request", return_value=ctx):
                with pytest.raises(APIError) as excinfo:
                    await client.invoke("hi")
                assert excinfo.value.status_code == 404
                assert "not found" in excinfo.value.message


def test_redact_headers_helper():
    """The redaction helper masks the documented sensitive headers."""
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
    """Authorization header must not appear verbatim in debug log output."""
    config = ClientConfig(
        url=MOCK_URL,
        headers={"Authorization": "Bearer test"},
        log_level="DEBUG",
    )
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            ctx = _mock_response(200, body_text='{"x": 1}', body_json={"x": 1})
            with caplog.at_level(logging.DEBUG, logger="oai_agent_client.agent_client"):
                with patch.object(client._session, "request", return_value=ctx):
                    await client.invoke("hi")

    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "Bearer test" not in combined
    assert "***REDACTED***" in combined


def test_old_connection_error_alias_works():
    """Importing the deprecated name still yields the new exception class."""
    from oai_agent_client import ConnectionError as ImportedConnectionError

    assert ImportedConnectionError is AgentConnectionError
    # And the alias from the exceptions module matches too.
    assert ConnectionError is AgentConnectionError


# ---------------------------------------------------------------------------
# Batch 2 production-readiness fix tests
# ---------------------------------------------------------------------------


def _mock_response_with_headers(status, body_text="", body_json=None, content_length=None, headers=None):
    """Like _mock_response but lets callers stub response headers."""
    response = AsyncMock()
    response.status = status
    if content_length is None:
        content_length = len(body_text.encode("utf-8")) if body_text else 0
    response.content_length = content_length
    response.text = AsyncMock(return_value=body_text)
    response.json = AsyncMock(return_value=body_json if body_json is not None else {})
    response.headers = headers or {}
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_401_raises_auth_error(remote_config):
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            ctx = _mock_response_with_headers(401, body_text="nope")
            with patch.object(client._session, "request", return_value=ctx):
                with pytest.raises(AuthError) as ei:
                    await client.invoke("hi")
                assert ei.value.status_code == 401
                # AuthError IS-A APIError, preserving backward compat.
                assert isinstance(ei.value, APIError)


@pytest.mark.asyncio
async def test_429_raises_rate_limit_with_retry_after(remote_config):
    # Disable retries so the error surfaces immediately.
    config = ClientConfig(url=MOCK_URL, max_retries=0)
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            ctx = _mock_response_with_headers(
                429, body_text="slow down", headers={"Retry-After": "7"}
            )
            with patch.object(client._session, "request", return_value=ctx):
                with pytest.raises(RateLimitError) as ei:
                    await client.invoke("hi")
                assert ei.value.retry_after == 7.0


@pytest.mark.asyncio
async def test_500_raises_server_error(remote_config):
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            ctx = _mock_response_with_headers(500, body_text="boom")
            with patch.object(client._session, "request", return_value=ctx):
                with pytest.raises(ServerError) as ei:
                    await client.invoke("hi")
                assert ei.value.status_code == 500
                assert isinstance(ei.value, APIError)


@pytest.mark.asyncio
async def test_timeout_raises_agent_timeout_error(remote_config):
    config = ClientConfig(url=MOCK_URL, max_retries=0)
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(
                client._session, "request", side_effect=asyncio.TimeoutError()
            ):
                with pytest.raises(AgentTimeoutError):
                    await client.invoke("hi")


@pytest.mark.asyncio
async def test_request_id_propagated_to_exception(remote_config):
    config = ClientConfig(url=MOCK_URL, max_retries=0)
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            ctx = _mock_response_with_headers(
                500, body_text="boom", headers={"X-Request-ID": "server-id-xyz"}
            )
            with patch.object(client._session, "request", return_value=ctx):
                with pytest.raises(ServerError) as ei:
                    await client.invoke("hi")
                # Server's X-Request-ID wins.
                assert ei.value.request_id == "server-id-xyz"


@pytest.mark.asyncio
async def test_request_id_in_outgoing_headers(remote_config):
    captured = {}

    def fake_request(method, url, json=None, headers=None, **kw):
        captured["headers"] = headers
        return _mock_response_with_headers(200, body_text='{"ok": 1}', body_json={"ok": 1})

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                await client.invoke("hi")
    assert "X-Request-ID" in captured["headers"]
    # UUID4 form: 8-4-4-4-12
    assert len(captured["headers"]["X-Request-ID"].split("-")) == 5


@pytest.mark.asyncio
async def test_caller_supplied_request_id_used(remote_config):
    captured = {}

    def fake_request(method, url, json=None, headers=None, **kw):
        captured["headers"] = headers
        return _mock_response_with_headers(200, body_text='{"ok": 1}', body_json={"ok": 1})

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                await client.invoke("hi", request_id="caller-id-123")
    assert captured["headers"]["X-Request-ID"] == "caller-id-123"


@pytest.mark.asyncio
async def test_streaming_uses_stream_timeout(remote_config):
    """The streaming path must pass its own ClientTimeout, not the request_timeout."""
    config = ClientConfig(
        url=MOCK_URL,
        request_timeout=1.0,
        stream_read_timeout=None,
        connect_timeout=3.0,
    )
    captured = {}

    def fake_request(method, url, json=None, headers=None, timeout=None, **kw):
        captured["timeout"] = timeout
        # Return a response context that yields no SSE lines.
        response = AsyncMock()
        response.status = 200
        response.headers = {}

        async def aiter():
            if False:
                yield b""
            return

        response.content = aiter()
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                async for _ in client.stream("hi"):
                    pass

    t = captured["timeout"]
    assert t is not None
    # Stream timeout is total=None (no overall cap) but has sock_connect set.
    assert t.total is None
    assert t.sock_connect == 3.0


def test_config_is_frozen():
    config = ClientConfig(url=MOCK_URL)
    with pytest.raises(Exception):  # pydantic raises ValidationError on frozen mutate
        config.request_timeout = 99.0


@pytest.mark.asyncio
async def test_update_headers_propagates_to_session(remote_config):
    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            client.update_headers(**{"X-Custom": "v1"})
            assert client._headers["X-Custom"] == "v1"
            # And the live session has it too.
            assert client._session.headers.get("X-Custom") == "v1"


@pytest.mark.asyncio
async def test_retry_on_503_for_GET(remote_config):
    """A GET that returns 503 should retry up to max_retries times."""
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,  # no real sleep
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def fake_request(method, url, json=None, headers=None, **kw):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return _mock_response_with_headers(503, body_text="busy")
        return _mock_response_with_headers(200, body_text='{"ok": 1}', body_json={"ok": 1})

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                result = await client._request("GET", "ping")
    assert result == {"ok": 1}
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_no_retry_on_503_for_POST_by_default(remote_config):
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=3,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def fake_request(method, url, json=None, headers=None, **kw):
        call_count["n"] += 1
        return _mock_response_with_headers(503, body_text="busy")

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                with pytest.raises(ServerError):
                    await client.invoke("hi")
    assert call_count["n"] == 1  # no retry


@pytest.mark.asyncio
async def test_retry_on_429_uses_retry_after_header():
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def fake_request(method, url, json=None, headers=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _mock_response_with_headers(429, headers={"Retry-After": "0"})
        return _mock_response_with_headers(200, body_text='{"ok": 1}', body_json={"ok": 1})

    sleep_calls = []

    async def fake_sleep(d):
        sleep_calls.append(d)

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                with patch("asyncio.sleep", side_effect=fake_sleep):
                    result = await client.invoke("hi")
    assert result == {"ok": 1}
    assert sleep_calls == [0.0]


@pytest.mark.asyncio
async def test_max_retries_respected(remote_config):
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def fake_request(method, url, json=None, headers=None, **kw):
        call_count["n"] += 1
        return _mock_response_with_headers(503, body_text="busy")

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
                with pytest.raises(ServerError):
                    await client._request("GET", "ping")
    # initial attempt + 2 retries = 3 calls
    assert call_count["n"] == 3


def test_backoff_increases_then_caps():
    # With jitter=0, deterministic.
    factor, cap, jitter = 1.0, 8.0, 0.0
    delays = [compute_backoff(i, factor, cap, jitter) for i in range(6)]
    # 1, 2, 4, 8, 8 (capped), 8 (capped)
    assert delays[0] == 1.0
    assert delays[1] == 2.0
    assert delays[2] == 4.0
    assert delays[3] == 8.0
    assert delays[4] == 8.0
    assert delays[5] == 8.0


def test_jitter_within_range():
    factor, cap, jitter = 1.0, 100.0, 1.0
    # Full jitter: result in [0, factor * 2**attempt]
    for attempt in range(4):
        upper = factor * (2 ** attempt)
        for _ in range(20):
            d = compute_backoff(attempt, factor, cap, jitter)
            assert 0.0 <= d <= upper + 1e-9


@pytest.mark.asyncio
async def test_per_call_retry_override_enables_post_retry(remote_config):
    """Caller can opt-in to retry for POST via the ``retry`` kwarg."""
    config = ClientConfig(
        url=MOCK_URL,
        max_retries=2,
        retry_backoff_factor=0.0,
        retry_jitter=0.0,
    )
    call_count = {"n": 0}

    def fake_request(method, url, json=None, headers=None, **kw):
        call_count["n"] += 1
        if call_count["n"] < 2:
            return _mock_response_with_headers(503, body_text="busy")
        return _mock_response_with_headers(200, body_text='{"ok": 1}', body_json={"ok": 1})

    with patch.object(AgentClient, "_wait_for_server", new_callable=AsyncMock):
        async with AgentClient(config=config) as client:
            with patch.object(client._session, "request", side_effect=fake_request):
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
