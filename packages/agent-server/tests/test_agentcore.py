"""Tests for AWS Bedrock AgentCore support: platform auth mode, session
binding, and the /invocations + /ping HTTP contract router."""
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from oai_agent_server.exceptions import AuthenticationException
from oai_agent_server.routers.agentcore import create_agentcore_router
from oai_agent_server.security.dependencies import verify_api_key

SESSION_HEADER = "x-amzn-bedrock-agentcore-runtime-session-id"


def _make_request(path: str = "/some-path", headers=None):
    request = MagicMock(spec=Request)
    url = MagicMock()
    url.path = path
    request.url = url
    client = MagicMock()
    client.host = "10.20.30.40"
    request.client = client
    request.headers = headers or {}
    request.state = MagicMock()
    request.app = MagicMock()
    request.app.state.agent_name = "test_agent"
    return request


# ---------------------------------------------------------------------------
# Platform auth mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_platform_mode_with_session_header_binds_state():
    request = _make_request(headers={SESSION_HEADER: "sess-123"})
    env = {'AGENT_AUTH_ENABLED': 'true', 'AGENT_AUTH_MODE': 'platform'}
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value=env):
        assert await verify_api_key(
            request, api_token=None, api_token_underscore=None,
            x_api_key=None, authorization=None,
        ) is True
    assert request.state.session_id == "sess-123"


@pytest.mark.asyncio
async def test_platform_mode_missing_session_header_rejected():
    request = _make_request(headers={})
    env = {'AGENT_AUTH_ENABLED': 'true', 'AGENT_AUTH_MODE': 'platform'}
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value=env):
        with pytest.raises(AuthenticationException) as excinfo:
            await verify_api_key(
                request, api_token=None, api_token_underscore=None,
                x_api_key=None, authorization=None,
            )
    assert "session header" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_token_mode_spoofed_session_header_still_requires_token():
    """The session header must never act as a credential outside platform mode."""
    request = _make_request(headers={SESSION_HEADER: "spoofed-session"})
    env = {'AGENT_AUTH_ENABLED': 'true'}  # default token mode
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value=env):
        with pytest.raises(AuthenticationException):
            await verify_api_key(
                request, api_token=None, api_token_underscore=None,
                x_api_key=None, authorization=None,
            )


@pytest.mark.asyncio
async def test_ping_path_bypasses_auth():
    request = _make_request(path="/ping")
    env = {'AGENT_AUTH_ENABLED': 'true'}
    with patch('oai_agent_server.security.dependencies.get_original_environ', return_value=env):
        assert await verify_api_key(
            request, api_token=None, api_token_underscore=None,
            x_api_key=None, authorization=None,
        ) is True


# ---------------------------------------------------------------------------
# AgentCore contract router
# ---------------------------------------------------------------------------

class _StubServerState:
    def __init__(self):
        self.active_requests = 0
        self.active_invocations = 0
        self.request_lock = threading.Lock()
        self.is_agent_ready = True
        self.start_time = time.time()


@pytest.fixture
def chat_service():
    service = MagicMock()
    service.process_chat = AsyncMock(
        return_value=JSONResponse(content={
            "content": "hello", "session_id": "s1", "interaction_id": "i1",
        })
    )
    service.process_stream_chat = AsyncMock(
        return_value=JSONResponse(content={"stream": True})
    )
    return service


@pytest.fixture
def server_state():
    return _StubServerState()


def _build_client(chat_service, server_state, allowed_modes=("agentcore",)):
    app = FastAPI()
    router = create_agentcore_router(chat_service, server_state, list(allowed_modes))
    assert router is not None
    app.include_router(router)
    return TestClient(app)


def test_router_disabled_without_mode(chat_service, server_state):
    assert create_agentcore_router(chat_service, server_state, ["chat", "health"]) is None


def test_invocations_requires_prompt(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    resp = client.post("/invocations", json={})
    assert resp.status_code == 400
    assert "prompt" in resp.json()["detail"]


def test_invocations_json_response(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    resp = client.post("/invocations", json={"prompt": "What's the weather?"})
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello"
    chat_service.process_chat.assert_awaited_once()
    chat_request = chat_service.process_chat.await_args.kwargs["chat_request"]
    assert chat_request.message == "What's the weather?"


def test_invocations_accepts_message_alias(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    resp = client.post("/invocations", json={"message": "hi there"})
    assert resp.status_code == 200
    chat_request = chat_service.process_chat.await_args.kwargs["chat_request"]
    assert chat_request.message == "hi there"


def test_invocations_stream_uses_stream_service(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    resp = client.post("/invocations", json={"prompt": "hi", "stream": True})
    assert resp.status_code == 200
    chat_service.process_stream_chat.assert_awaited_once()
    chat_service.process_chat.assert_not_awaited()


def test_invocations_resets_busy_counter(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    client.post("/invocations", json={"prompt": "hi"})
    assert server_state.active_invocations == 0


def test_ping_healthy(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    resp = client.get("/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "Healthy"
    assert isinstance(body["time_of_last_update"], int)


def test_ping_healthy_busy_during_invocation(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    server_state.active_invocations = 1  # simulate in-flight invocation
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "HealthyBusy"


def test_ping_unhealthy_when_agent_not_ready(chat_service, server_state):
    client = _build_client(chat_service, server_state)
    server_state.is_agent_ready = False
    resp = client.get("/ping")
    assert resp.status_code == 503
    assert resp.json()["status"] == "Unhealthy"


# ---------------------------------------------------------------------------
# ALWAYS_ACTIVE_MODES override
# ---------------------------------------------------------------------------

def test_always_active_modes_default(monkeypatch):
    from oai_agent_server.main import AgentHTTPServer
    monkeypatch.delenv("ALWAYS_ACTIVE_MODES", raising=False)
    assert AgentHTTPServer._resolve_always_active_modes() == set(
        AgentHTTPServer.ALWAYS_ACTIVE_MODES
    )


def test_always_active_modes_env_override(monkeypatch):
    from oai_agent_server.main import AgentHTTPServer
    monkeypatch.setenv("ALWAYS_ACTIVE_MODES", "health, chat")
    assert AgentHTTPServer._resolve_always_active_modes() == {"health", "chat"}


def test_always_active_modes_empty_env(monkeypatch):
    from oai_agent_server.main import AgentHTTPServer
    monkeypatch.setenv("ALWAYS_ACTIVE_MODES", "")
    assert AgentHTTPServer._resolve_always_active_modes() == set()
