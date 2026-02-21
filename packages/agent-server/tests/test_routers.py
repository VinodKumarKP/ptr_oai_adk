import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from oai_agent_server.routers.agent import create_agent_router
from oai_agent_server.routers.chat import create_chat_router
from oai_agent_server.routers.health import create_health_router
from oai_agent_server.routers.logs import create_logs_router
from oai_agent_server.routers.tokens import create_token_router
from oai_agent_server.main import ServerState
from oai_agent_server.security.dependencies import verify_api_key, verify_jwt_token


# --- Agent Router Tests ---
@pytest.fixture
def agent_router_app(mock_agent):
    agent_service = MagicMock()
    agent_service.initialize_agent = AsyncMock(return_value={"status": "ok"})
    agent_service.restart_server = AsyncMock(return_value={"status": "restarting"})
    agent_service.kill_switch = AsyncMock(return_value={"status": "killed"})
    agent_service.get_agent_info = AsyncMock(return_value={"name": "test"})
    agent_service.get_prompts = AsyncMock(return_value={"prompts": []})
    
    router = create_agent_router(agent_service, enable_request_isolation=True)
    app = FastAPI()
    app.include_router(router)
    
    # Override dependency to bypass auth
    app.dependency_overrides[verify_api_key] = lambda: True
    
    return app, agent_service

def test_agent_initialize(agent_router_app):
    app, service = agent_router_app
    client = TestClient(app)
    response = client.post("/agent/initialize")
    assert response.status_code == 200
    service.initialize_agent.assert_called_once()

def test_agent_restart(agent_router_app):
    app, service = agent_router_app
    client = TestClient(app)
    response = client.post("/restart")
    assert response.status_code == 200
    service.restart_server.assert_called_once()

def test_agent_kill(agent_router_app):
    app, service = agent_router_app
    client = TestClient(app)
    response = client.post("/kill")
    assert response.status_code == 200
    service.kill_switch.assert_called_once()

def test_agent_info(agent_router_app):
    app, service = agent_router_app
    client = TestClient(app)
    response = client.get("/agent/info")
    assert response.status_code == 200
    service.get_agent_info.assert_called_once()

def test_agent_prompts(agent_router_app):
    app, service = agent_router_app
    client = TestClient(app)
    response = client.get("/prompts")
    assert response.status_code == 200
    service.get_prompts.assert_called_once()


# --- Chat Router Tests ---
@pytest.fixture
def chat_router_app():
    chat_service = MagicMock()
    chat_service.process_chat = AsyncMock(return_value={"response": "ok", "session_id": "1"})
    chat_service.process_stream_chat = AsyncMock(return_value={"response": "stream", "session_id": "1"})
    
    router = create_chat_router(chat_service)
    app = FastAPI()
    app.include_router(router)
    
    app.dependency_overrides[verify_api_key] = lambda: True
    
    return app, chat_service

def test_chat_endpoint(chat_router_app):
    app, service = chat_router_app
    client = TestClient(app)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    service.process_chat.assert_called_once()

def test_chat_with_files(chat_router_app):
    app, service = chat_router_app
    client = TestClient(app)
    files = {'files': ('test.txt', b'content')}
    response = client.post("/chat/with-files", data={"message": "hello"}, files=files)
    assert response.status_code == 200
    service.process_chat.assert_called()

def test_chat_stream(chat_router_app):
    app, service = chat_router_app
    client = TestClient(app)
    response = client.post("/chat/stream", json={"message": "hello"})
    assert response.status_code == 200
    service.process_stream_chat.assert_called_once()


# --- Health Router Tests ---
@pytest.fixture
def health_router_app():
    server_state = ServerState()
    router = create_health_router("test_agent", server_state, True)
    app = FastAPI()
    app.include_router(router)
    return app, server_state

def test_health_root(health_router_app):
    app, _ = health_router_app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "endpoints" in response.json()

def test_health_check(health_router_app):
    app, _ = health_router_app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_server_status(health_router_app):
    app, state = health_router_app
    state.active_requests = 5
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["active_requests"] == 5

def test_debug_env(health_router_app):
    app, _ = health_router_app
    client = TestClient(app)
    response = client.get("/debug/env")
    assert response.status_code == 200


# -- Token Router Tests
@pytest.fixture
def token_router_app():
    token_service = MagicMock()
    # Mock generate_token (synchronous)
    token_service.generate_token = MagicMock(return_value={"token": "test_token", "user_id": None})
    
    # Pass allowed_modes as a list, or omit it to use default
    router = create_token_router(token_service, allowed_modes=["token"])
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[verify_jwt_token] = lambda: True

    return app, token_service

def test_token(token_router_app):
    app, service = token_router_app
    client = TestClient(app)
    response = client.post("/token/custom")
    assert response.status_code == 200
    assert response.json()["token"] == "test_token"
    # Verify default TTL is 3600 (passed as positional)
    service.generate_token.assert_called_with("unknown", None, None, 3600)

def test_token_short_term(token_router_app):
    app, service = token_router_app
    client = TestClient(app)
    response = client.post("/token/short-term")
    assert response.status_code == 200
    assert response.json()["token"] == "test_token"
    # Verify TTL is 300 (passed as keyword argument in implementation)
    service.generate_token.assert_called_with("unknown", None, None, ttl_seconds=300)

def test_token_long_term(token_router_app):
    app, service = token_router_app
    client = TestClient(app)
    response = client.post("/token/long-term")
    assert response.status_code == 200
    assert response.json()["token"] == "test_token"
    # Verify TTL is 2592000 (30 days)
    service.generate_token.assert_called_with("unknown", None, None, ttl_seconds=2592000)

def test_token_permanent(token_router_app):
    app, service = token_router_app
    client = TestClient(app)
    response = client.post("/token/permanent")
    assert response.status_code == 200
    assert response.json()["token"] == "test_token"
    # Verify TTL is None
    service.generate_token.assert_called_with("unknown", None, None, ttl_seconds=None)


# --- Logs Router Tests ---
@pytest.fixture
def logs_router_app():
    logging_service = MagicMock()
    logging_service.get_logs = AsyncMock(return_value=[])
    logging_service.get_session_logs = AsyncMock(return_value=[])
    logging_service.get_log_stats = AsyncMock(return_value={})
    logging_service.get_user_stats = AsyncMock(return_value=[])
    
    router = create_logs_router(logging_service)
    app = FastAPI()
    app.include_router(router)
    
    app.dependency_overrides[verify_api_key] = lambda: True

    return app, logging_service

def test_get_logs(logs_router_app):
    app, service = logs_router_app
    client = TestClient(app)
    response = client.get("/logs")
    assert response.status_code == 200
    service.get_logs.assert_called_once()

def test_get_session_logs(logs_router_app):
    app, service = logs_router_app
    client = TestClient(app)
    response = client.get("/logs/sessions/123")
    assert response.status_code == 200
    service.get_session_logs.assert_called_once()

def test_get_log_stats(logs_router_app):
    app, service = logs_router_app
    client = TestClient(app)
    response = client.get("/logs/stats")
    assert response.status_code == 200
    service.get_log_stats.assert_called_once()

def test_get_user_stats(logs_router_app):
    app, service = logs_router_app
    client = TestClient(app)
    response = client.get("/logs/stats/users")
    assert response.status_code == 200
    service.get_user_stats.assert_called_once()
