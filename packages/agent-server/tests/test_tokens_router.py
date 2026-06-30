import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from oai_agent_server.routers.tokens import create_token_router
from oai_agent_server.security.dependencies import verify_jwt_token

@pytest.fixture
def mock_token_service():
    service = MagicMock()
    service.generate_token.return_value = {"token": "generated-token"}
    service.get_all_tokens.return_value = [{"token": "t1", "is_expired": False}]
    service.revoke_all_tokens.return_value = 2
    service.revoke_token.return_value = True
    return service

@pytest.fixture
def token_router_app(mock_token_service):
    router = create_token_router(mock_token_service)
    app = FastAPI()
    app.include_router(router)
    app.state.agent_name = "test_agent"
    
    # Bypass auth dependency
    app.dependency_overrides[verify_jwt_token] = lambda: {}
    
    return app

def test_generate_token_custom(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.post("/token/custom?user_id=u1&role_id=r1&ttl_seconds=3600")
    assert response.status_code == 200
    assert response.json() == {"token": "generated-token"}
    mock_token_service.generate_token.assert_called_once_with(
        "test_agent", "u1", "r1", 3600, max_tokens=10
    )

def test_generate_token_short_term(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.post("/token/short-term?user_id=u1&role_id=r1")
    assert response.status_code == 200
    mock_token_service.generate_token.assert_called_once_with(
        "test_agent", "u1", "r1", 300, max_tokens=10
    )

def test_generate_token_long_term(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.post("/token/long-term?user_id=u1&role_id=r1")
    assert response.status_code == 200
    mock_token_service.generate_token.assert_called_once_with(
        "test_agent", "u1", "r1", 2592000, max_tokens=10
    )

def test_generate_token_permanent(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.post("/token/permanent?user_id=u1&role_id=r1")
    assert response.status_code == 200
    mock_token_service.generate_token.assert_called_once_with(
        "test_agent", "u1", "r1", None, max_tokens=10
    )

def test_generate_token_limit_reached(token_router_app, mock_token_service):
    mock_token_service.generate_token.side_effect = Exception("Token limit reached")
    client = TestClient(token_router_app)
    response = client.post("/token/custom?user_id=u1")
    assert response.status_code == 429
    assert "Token limit reached" in response.json()["detail"]

def test_generate_token_internal_error(token_router_app, mock_token_service):
    mock_token_service.generate_token.side_effect = Exception("Database error")
    client = TestClient(token_router_app)
    response = client.post("/token/custom?user_id=u1")
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_list_tokens(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.get("/token/list?include_expired=true")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["max_tokens"] == 10
    assert data["can_generate"] is True
    mock_token_service.get_all_tokens.assert_called_once_with("test_agent", include_expired=True)

def test_revoke_all_tokens(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.delete("/token/revoke-all")
    assert response.status_code == 200
    assert response.json()["revoked"] == 2
    mock_token_service.revoke_all_tokens.assert_called_once_with("test_agent")

def test_revoke_token_success(token_router_app, mock_token_service):
    client = TestClient(token_router_app)
    response = client.delete("/token/revoke/my-token-str")
    assert response.status_code == 200
    assert response.json()["revoked"] is True
    mock_token_service.revoke_token.assert_called_once_with("my-token-str")

def test_revoke_token_not_found(token_router_app, mock_token_service):
    mock_token_service.revoke_token.return_value = False
    client = TestClient(token_router_app)
    response = client.delete("/token/revoke/my-token-str")
    assert response.status_code == 404
