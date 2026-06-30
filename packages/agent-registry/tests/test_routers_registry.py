import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from oai_agent_registry.routers.registry import router
from oai_agent_registry.dependencies import get_registry
from oai_agent_registry.models import AgentActionHistory, AgentAction

app = FastAPI()
app.include_router(router)

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.get_info = AsyncMock(return_value={"agents": {}})
    registry.health_check = AsyncMock(return_value={"status": "ok"})
    registry.reload_config = AsyncMock(return_value={"status": "reloaded"})
    registry.register_agent = AsyncMock(return_value={"status": "registered"})
    registry.deregister_agent = AsyncMock(return_value={"status": "deregistered"})
    registry.execute_lifecycle_action = AsyncMock(return_value={"status": "action executed"})
    
    # history directly calls db_logger
    registry.db_logger = MagicMock()
    registry.db_logger.get_agent_actions = AsyncMock(return_value=[{"id": 1, "agent_name": "test_agent", "action": "start", "version": "1.0", "created_at": "2023-01-01T00:00:00Z"}])
    registry.db_logger.get_agent_action_count = AsyncMock(return_value=0)
    
    return registry

@pytest.fixture
def client(mock_registry):
    app.dependency_overrides[get_registry] = lambda: mock_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_root(client):
    res = client.get("/")
    assert res.status_code == 200

def test_info(client, mock_registry):
    res = client.get("/info")
    assert res.status_code == 200

def test_health(client, mock_registry):
    res = client.get("/health")
    assert res.status_code == 200

def test_reload_config(client, mock_registry):
    res = client.post("/reload-config")
    assert res.status_code == 200

def test_register_agent(client, mock_registry):
    res = client.post("/register", json={"name": "test_agent", "endpoint": "http", "port": 8000, "source": "src"})
    assert res.status_code == 200

def test_deregister_agent(client, mock_registry):
    res = client.post("/deregister", json={"name": "test_agent"})
    assert res.status_code == 200

def test_lifecycle_action(client, mock_registry):
    res = client.post("/lifecycle/test_agent", json={"action": "start"})
    assert res.status_code == 200

def test_get_history(client, mock_registry):
    res = client.get("/history/test_agent")
    assert res.status_code == 200

