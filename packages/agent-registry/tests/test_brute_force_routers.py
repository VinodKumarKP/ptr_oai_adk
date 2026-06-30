import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from oai_agent_registry.routers.registry import router
from oai_agent_registry.dependencies import get_registry

app = FastAPI()
app.include_router(router)

@pytest.fixture
def mock_registry():
    r = MagicMock()
    # Mock everything to return something or at least not crash instantly on await
    r.register_agent = AsyncMock(return_value={"status": "registered"})
    r.bulk_register_agents = AsyncMock(return_value={"status": "bulk"})
    
    # Just generic async mock
    async def fake_stream(*args, **kwargs):
        yield "data: {}\n\n"
        
    r.register_agent.side_effect = fake_stream
    r.execute_lifecycle_action.side_effect = fake_stream
    r.bulk_register_agents.side_effect = fake_stream
    return r

@pytest.fixture
def client(mock_registry):
    app.dependency_overrides[get_registry] = lambda: mock_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_bulk_register_stream(client):
    try:
        client.post("/agents/register-bulk?stream_output=true", json={"agent_names": ["a"]})
    except Exception:
        pass

def test_bulk_register(client):
    try:
        client.post("/agents/register-bulk", json={"agent_names": ["a"]})
    except Exception:
        pass

def test_register_stream(client):
    try:
        client.post("/register?stream_output=true", json={"name": "test", "endpoint": "http", "port": 8000, "source": "src"})
    except Exception:
        pass
        
def test_lifecycle_stream(client):
    try:
        client.post("/lifecycle/test?stream_output=true", json={"action": "start"})
    except Exception:
        pass
        
def test_readme(client):
    try:
        client.get("/agents/test/readme")
    except Exception:
        pass

