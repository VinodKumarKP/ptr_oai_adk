import os
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
    from oai_agent_registry.security.dependencies import verify_api_key
    app.dependency_overrides[get_registry] = lambda: mock_registry
    app.dependency_overrides[verify_api_key] = lambda: True
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

@patch("oai_agent_registry.services.agent_discovery.AgentDiscovery")
def test_discover_agents(mock_discovery_cls, client, mock_registry):
    mock_discovery = MagicMock()
    mock_discovery.discover = AsyncMock(return_value={
        "git_repository_url": "https://github.com/example/repo",
        "total_found": 1,
        "available_to_register": 1,
        "already_registered": 0,
        "invalid": 0,
        "agents": [
            {
                "name": "agent1",
                "description": "desc",
                "framework": "openai",
                "agent_type": "type",
                "tags": [],
                "prompts": [],
                "port": 8080,
                "source": "src",
                "status": "available",
                "config_file": "agent.yaml",
                "error": None,
                "required_env": []
            }
        ]
    })
    mock_discovery_cls.return_value = mock_discovery
    
    # Missing git_repository_url
    res = client.post("/agents/discover", json={})
    assert res.status_code == 400
    
    # Success
    res = client.post("/agents/discover", json={"git_repository_url": "https://github.com/example/repo"})
    assert res.status_code == 200
    assert res.json()["total_found"] == 1
    
    # ValueError
    mock_discovery.discover.side_effect = ValueError("invalid repo")
    res = client.post("/agents/discover", json={"git_repository_url": "https://github.com/example/repo"})
    assert res.status_code == 400
    
    # General Exception
    mock_discovery.discover.side_effect = Exception("error")
    res = client.post("/agents/discover", json={"git_repository_url": "https://github.com/example/repo"})
    assert res.status_code == 500


@patch("oai_agent_registry.services.agent_discovery.AgentDiscovery")
def test_register_bulk(mock_discovery_cls, client, mock_registry):
    mock_discovery = MagicMock()
    mock_discovery.discover = AsyncMock(return_value={
        "agents": [
            {
                "name": "agent1",
                "framework": "openai",
                "required_env": [{"name": "API_KEY", "sensitive": True}]
            },
            {
                "name": "agent2",
                "error": "Failed to parse"
            }
        ]
    })
    mock_discovery_cls.return_value = mock_discovery
    
    # Empty agent names
    res = client.post("/agents/register-bulk", json={"git_repository_url": "url", "agent_names": []})
    assert res.status_code == 400
    
    # Discovery ValueError
    mock_discovery.discover.side_effect = ValueError("bad git")
    res = client.post("/agents/register-bulk", json={"git_repository_url": "url", "agent_names": ["agent1"]})
    assert res.status_code == 400
    
    # Discovery Exception
    mock_discovery.discover.side_effect = Exception("error")
    res = client.post("/agents/register-bulk", json={"git_repository_url": "url", "agent_names": ["agent1"]})
    assert res.status_code == 500
    
    # Reset mock and test success (non-streaming)
    mock_discovery.discover.side_effect = None
    res = client.post("/agents/register-bulk", json={"git_repository_url": "url", "agent_names": ["agent1", "agent2", "agent3"]})
    assert res.status_code == 200
    data = res.json()
    assert "agent1" in data["successful"]
    assert any(x["agent_name"] == "agent2" for x in data["failed"])
    assert any(x["agent_name"] == "agent3" for x in data["failed"])
    
    # Streaming bulk deploy
    res = client.post("/agents/register-bulk?stream_output=true", json={"git_repository_url": "url", "agent_names": ["agent1"]})
    assert res.status_code == 200


def test_update_server_env_vars(client, mock_registry):
    mock_registry.update_server_env_vars = AsyncMock(return_value={"status": "success"})
    res = client.patch("/agents/test_agent/env-vars", json={"env_vars": {"A": "B"}})
    assert res.status_code == 200


@patch("oai_agent_registry.routers.registry.read_local_readme")
@patch("oai_agent_registry.routers.registry.fetch_readme")
def test_get_agent_readme(mock_fetch, mock_read_local, client, mock_registry):
    mock_registry.db_logger.get_agent_details = AsyncMock(return_value={"source": "url"})
    
    # 1. Local exists
    mock_read_local.return_value = ("local readme", {"cached": True, "file_path": "path"})
    with patch.dict(os.environ, {"AGENT_LOCAL_DIR": "/tmp"}):
        res = client.get("/agents/test_agent/readme")
        assert res.status_code == 200
        assert res.json()["content"] == "local readme"
        
    # 2. Local None, Fetch success
    mock_read_local.return_value = (None, {})
    mock_fetch.return_value = ("github readme", {"cached": False})
    with patch.dict(os.environ, {"AGENT_LOCAL_DIR": "/tmp"}):
        res = client.get("/agents/test_agent/readme")
        assert res.status_code == 200
        assert res.json()["content"] == "github readme"


@patch("oai_agent_registry.routers.registry.invalidate_readme_cache")
def test_invalidate_cache(mock_invalidate, client, mock_registry):
    with patch.dict(os.environ, {"AGENT_LOCAL_DIR": "/tmp"}):
        res = client.post("/agents/test_agent/readme/invalidate-cache")
        assert res.status_code == 200
        assert res.json()["cache_cleared"] is True


@patch("oai_agent_registry.routers.registry.readme_cache_stats")
def test_cache_stats(mock_stats, client):
    mock_stats.return_value = {
        "entries": [
            {"key": "agent:test", "value": "val"},
            {"key": "other:test", "value": "val"}
        ]
    }
    res = client.get("/readme/cache-stats")
    assert res.status_code == 200
    assert len(res.json()["entries"]) == 1


def test_proxy_request(client, mock_registry):
    mock_registry.proxy_request = AsyncMock(return_value={"proxied": True})
    res = client.post("/test_agent/some/path", json={"data": "val"})
    assert res.status_code == 200

