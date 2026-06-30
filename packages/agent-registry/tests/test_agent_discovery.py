import pytest
from unittest.mock import patch, MagicMock

from oai_agent_registry.services.agent_discovery import (
    _parse_github_repo,
    _is_sensitive_by_name,
    _normalize_agent_name,
    _parse_env_requirements,
    AgentDiscovery
)

def test_parse_github_repo():
    assert _parse_github_repo("https://github.com/owner/repo") == "owner/repo"
    assert _parse_github_repo("https://github.com/owner/repo.git") == "owner/repo"
    assert _parse_github_repo("http://github.com/owner/repo/") == "owner/repo"
    assert _parse_github_repo("invalid-url") is None

def test_is_sensitive_by_name():
    assert _is_sensitive_by_name("API_KEY") is True
    assert _is_sensitive_by_name("my_password") is True
    assert _is_sensitive_by_name("NORMAL_ENV") is False

def test_normalize_agent_name():
    assert _normalize_agent_name("My Agent!") == "my_agent"
    assert _normalize_agent_name("  Test-Agent_123  ") == "test_agent_123"

def test_parse_env_requirements():
    env_section = {
        "REQ_VAR": "${REQ_VAR}",
        "DEF_VAR": "${DEF_VAR:-default_val}",
        "LIT_VAR": "literal",
        "NULL_VAR": None,
        "API_KEY": "${API_KEY}"
    }
    explicit_sensitive = ["LIT_VAR"]
    
    reqs = _parse_env_requirements(env_section, explicit_sensitive)
    assert len(reqs) == 5
    
    # REQ_VAR
    req1 = next(r for r in reqs if r["name"] == "REQ_VAR")
    assert req1["required"] is True
    assert req1["default"] is None
    assert req1["sensitive"] is False

    # DEF_VAR
    req2 = next(r for r in reqs if r["name"] == "DEF_VAR")
    assert req2["required"] is False
    assert req2["default"] == "default_val"
    assert req2["sensitive"] is False
    
    # LIT_VAR
    req3 = next(r for r in reqs if r["name"] == "LIT_VAR")
    assert req3["required"] is False
    assert req3["default"] == "literal"
    assert req3["sensitive"] is True  # due to explicit

    # NULL_VAR
    req4 = next(r for r in reqs if r["name"] == "NULL_VAR")
    assert req4["required"] is True
    assert req4["default"] is None
    
    # API_KEY
    req5 = next(r for r in reqs if r["name"] == "API_KEY")
    assert req5["required"] is True
    assert req5["sensitive"] is True  # due to heuristic
    
    assert _parse_env_requirements("not a dict") == []

@pytest.fixture
def discovery():
    return AgentDiscovery()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_find_yaml_entries(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"type": "file", "name": "agent.yaml", "download_url": "http://dl.com/agent.yaml"},
        {"type": "dir", "name": "subdir"},
        {"type": "file", "name": "template.yaml"}
    ]
    mock_get.return_value = mock_response

    async with __import__("httpx").AsyncClient() as client:
        entries = await discovery._find_yaml_entries(client, "owner/repo", {})
        
    assert len(entries) == 1
    assert entries[0]["name"] == "agent.yaml"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_find_yaml_entries_with_config_path(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"type": "file", "name": "custom.yaml", "download_url": "http://dl.com/custom.yaml"}
    ]
    mock_get.return_value = mock_response

    async with __import__("httpx").AsyncClient() as client:
        entries = await discovery._find_yaml_entries(
            client, "owner/repo", {}, config_path="custom_dir/"
        )
        
    assert len(entries) == 1
    assert entries[0]["name"] == "custom.yaml"

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_parse_yaml_entry_success(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
name: My Cool Agent
description: This is a test
type: langgraph
port: 8080
env:
  API_KEY: ${API_KEY}
tags: [test]
prompts: [prompt1]
    """
    mock_get.return_value = mock_response

    entry = {"name": "test_agent.yaml", "download_url": "http://dl.com/test_agent.yaml"}
    async with __import__("httpx").AsyncClient() as client:
        info = await discovery._parse_yaml_entry(client, entry, "https://github.com/owner/repo", {})
        
    assert info is not None
    assert info["name"] == "test_agent"
    assert "My Cool Agent" in info["description"]
    assert info["framework"] == "langgraph"
    assert info["port"] == 8080
    assert len(info["required_env"]) == 1
    assert info["error"] is None

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_parse_yaml_entry_failure(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    entry = {"name": "not_found.yaml", "download_url": "http://dl.com/not_found.yaml"}
    async with __import__("httpx").AsyncClient() as client:
        info = await discovery._parse_yaml_entry(client, entry, "https://github.com/owner/repo", {})
        
    assert info is not None
    assert info["name"] == "not_found"
    assert "HTTP 404" in info["error"]

@pytest.mark.asyncio
@patch.object(AgentDiscovery, "_find_yaml_entries")
@patch.object(AgentDiscovery, "_parse_yaml_entry")
async def test_discover(mock_parse_yaml, mock_find_yaml, discovery):
    mock_find_yaml.return_value = [{"name": "agent1.yaml"}, {"name": "agent2.yaml"}]
    
    mock_parse_yaml.side_effect = [
        {"name": "agent1", "status": "", "error": None},
        {"name": "agent2", "status": "", "error": "some error"}
    ]
    
    result = await discovery.discover(
        git_repository_url="https://github.com/owner/repo",
        existing_agent_names=["agent1"],
        auth_token="test_token"
    )
    
    assert result["total_found"] == 2
    assert result["already_registered"] == 1
    assert result["invalid"] == 1
    assert result["available_to_register"] == 0
    
@pytest.mark.asyncio
async def test_discover_invalid_url(discovery):
    with pytest.raises(ValueError, match="Cannot parse GitHub URL"):
        await discovery.discover("invalid_url")


def test_load_yaml_fallbacks():
    import os
    from unittest.mock import patch
    from oai_agent_registry.services.agent_discovery import _load_yaml
    
    # Test PyYAML parse error
    assert _load_yaml("{") is None


@pytest.mark.asyncio
@patch.object(AgentDiscovery, "_find_yaml_entries")
@patch.object(AgentDiscovery, "_parse_yaml_entry")
async def test_discover_unauthenticated_and_status_available(mock_parse_yaml, mock_find_yaml, discovery):
    import os
    mock_find_yaml.return_value = [{"name": "agent1.yaml"}]
    mock_parse_yaml.return_value = {"name": "agent1", "status": "", "error": None}
    
    with patch.dict(os.environ, {}, clear=True):
        result = await discovery.discover("https://github.com/owner/repo")
        
    assert result["available_to_register"] == 1
    assert result["agents"][0]["status"] == "available"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_find_yaml_entries_not_found_explicit_and_no_yaml(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    
    async with __import__("httpx").AsyncClient() as client:
        # User supplied an explicit path that returns 404
        entries = await discovery._find_yaml_entries(client, "owner/repo", {}, config_path="explicit/")
        assert entries == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_find_yaml_entries_invalid_json(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"not": "a list"}
    mock_get.return_value = mock_response
    
    async with __import__("httpx").AsyncClient() as client:
        entries = await discovery._find_yaml_entries(client, "owner/repo", {})
        assert entries == []


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_parse_yaml_entry_not_mapping_and_exception(mock_get, discovery):
    # 1. Not a mapping
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "not a mapping"
    mock_get.return_value = mock_response
    
    entry = {"name": "test.yaml", "download_url": "http://dl.com/test.yaml"}
    async with __import__("httpx").AsyncClient() as client:
        res = await discovery._parse_yaml_entry(client, entry, "https://github.com/owner/repo", {})
        assert "YAML did not parse to a mapping" in res["error"]
        
    # 2. General exception during parsing
    mock_get.side_effect = Exception("HTTP client error")
    async with __import__("httpx").AsyncClient() as client:
        res = await discovery._parse_yaml_entry(client, entry, "https://github.com/owner/repo", {})
        assert "HTTP client error" in res["error"]
