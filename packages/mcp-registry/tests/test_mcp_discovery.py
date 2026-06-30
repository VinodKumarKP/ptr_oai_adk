import pytest
from unittest.mock import patch, MagicMock

from oai_mcp_registry.services.mcp_discovery import (
    _parse_github_repo,
    _is_sensitive_by_name,
    _normalize_server_name,
    MCPDiscovery
)

def test_parse_github_repo():
    assert _parse_github_repo("https://github.com/owner/repo") == "owner/repo"
    assert _parse_github_repo("https://github.com/owner/repo.git") == "owner/repo"
    assert _parse_github_repo("http://github.com/owner/repo/") == "owner/repo"
    assert _parse_github_repo("invalid-url") is None

def test_is_sensitive_by_name():
    assert _is_sensitive_by_name("API_KEY") is True
    assert _is_sensitive_by_name("my_password") is True
    assert _is_sensitive_by_name("DB_SECRET") is True
    assert _is_sensitive_by_name("NORMAL_ENV") is False

def test_normalize_server_name():
    assert _normalize_server_name("My Server Name!") == "my_server_name"
    assert _normalize_server_name("  Test-Server_123  ") == "test_server_123"

@pytest.fixture
def discovery():
    return MCPDiscovery()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_find_yaml_entries(mock_get, discovery):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"type": "file", "name": "server.yaml", "download_url": "http://dl.com/server.yaml"},
        {"type": "dir", "name": "subdir"},
        {"type": "file", "name": "template.yaml"}
    ]
    mock_get.return_value = mock_response

    async with __import__("httpx").AsyncClient() as client:
        entries = await discovery._find_yaml_entries(client, "owner/repo", {})
        
    assert len(entries) == 1
    assert entries[0]["name"] == "server.yaml"

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
name: My Cool Server
description: This is a test
port: 8080
env_vars:
  - name: API_KEY
    default: secret123
  - name: NORMAL_VAR
    default: val
tags: [test, mock]
    """
    mock_get.return_value = mock_response

    entry = {"name": "test_server.yaml", "download_url": "http://dl.com/test_server.yaml"}
    async with __import__("httpx").AsyncClient() as client:
        info = await discovery._parse_yaml_entry(client, entry, "https://github.com/owner/repo", {})
        
    assert info is not None
    assert info["name"] == "test_server"
    assert "My Cool Server" in info["description"]
    assert info["port"] == 8080
    assert "API_KEY" in info["env_vars"]
    assert "API_KEY" in info["sensitive_vars"]
    assert "NORMAL_VAR" not in info["sensitive_vars"]
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
@patch.object(MCPDiscovery, "_find_yaml_entries")
@patch.object(MCPDiscovery, "_parse_yaml_entry")
async def test_discover(mock_parse_yaml, mock_find_yaml, discovery):
    mock_find_yaml.return_value = [{"name": "server1.yaml"}, {"name": "server2.yaml"}]
    
    mock_parse_yaml.side_effect = [
        {"name": "server1", "status": "", "error": None},
        {"name": "server2", "status": "", "error": "some error"}
    ]
    
    result = await discovery.discover(
        git_repository_url="https://github.com/owner/repo",
        existing_server_names=["server1"],
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
