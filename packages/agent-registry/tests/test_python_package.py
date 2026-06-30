import pytest
from pathlib import Path
from unittest.mock import patch

from oai_agent_registry.services.deployers.python_package import PythonPackageDeployer

@pytest.fixture
def manager(tmp_path):
    manager = PythonPackageDeployer(
        seed_config={},
        base_dir=str(tmp_path / "python"),
        agent_base_url="http://agent-base",
        agent_local_registry_url="http://agent-local"
    )
    return manager

def test_get_service_env(manager):
    env = manager._get_service_env(
        "test_service", 8080, "http://base", "http://local", {"MY_VAR": "val"}, "python_package"
    )
    assert env["MY_VAR"] == "val"
    assert "AGENT_NAME" in env

def test_get_server_py_path(manager, tmp_path):
    # Test primary path
    primary_dir = tmp_path / "agent_registry_servers" / "servers" / "test_svc"
    primary_dir.mkdir(parents=True)
    (primary_dir / "server.py").touch()
    
    path = manager._get_server_py_path(tmp_path, "test_svc")
    assert path == primary_dir / "server.py"
    
    # Test fallback path
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    path = manager._get_server_py_path(fallback_dir, "test_svc")
    assert path == fallback_dir / "server.py"

def test_get_start_command(manager):
    cmd = manager._get_start_command(Path("/venv/bin/python"), Path("/srv/server.py"), 8080)
    assert cmd == ["/venv/bin/python", "/srv/server.py", "--port", "8080", "--transport", "streamable-http"]

@pytest.mark.asyncio
async def test_deploy_agent(manager):
    with patch.object(manager, "_deploy_service") as mock_deploy:
        mock_deploy.return_value = "deployed"
        res = await manager.deploy_agent("svc", "url")
        assert res == "deployed"
        mock_deploy.assert_called_once()

@pytest.mark.asyncio
async def test_stream_deploy_agent(manager):
    async def mock_stream(*args, **kwargs):
        yield "l1"
        yield "l2"
        
    with patch.object(manager, "_stream_deploy_service", side_effect=mock_stream):
        lines = [line async for line in manager.stream_deploy_agent("svc", "url")]
        assert lines == ["l1", "l2"]

def test_start_stop_remove_agent(manager):
    with patch.object(manager, "_start_service") as mock_start, \
         patch.object(manager, "_stop_service") as mock_stop, \
         patch.object(manager, "_remove_service") as mock_remove:
         
         manager.start_agent("svc")
         mock_start.assert_called_once_with("svc")
         
         manager.stop_agent("svc")
         mock_stop.assert_called_once_with("svc")
         
         manager.remove_agent("svc")
         mock_remove.assert_called_once_with("svc")
