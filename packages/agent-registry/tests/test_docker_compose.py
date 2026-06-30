import pytest
from unittest.mock import patch, MagicMock

from oai_agent_registry.services.deployers.docker_compose import DockerComposeManager

@pytest.fixture
def manager(tmp_path):
    manager = DockerComposeManager(
        seed_config={},
        compose_output_path=str(tmp_path / "docker-compose.yaml"),
        base_compose_path=str(tmp_path / "docker-compose.base.yaml"),
        agent_base_url="http://agent-base",
        agent_local_registry_url="http://agent-local"
    )
    return manager

def test_get_service_env(manager):
    env = manager._get_service_env(
        "test_service", 8080, {"MY_VAR": "val"}, "http://base", "http://local"
    )
    assert env["MY_VAR"] == "val"
    assert "AGENT_NAME" in env

def test_build_service_dict(manager):
    config = {
        "port": 8080,
        "current_version": "v1.0.0",
        "source": "http://github.com/a/b",
        "env": {"TEST": "value"}
    }
    
    svc = manager._build_service_dict("my_server", config, "http://base", "http://local", refresh_repo=True)
    
    assert svc["image"] == "my-server:v1.0.0"
    assert svc["build"]["args"]["REFRESH_REPO"] == "true"
    assert "my-server" in svc["container_name"]
    assert "8080:8080" in svc["ports"]
    
def test_build_infra_compose_dict(manager):
    infra = manager._build_infra_compose_dict()
    assert infra["name"] == "agent-registry"
    assert "base" in infra["services"]
    assert "valkey" in infra["services"]
    assert "postgres" in infra["services"]

def test_add_agent_from_registration(manager):
    with patch.object(manager, "_add_service_from_registration") as mock_add:
        manager.add_agent_from_registration(
            "test_srv", "http://src", port=9090
        )
        mock_add.assert_called_once_with(
            "test_srv", "http://src", None, None, None, None, 9090, None
        )

@pytest.mark.asyncio
async def test_deploy_agent(manager):
    with patch.object(manager, "_add_service_from_registration") as mock_add, \
         patch.object(manager, "write_compose_file") as mock_write, \
         patch.object(manager, "_run_compose_up_service") as mock_up:
         
         mock_up.return_value = "started"
         res = await manager.deploy_agent("test_srv", "http://src", refresh_repo=True)
         assert res == "started"
         mock_add.assert_called_once()
         mock_write.assert_called_once_with(refresh_repo=True)
         mock_up.assert_called_once_with("test_srv", no_build=False)

@pytest.mark.asyncio
async def test_stream_deploy_agent(manager):
    async def mock_stream(*args, **kwargs):
        yield "line1"
        yield "line2"
        
    with patch.object(manager, "_stream_deploy_service", side_effect=mock_stream):
        lines = [line async for line in manager.stream_deploy_agent("test_srv", "http://src")]
        assert lines == ["line1", "line2"]

def test_start_stop_remove_agent(manager):
    with patch.object(manager, "_run_compose_up_service") as mock_up, \
         patch.object(manager, "_run_compose_stop_service") as mock_stop, \
         patch.object(manager, "_run_compose_down_service") as mock_down:
         
         manager._dynamic_services["test_srv"] = {}
         
         manager.start_agent("test_srv")
         mock_up.assert_called_once_with("test_srv")
         
         manager.stop_agent("test_srv")
         mock_stop.assert_called_once_with("test_srv")
         
         manager.remove_agent("test_srv")
         mock_down.assert_called_once_with("test_srv")
         assert "test_srv" not in manager._dynamic_services

def test_get_all_agents(manager):
    manager._seed_config = {"seed1": {}}
    manager._dynamic_services = {"dyn1": {}}
    servers = manager.get_all_agents()
    assert "seed1" in servers
    assert "dyn1" in servers
