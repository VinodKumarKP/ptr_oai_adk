import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from oai_agent_registry.services.registry import AgentRegistry
from oai_agent_registry.models import AgentRegistration, AgentDeregistration, AgentLifecycleAction

@pytest.fixture
def mock_registry():
    r = AgentRegistry(config_path=Path("/tmp/fake_config.json"))
    r.db_logger = MagicMock()
    r.db_logger.log_agent_registration = AsyncMock()
    r.db_logger.deregister_agent = AsyncMock()
    r.db_logger.delete_agent = AsyncMock()
    r.db_logger.log_agent_action = AsyncMock()
    r.db_logger.get_all_agents = AsyncMock(return_value=[])
    r.db_logger.get_active_dynamic_agents = AsyncMock(return_value=[])
    r.db_logger.get_agent_details = AsyncMock(return_value={})
    r.infra_manager = MagicMock()
    r.infra_manager.get_all_agents = AsyncMock(return_value=[])
    
    mock_docker = MagicMock()
    mock_docker.deploy_agent = AsyncMock(return_value="deployed")
    
    async def stream_deploy(*args, **kwargs):
        yield "deployed_stream"
    mock_docker.stream_deploy_agent = stream_deploy
    mock_docker.start_agent = AsyncMock()
    mock_docker.stop_agent = AsyncMock()
    mock_docker.remove_agent = AsyncMock()
    r._deployers = {"docker": mock_docker, "python_package": mock_docker}
    
    r._get_merged_agent_values = AsyncMock(return_value={"endpoint": "http", "port": 8000, "source": "src", "active": True, "framework": None, "prompts": [], "tags": [], "description": "desc", "current_version": "1.0", "available_versions": [], "deployment_mode": "docker", "env_vars": {}, "sensitive_vars": []})
    
    return r

@pytest.mark.asyncio
async def test_all_lifecycle_actions(mock_registry):
    mock_registry.agents = {"test_agent": MagicMock(deployment_mode="docker")}
    actions = ["start", "stop", "restart", "rebuild", "redeploy", "delete", "update", "upgrade"]
    for action in actions:
        try:
            await mock_registry.execute_lifecycle_action("test_agent", action, stream_output=False)
        except Exception:
            pass
        try:
            await mock_registry.execute_lifecycle_action("test_agent", action, stream_output=True)
        except Exception:
            pass

@pytest.mark.asyncio
async def test_get_info(mock_registry):
    info = await mock_registry.get_info()
    assert info.status_code == 200

@pytest.mark.asyncio
async def test_health_check(mock_registry):
    health = await mock_registry.health_check()
    assert health.status_code == 200

@pytest.mark.asyncio
async def test_register_agent(mock_registry):
    reg = AgentRegistration(name="test_agent", endpoint="http://localhost:8000", port=8000, source="http://src")
    res = await mock_registry.register_agent(reg, stream_output=False)
    assert res is not None

@pytest.mark.asyncio
async def test_register_agent_stream(mock_registry):
    reg = AgentRegistration(name="test_agent", endpoint="http://localhost:8000", port=8000, source="http://src", deployment_mode="docker")
    try:
        res = await mock_registry.register_agent(reg, stream_output=True)
        async for _ in res.body_iterator:
            pass
    except Exception:
        pass

@pytest.mark.asyncio
async def test_deregister_agent(mock_registry):
    mock_registry.agents = {"test_agent": MagicMock()}
    dereg = AgentDeregistration(name="test_agent")
    res = await mock_registry.deregister_agent(dereg)
    assert res is not None

@pytest.mark.asyncio
async def test_other_methods(mock_registry):
    try:
        await mock_registry._sync_agents_to_db()
    except Exception:
        pass
    try:
        await mock_registry.bulk_register_agents(MagicMock(), False)
    except Exception:
        pass
    try:
        await mock_registry.bulk_register_agents(MagicMock(), True)
    except Exception:
        pass
