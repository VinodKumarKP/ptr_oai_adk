import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from oai_agent_server.services.agent_service import AgentService
from oai_agent_server.main import ServerState
from oai_agent_server.exceptions import AgentInitializationException
import asyncio
import json

@pytest.fixture
def server_state():
    return ServerState()

@pytest.fixture
def agent_service(mock_agent, server_state):
    return AgentService(mock_agent, server_state, MagicMock())

@pytest.mark.asyncio
async def test_initialize_agent_success(agent_service):
    agent_service.agent.initialize = AsyncMock()
    response = await agent_service.initialize_agent()
    assert response["message"] == "Agent initialized successfully"
    agent_service.agent.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_initialize_agent_failure(agent_service):
    agent_service.agent.initialize = AsyncMock(side_effect=Exception("Init failed"))
    with pytest.raises(AgentInitializationException):
        await agent_service.initialize_agent()

@pytest.mark.asyncio
async def test_restart_server_success(agent_service):
    with patch('asyncio.create_task'):
        response = await agent_service.restart_server()
        assert response.status_code == 200
        assert agent_service.server_state.is_shutting_down is True

@pytest.mark.asyncio
async def test_restart_server_with_active_requests(agent_service):
    agent_service.server_state.active_requests = 1
    agent_service.server_state.shutdown_timeout = 0.1 # Short timeout
    
    # Mock sleep to avoid waiting
    with patch('asyncio.sleep', new_callable=AsyncMock), \
         patch('asyncio.create_task'):
        
        response = await agent_service.restart_server()
        assert response.status_code == 200
        # Should have logged timeout message
        agent_service.logger.info.assert_called()

@pytest.mark.asyncio
async def test_restart_server_exception(agent_service):
    # Simulate exception by making logger raise
    # The implementation catches Exception and returns 500
    # BUT, the first logger.info is inside the try block.
    # If logger.info raises, it goes to except block.
    # The except block ALSO calls logger.info.
    # If logger.info raises there too, the test fails with that exception.
    
    # We need to make logger.info raise ONLY ONCE or handle the second call.
    # Or better, mock something else to raise.
    # restart_server calls:
    # 1. logger.info (start)
    # 2. server_state.is_shutting_down = True
    # 3. while loop...
    # 4. asyncio.create_task
    
    # Let's mock asyncio.create_task to raise, which happens at the end.
    with patch('asyncio.create_task', side_effect=Exception("Task error")):
        response = await agent_service.restart_server()
        assert response.status_code == 500
        body = json.loads(response.body)
        assert "Restart failed" in body['detail']

@pytest.mark.asyncio
async def test_kill_switch_success(agent_service):
    with patch('asyncio.create_task'):
        response = await agent_service.kill_switch()
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_kill_switch_exception(agent_service):
    # Same issue as restart_server, logger.info in except block might raise if we mock logger.
    # Mock asyncio.create_task to raise.
    with patch('asyncio.create_task', side_effect=Exception("Task error")):
        response = await agent_service.kill_switch()
        assert response.status_code == 500
        body = json.loads(response.body)
        assert "Kill failed" in body['detail']

@pytest.mark.asyncio
async def test_get_agent_info(agent_service):
    info = await agent_service.get_agent_info(auth_enabled=True, request_isolation=True)
    assert info["agent_name"] == "test_agent"
    assert info["auth_enabled"] is True

@pytest.mark.asyncio
async def test_get_prompts(agent_service):
    agent_service.agent.agent_config = {"prompts": ["p1", "p2"]}
    response = await agent_service.get_prompts()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_delayed_restart(agent_service):
    with patch('os._exit') as mock_exit, \
         patch('asyncio.sleep', new_callable=AsyncMock):
        await agent_service._delayed_restart()
        mock_exit.assert_called_with(0)

@pytest.mark.asyncio
async def test_immediate_kill(agent_service):
    with patch('os._exit') as mock_exit, \
         patch('asyncio.sleep', new_callable=AsyncMock):
        await agent_service._immediate_kill()
        mock_exit.assert_called_with(1)
