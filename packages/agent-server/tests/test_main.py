import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_server.main import AgentHTTPServer, ServerState, parse_args, main

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.agent_name = "test_agent"
    agent.initialize = AsyncMock()
    return agent

def test_server_state():
    state = ServerState()
    assert state.active_requests == 0
    assert state.is_shutting_down is False

def test_agent_http_server_init(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger'):
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        assert server.agent_name == "test_agent"
        assert server.app.title == "Agent HTTP Server - test_agent"

@pytest.mark.asyncio
async def test_agent_http_server_startup(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger') as MockDBLogger:
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        server.db_logger.initialize = AsyncMock()
        
        await server.startup()
        mock_agent.initialize.assert_called_once()
        server.db_logger.initialize.assert_called_once()

def test_agent_http_server_run(mock_agent):
    with patch('oai_agent_server.main.get_logger'), \
         patch('oai_agent_server.main.ConfigManager'), \
         patch('oai_agent_server.main.DatabaseLogger'), \
         patch('uvicorn.run') as mock_uvicorn:
        
        server = AgentHTTPServer(mock_agent, "test_agent")
        server.run()
        mock_uvicorn.assert_called_once()

def test_parse_args():
    with patch('sys.argv', ['main.py', 'my_agent', '--port', '9000']):
        args = parse_args()
        assert args.agent_name == 'my_agent'
        assert args.port == 9000

def test_main_entry_point(mock_agent):
    with patch('oai_agent_server.main.parse_args') as mock_parse, \
         patch('oai_agent_server.main.AgentHTTPServer') as MockServer:
        
        mock_parse.return_value.port = 8000
        mock_server_instance = MockServer.return_value
        mock_server_instance.agent_name = "test_agent"
        mock_server_instance.base_config_manager.load_agent_config.return_value = {}
        
        main(mock_server_instance)
        mock_server_instance.run.assert_called_once()
