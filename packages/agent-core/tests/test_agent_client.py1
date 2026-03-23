"""
Unit tests for AgentClient
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from oai_agent_core.components.agent.agent_client import (
    AgentClient,
    ConnectionError,
    ServerConfig
)


class TestServerConfig:
    """Test ServerConfig validation"""

    def test_valid_command_config(self):
        """Test valid command-based configuration"""
        config = ServerConfig(
            name="test-server",
            command="python",
            args=["server.py", "--port", "8000"]
        )
        assert config.name == "test-server"
        assert config.command == "python"
        assert config.args == ["server.py", "--port", "8000"]
        assert config.url is None

    def test_valid_url_config(self):
        """Test valid URL-based configuration"""
        config = ServerConfig(
            name="test-server",
            url="http://localhost:8000"
        )
        assert config.name == "test-server"
        assert config.url == "http://localhost:8000"
        assert config.command is None
        assert config.args is None

    def test_both_command_and_url_raises_error(self):
        """Test that specifying both command and URL raises error"""
        with pytest.raises(ValueError, match="Cannot specify both"):
            ServerConfig(
                name="test-server",
                command="python",
                args=["server.py"],
                url="http://localhost:8000"
            )

    def test_neither_command_nor_url_raises_error(self):
        """Test that specifying neither command nor URL raises error"""
        with pytest.raises(ValueError, match="Either 'Command' and 'args'"):
            ServerConfig(name="test-server")

    def test_empty_name_raises_error(self):
        """Test that empty name raises error"""
        with pytest.raises(ValueError):
            ServerConfig(
                name="",
                url="http://localhost:8000"
            )

    def test_empty_args_raises_error(self):
        """Test that empty args list raises error"""
        with pytest.raises(ValueError, match="'args' list cannot be empty"):
            ServerConfig(
                name="test-server",
                command="python",
                args=[]
            )


class TestAgentClientInitialization:
    """Test AgentClient initialization"""

    def test_init_with_command_config(self, mock_logger):
        """Test initialization with command configuration"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py'],
            'port': 8000
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        assert client.agent_name == 'test-agent'
        assert client.type == 'stdio'
        assert client.server_port == 8000
        assert client.server_url == "http://localhost:8000"
        assert not client.connected

    def test_init_with_url_config(self, mock_logger):
        """Test initialization with URL configuration"""
        config = {
            'name': 'test-agent',
            'url': 'http://example.com:9000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        assert client.agent_name == 'test-agent'
        assert client.type == 'http'
        assert client.server_url == "http://example.com:9000"
        assert not client.connected

    def test_init_creates_logger_if_not_provided(self):
        """Test that logger is created if not provided"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        with patch('oai_agent_core.components.agent.agent_client.create_agent_logger') as mock_create_logger:
            mock_create_logger.return_value = MagicMock()
            client = AgentClient(server_config=config)

            mock_create_logger.assert_called_once()
            assert client.logger is not None

    def test_init_with_custom_timeout(self, mock_logger):
        """Test initialization with custom timeout"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, timeout=600, logger=mock_logger)

        assert client.timeout.total == 600

    def test_init_invalid_config_raises_error(self, mock_logger):
        """Test that invalid config raises error"""
        config = {
            'name': 'test-agent'
            # Missing both command/args and url
        }

        with pytest.raises(ValueError):
            AgentClient(server_config=config, logger=mock_logger)


class TestAgentClientServerManagement:
    """Test server management functionality"""

    @pytest.mark.asyncio
    async def test_start_server_success(self, mock_logger):
        """Test successful server startup"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py'],
            'port': 8000,
            'startup_delay': 0.1
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        # Mock subprocess
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Server is running
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = MagicMock(return_value='')
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = MagicMock(return_value='')

        # Mock asyncio.create_task to prevent actual task creation
        with patch('asyncio.create_task'):
            with patch('oai_agent_core.components.agent.agent_client.subprocess.Popen', return_value=mock_process):
                # First call returns False (server not ready yet), then True (server ready)
                with patch.object(client, '_wait_for_server_ready', side_effect=[False, True]):
                    result = await client._start_server()

                    assert result is True
                    assert client.server_process is not None
                    assert client.server_process == mock_process

    @pytest.mark.asyncio
    async def test_start_server_already_running(self, mock_logger):
        """Test starting server when already running"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py'],
            'port': 8000
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        with patch.object(client, '_wait_for_server_ready', return_value=True):
            result = await client._start_server()

            assert result is True

    @pytest.mark.asyncio
    async def test_start_server_process_dies(self, mock_logger):
        """Test server startup when process dies immediately"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py'],
            'port': 8000,
            'startup_delay': 0.1
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process died
        mock_process.communicate.return_value = ("stdout", "stderr")
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        with patch('oai_agent_core.components.agent.agent_client.subprocess.Popen', return_value=mock_process):
            with patch.object(client, '_wait_for_server_ready', side_effect=[False, False]):
                result = await client._start_server()

                assert result is False

    @pytest.mark.asyncio
    async def test_stop_server_graceful(self, mock_logger):
        """Test graceful server shutdown"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py']
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock()
        client.server_process = mock_process

        with patch.object(client, '_wait_for_process', return_value=None):
            await client._stop_server()

            mock_process.terminate.assert_called_once()
            assert client.server_process is None

    @pytest.mark.asyncio
    async def test_stop_server_force_kill(self, mock_logger):
        """Test force kill when graceful shutdown times out"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py']
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        client.server_process = mock_process

        async def timeout_wait():
            raise asyncio.TimeoutError()

        with patch.object(client, '_wait_for_process', side_effect=[timeout_wait(), None]):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                await client._stop_server()

                mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_success(self, mock_logger):
        """Test successful server health check"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000',
            'health_endpoint': '/health'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 200

        mock_get_context = MagicMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_context

        mock_session_context = MagicMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_context):
            result = await client._wait_for_server_ready(health_check_attempts=1)

            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_timeout(self, mock_logger):
        """Test server health check timeout"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000',
            'health_endpoint': '/health'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.__aenter__.return_value.get.side_effect = Exception("Connection refused")

            result = await client._wait_for_server_ready(health_check_attempts=2)

            assert result is False


class TestAgentClientConnection:
    """Test client connection functionality"""

    @pytest.mark.asyncio
    async def test_connect_to_server_success(self, mock_logger):
        """Test successful server connection"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 200

        mock_get_context = MagicMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get.return_value = mock_get_context
        mock_session.closed = False

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch.object(client, 'agent_info', return_value={'agent_config': {}}):
                result = await client._connect_to_server()

                assert result is True
                assert client.connected is True

    @pytest.mark.asyncio
    async def test_connect_to_server_failure(self, mock_logger):
        """Test failed server connection"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.return_value.get.side_effect = Exception("Connection refused")

            result = await client._connect_to_server()

            assert result is False
            assert client.connected is False

    @pytest.mark.asyncio
    async def test_start_full_lifecycle(self, mock_logger):
        """Test full start lifecycle"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py']
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        with patch.object(client, '_start_server', return_value=True):
            with patch.object(client, '_connect_to_server', return_value=True):
                result = await client.start()

                assert result is True


class TestAgentClientRequests:
    """Test HTTP request functionality"""

    @pytest.mark.asyncio
    async def test_regular_request_success(self, mock_logger):
        """Test successful regular request"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json = AsyncMock(return_value={'result': 'success'})

        mock_request_context = MagicMock()
        mock_request_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request.return_value = mock_request_context
        mock_session.closed = False

        client.session = mock_session

        result = await client._regular_request('POST', 'http://localhost:8000/test', {'key': 'value'})

        assert result == {'result': 'success'}

    @pytest.mark.asyncio
    async def test_regular_request_error_status(self, mock_logger):
        """Test regular request with error status"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='Internal Server Error')

        mock_request_context = MagicMock()
        mock_request_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.request.return_value = mock_request_context
        mock_session.closed = False

        client.session = mock_session

        result = await client._regular_request('GET', 'http://localhost:8000/test')

        assert 'error' in result
        assert 'HTTP 500' in result['error']

    @pytest.mark.asyncio
    async def test_send_request_not_connected(self, mock_logger):
        """Test send_request when not connected"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = False

        with pytest.raises(ConnectionError, match="Client not connected"):
            await client.send_request('GET', '/test')

    @pytest.mark.asyncio
    async def test_parse_sse_line_data(self, mock_logger):
        """Test parsing SSE data line"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        result = client._parse_sse_line('data: {"message": "hello"}')

        assert result['type'] == 'data'
        assert result['content'] == {'message': 'hello'}

    @pytest.mark.asyncio
    async def test_parse_sse_line_done(self, mock_logger):
        """Test parsing SSE done signal"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        result = client._parse_sse_line('data: [DONE]')

        assert result['type'] == 'done'

    @pytest.mark.asyncio
    async def test_parse_sse_line_event(self, mock_logger):
        """Test parsing SSE event line"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        result = client._parse_sse_line('event: update')

        assert result['type'] == 'event'
        assert result['event'] == 'update'


class TestAgentClientChat:
    """Test chat functionality"""

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, mock_logger):
        """Test non-streaming chat"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        with patch.object(client, 'post', return_value={'response': 'Hello!'}):
            result = await client.chat('Hi', stream=False)

            assert result == {'response': 'Hello!'}

    @pytest.mark.asyncio
    async def test_chat_streaming(self, mock_logger):
        """Test streaming chat"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        async def mock_stream():
            yield {'type': 'data', 'content': 'Hello'}
            yield {'type': 'done'}

        with patch.object(client, 'post', return_value=mock_stream()):
            result = await client.chat('Hi', stream=True)

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0]['type'] == 'data'

    def test_chat_sync_non_streaming(self, mock_logger):
        """Test synchronous non-streaming chat"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True
        # Create a mock session that's not closed
        mock_session = MagicMock()
        mock_session.closed = False
        client.session = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Hello!'}

        with patch('oai_agent_core.components.agent.agent_client.requests.post', return_value=mock_response):
            result = client.chat_sync('Hi', stream=False)

            assert result == {'response': 'Hello!'}

    def test_chat_sync_streaming(self, mock_logger):
        """Test synchronous streaming chat"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True
        # Create a mock session that's not closed
        mock_session = MagicMock()
        mock_session.closed = False
        client.session = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"message": "Hello"}',
            'data: [DONE]'
        ]
        mock_response.close = MagicMock()

        with patch('oai_agent_core.components.agent.agent_client.requests.post', return_value=mock_response):
            result = client.chat_sync('Hi', stream=True)

            chunks = list(result)
            # Should have 2 chunks: one data chunk and one done chunk
            assert len(chunks) == 2
            assert chunks[0]['type'] == 'data'
            assert chunks[1]['type'] == 'done'


class TestAgentClientHealthAndInfo:
    """Test health check and info functionality"""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_logger):
        """Test health check when healthy"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True

        with patch.object(client, 'get', return_value={'status': 'ok'}):
            result = await client.health_check()

            assert result['status'] == 'healthy'

    @pytest.mark.asyncio
    async def test_agent_info(self, mock_logger):
        """Test getting agent info"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        expected_info = {'agent_config': {'name': 'test-agent'}}

        with patch.object(client, 'get', return_value=expected_info):
            result = await client.agent_info()

            assert result == expected_info


class TestAgentClientContextManager:
    """Test async context manager functionality"""

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_logger):
        """Test async context manager usage"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        with patch.object(AgentClient, 'start', return_value=True):
            with patch.object(AgentClient, 'stop'):
                async with AgentClient(server_config=config, logger=mock_logger) as client:
                    assert client is not None


class TestAgentClientUtilities:
    """Test utility functions"""

    def test_is_server_running_true(self, mock_logger):
        """Test is_server_running when server is running"""
        config = {
            'name': 'test-agent',
            'command': 'python',
            'args': ['server.py']
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        client.server_process = mock_process

        assert client.is_server_running() is True

    def test_is_server_running_false(self, mock_logger):
        """Test is_server_running when server is not running"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        assert client.is_server_running() is False

    def test_is_connected_true(self, mock_logger):
        """Test is_connected when connected"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)
        client.connected = True
        client.session = MagicMock()
        client.session.closed = False

        assert client.is_connected() is True

    def test_is_connected_false(self, mock_logger):
        """Test is_connected when not connected"""
        config = {
            'name': 'test-agent',
            'url': 'http://localhost:8000'
        }

        client = AgentClient(server_config=config, logger=mock_logger)

        assert client.is_connected() is False