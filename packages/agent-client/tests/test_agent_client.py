import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from oai_agent_client import ClientConfig, AgentClient
from oai_agent_client.exceptions import APIError, ConnectionError, ServerStartupError

# Mock server URL
MOCK_URL = "http://localhost:8000"

@pytest.fixture
def remote_config():
    """Fixture for a valid remote server configuration."""
    return ClientConfig(url=MOCK_URL, headers={"Authorization": "Bearer test"})

@pytest.fixture
def local_config():
    """Fixture for a valid local server configuration."""
    return ClientConfig(
        command="dummy_command",
        args=["--port", "8000"],
        headers={"Authorization": "Bearer test"}
    )

class TestClientConfig:
    """Tests for the ClientConfig model."""
    def test_valid_remote_config(self):
        config = ClientConfig(url=MOCK_URL, headers={"X-Token": "test"})
        # Pydantic's HttpUrl type adds a trailing slash.
        assert str(config.url) == MOCK_URL + "/"
        assert config.headers == {"X-Token": "test"}

    def test_valid_local_config(self):
        config = ClientConfig(command="python", args=["-m", "http.server"])
        assert config.command == "python"
        assert config.args == ["-m", "http.server"]

    def test_missing_url_and_command_fails(self):
        with pytest.raises(ValueError):
            ClientConfig()

    def test_both_url_and_command_fails(self):
        with pytest.raises(ValueError):
            ClientConfig(url=MOCK_URL, command="python")

@pytest.mark.asyncio
async def test_invoke_success(remote_config):
    """Test a successful invoke call."""
    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client, '_request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = {"response": "success"}
                
                response = await client.invoke("hello", config={"session_id": "123"})
                
                assert response == {"response": "success"}
                mock_request.assert_called_once_with(
                    "POST", "chat", data={"message": "hello", "session_id": "123"}
                )

@pytest.mark.asyncio
async def test_stream_success(remote_config):
    """Test a successful stream call."""
    async def mock_stream_gen():
        yield {"chunk": 1}
        yield {"chunk": 2}

    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client, '_stream_request', return_value=mock_stream_gen()) as mock_stream:
                chunks = [chunk async for chunk in client.stream("hello stream")]
                
                assert len(chunks) == 2
                assert chunks[0] == {"chunk": 1}
                mock_stream.assert_called_once_with(
                    "POST", "chat/stream", data={"message": "hello stream"}
                )

@pytest.mark.asyncio
async def test_api_error_handling(remote_config):
    """Test that APIError is raised on server error."""
    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=remote_config) as client:
            with patch.object(client._session, 'request') as mock_request:
                mock_response = AsyncMock()
                mock_response.status = 500
                mock_response.text.return_value = "Internal Server Error"
                
                mock_request.return_value.__aenter__.return_value = mock_response
                
                with pytest.raises(APIError) as excinfo:
                    await client.invoke("test")
                
                assert excinfo.value.status_code == 500
                assert "Internal Server Error" in excinfo.value.message

@pytest.mark.asyncio
async def test_connection_error_handling(remote_config):
    """Test that ConnectionError is raised on connection failure."""
    remote_config.url = "http://localhost:9999"
    with pytest.raises(ConnectionError):
        async with AgentClient(config=remote_config):
            pass

@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec')
async def test_local_server_management(mock_subprocess, local_config):
    """Test that the client starts and stops a local server process."""
    mock_process = AsyncMock()
    mock_process.stdout.at_eof.side_effect = [False, True]
    mock_process.stdout.readline.return_value = b"log line"
    mock_subprocess.return_value = mock_process

    with patch.object(AgentClient, '_wait_for_server', new_callable=AsyncMock):
        async with AgentClient(config=local_config) as client:
            mock_subprocess.assert_called_once_with(
                "dummy_command", "--port", "8000",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            assert client._server_process is not None

        mock_process.terminate.assert_called_once()

@pytest.mark.asyncio
@patch('asyncio.create_subprocess_exec', side_effect=OSError("File not found"))
async def test_server_startup_error(mock_subprocess, local_config):
    """Test that ServerStartupError is raised if the command fails."""
    with pytest.raises(ServerStartupError):
        async with AgentClient(config=local_config):
            pass
