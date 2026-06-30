"""Coverage for sync_client module - targeting untested paths."""
import json
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import httpx
import pytest

from oai_agent_client.sync_client import SyncAgentClient
from oai_agent_client.config import ClientConfig
from oai_agent_client.exceptions import (
    AgentConnectionError,
    AgentTimeoutError,
    ConfigurationError,
)


class TestSyncClientInit:
    """Test SyncAgentClient initialization."""

    def test_init_with_command_raises_error(self):
        """Test that passing command to SyncAgentClient raises ConfigurationError."""
        # Create a config with command and no url (valid for AsyncAgentClient)
        config = ClientConfig(command="python -m agent", args=[])
        with pytest.raises(ConfigurationError, match="AsyncAgentClient"):
            SyncAgentClient(config=config)

    def test_init_valid_config(self):
        """Test valid SyncAgentClient initialization."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)
        assert client.config == config
        assert client._client is None


class TestSyncClientLifecycle:
    """Test client lifecycle methods."""

    def test_context_manager_enter(self):
        """Test __enter__ creates client and waits for server."""
        config = ClientConfig(url="http://localhost:8000", startup_timeout=1)
        client = SyncAgentClient(config=config)

        with patch("oai_agent_client.sync_client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = MagicMock(status_code=200)
            mock_client_class.return_value = mock_client

            with client:
                assert client._client is mock_client

    def test_enter_connection_timeout(self):
        """Test __enter__ raises when server doesn't respond."""
        config = ClientConfig(url="http://localhost:8000", startup_timeout=1)
        client = SyncAgentClient(config=config)

        with patch("oai_agent_client.sync_client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.HTTPError("Connection failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(AgentConnectionError, match="did not become healthy"):
                with client:
                    pass

    def test_close_when_not_started(self):
        """Test close when client was never started."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        # Should not raise
        client.close()
        assert client._client is None

    def test_close_with_error(self):
        """Test close when client.close() raises."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("Close failed")
        client._client = mock_client

        # Should log error but not raise
        client.close()
        assert client._client is None


class TestSyncClientHeaders:
    """Test header management."""

    def test_update_headers_without_client(self):
        """Test updating headers before client is created."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        client.update_headers(Authorization="Bearer token")
        assert client._headers["Authorization"] == "Bearer token"

    def test_update_headers_with_client(self):
        """Test updating headers on live client."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        client._client = mock_client

        client.update_headers(Authorization="Bearer token")
        mock_client.headers.update.assert_called_once()

    def test_update_headers_client_error(self):
        """Test update_headers when client mutation fails."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.headers.update.side_effect = Exception("Failed")
        client._client = mock_client

        # Should log but not raise
        client.update_headers(Authorization="Bearer token")


class TestSyncClientHealth:
    """Test health check methods."""

    def test_check_health_not_started(self):
        """Test health check when client not started."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        with pytest.raises(AgentConnectionError, match="not started"):
            client.check_health()

    def test_check_health_success(self):
        """Test successful health check."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(status_code=200)
        client._client = mock_client

        assert client.check_health() is True

    def test_check_health_failure(self):
        """Test failed health check."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        client._client = mock_client

        assert client.check_health() is False


class TestSyncClientRequest:
    """Test request execution."""

    def test_request_without_client(self):
        """Test request when client not started."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        with pytest.raises(AgentConnectionError, match="not started"):
            client._request("GET", "/health")

    def test_request_connect_timeout(self):
        """Test request with connection timeout."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.request.side_effect = httpx.ConnectTimeout("timeout")
        client._client = mock_client

        with pytest.raises(AgentTimeoutError, match="Connection timeout"):
            client._request("GET", "/invoke")

    def test_request_read_timeout(self):
        """Test request with read timeout."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.request.side_effect = httpx.ReadTimeout("timeout")
        client._client = mock_client

        with pytest.raises(AgentTimeoutError, match="Read timeout"):
            client._request("POST", "/chat")

    def test_request_generic_timeout(self):
        """Test request with generic timeout."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.request.side_effect = httpx.TimeoutException("timeout")
        client._client = mock_client

        with pytest.raises(AgentTimeoutError, match="Request timeout"):
            client._request("GET", "/status")

    def test_request_http_error(self):
        """Test request with generic HTTP error."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.request.side_effect = httpx.HTTPError("Connection failed")
        client._client = mock_client

        with pytest.raises(AgentConnectionError, match="Connection failed"):
            client._request("GET", "/invoke")

    def test_request_empty_response(self):
        """Test request with 204 No Content response."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_client.request.return_value = MagicMock(
            status_code=204, text="", headers={"X-Request-ID": "req-123"}
        )
        client._client = mock_client

        result = client._request("DELETE", "/resource")
        assert result == {}

    def test_request_json_decode_error(self):
        """Test request with invalid JSON response."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_response = MagicMock(
            status_code=200, text="invalid json", headers={"X-Request-ID": "req-123"}
        )
        mock_response.json.side_effect = json.JSONDecodeError("error", "", 0)
        mock_client.request.return_value = mock_response
        client._client = mock_client

        with pytest.raises(json.JSONDecodeError):
            client._request("GET", "/data")


class TestSyncClientInvoke:
    """Test public API methods."""

    def test_invoke_success(self):
        """Test invoke with successful response."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_response = MagicMock(
            status_code=200,
            text='{"response":"hello"}',
            headers={"X-Request-ID": "req-123"}
        )
        mock_response.json.return_value = {"response": "hello"}
        mock_client.request.return_value = mock_response
        client._client = mock_client

        result = client.invoke("hi there")
        assert result["response"] == "hello"

    def test_invoke_with_config(self):
        """Test invoke with additional config."""
        config = ClientConfig(url="http://localhost:8000")
        client = SyncAgentClient(config=config)

        mock_client = MagicMock()
        mock_response = MagicMock(
            status_code=200,
            text='{"response":"hi"}',
            headers={"X-Request-ID": "req-123"}
        )
        mock_response.json.return_value = {"response": "hi"}
        mock_client.request.return_value = mock_response
        client._client = mock_client

        result = client.invoke("hello", config={"temperature": 0.5})
        assert "response" in result
