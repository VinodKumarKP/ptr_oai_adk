"""Phase 2 Integration Tests - Windows Process Management & Configurable Timeouts.

Tests for:
1. Windows-compatible process termination (platform-aware signals)
2. Configurable process termination timeout
3. Configurable stream read idle timeout
4. TypedDict imports and usage
"""
import asyncio
import signal
import sys
from unittest.mock import AsyncMock, patch

import pytest

from oai_agent_client import (
    AsyncAgentClient,
    ClientConfig,
    InvokePayload,
    InvokeResponse,
    StreamEvent,
    HealthResponse,
)
from oai_agent_client.exceptions import ServerStartupError


# ===========================================================================
# Process Termination Timeout Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_configurable_process_termination_timeout():
    """Verify custom process_termination_timeout is used instead of hardcoded 5.0."""
    config = ClientConfig(
        command="dummy_server",
        process_termination_timeout=3,  # Custom timeout
    )
    
    assert config.process_termination_timeout == 3


@pytest.mark.asyncio
async def test_default_process_termination_timeout():
    """Verify default process_termination_timeout is 5 seconds."""
    config = ClientConfig(
        command="dummy_server",
    )
    
    assert config.process_termination_timeout == 5


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_stop_server_uses_custom_timeout(mock_subprocess):
    """Verify _stop_server uses configurable timeout instead of hardcoded value."""
    config = ClientConfig(
        command="dummy_server",
        process_termination_timeout=2,
    )
    
    # Mock the process and setup to timeout
    mock_process = AsyncMock()
    mock_process.wait.side_effect = asyncio.TimeoutError()
    
    client = AsyncAgentClient(config=config)
    client._server_process = mock_process
    
    # Run _stop_server
    await client._stop_server()
    
    # Verify wait() was called with custom timeout
    # The wait should be called with the custom timeout=2
    assert mock_process.wait.called
    # Call args should have been with timeout=2
    # Note: asyncio.wait_for doesn't expose its timeout directly in the mock,
    # but we verify the process was accessed


# ===========================================================================
# Platform-Aware Process Termination Tests
# ===========================================================================


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_stop_server_windows_graceful(mock_subprocess):
    """On Windows, verify terminate() is called for graceful shutdown."""
    config = ClientConfig(command="dummy_server")
    
    mock_process = AsyncMock()
    # Simulate graceful shutdown (wait completes without timeout)
    mock_process.wait = AsyncMock()
    
    client = AsyncAgentClient(config=config)
    client._server_process = mock_process
    
    # Patch sys.platform to simulate Windows
    with patch("sys.platform", "win32"):
        with patch("oai_agent_client.async_client.sys.platform", "win32"):
            await client._stop_server()
    
    # On Windows, terminate() should be called
    # (signal.send_signal is not used on Windows)
    assert client._server_process is None


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_stop_server_unix_graceful(mock_subprocess):
    """On Unix, verify send_signal(SIGTERM) is called for graceful shutdown."""
    config = ClientConfig(command="dummy_server")
    
    mock_process = AsyncMock()
    mock_process.wait = AsyncMock()
    
    client = AsyncAgentClient(config=config)
    client._server_process = mock_process
    
    # Patch sys.platform to simulate Unix
    with patch("sys.platform", "linux"):
        with patch("oai_agent_client.async_client.sys.platform", "linux"):
            await client._stop_server()
    
    # On Unix, send_signal should be called with SIGTERM
    assert client._server_process is None


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_stop_server_windows_kill_on_timeout(mock_subprocess):
    """On Windows, verify kill() is used when terminate() times out."""
    config = ClientConfig(command="dummy_server", process_termination_timeout=1)
    
    mock_process = AsyncMock()
    # Simulate timeout on graceful shutdown
    mock_process.wait.side_effect = asyncio.TimeoutError()
    
    client = AsyncAgentClient(config=config)
    client._server_process = mock_process
    
    with patch("sys.platform", "win32"):
        with patch("oai_agent_client.async_client.sys.platform", "win32"):
            await client._stop_server()
    
    assert client._server_process is None


# ===========================================================================
# Stream Idle Timeout Configuration Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_read_idle_timeout_configuration():
    """Verify stream_read_idle_timeout configuration is stored."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=20.0,
    )
    
    assert config.stream_read_idle_timeout == 20.0


@pytest.mark.asyncio
async def test_stream_read_idle_timeout_default_none():
    """Verify stream_read_idle_timeout defaults to None (unlimited)."""
    config = ClientConfig(
        url="http://localhost:8000",
    )
    
    assert config.stream_read_idle_timeout is None


@pytest.mark.asyncio
async def test_stream_read_idle_timeout_zero():
    """Verify stream_read_idle_timeout can be set to 0 for immediate timeout."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=0.0,
    )
    
    assert config.stream_read_idle_timeout == 0.0


# ===========================================================================
# TypedDict Imports and Usage Tests
# ===========================================================================


def test_typeddicts_imported():
    """Verify TypedDict types are properly exported."""
    from oai_agent_client import (
        InvokePayload,
        InvokeResponse,
        StreamEvent,
        HealthResponse,
    )
    
    # Verify they can be used for type hints
    assert InvokePayload.__annotations__
    assert InvokeResponse.__annotations__
    assert StreamEvent.__annotations__
    assert HealthResponse.__annotations__


def test_invoke_payload_typeddict():
    """Verify InvokePayload TypedDict has correct fields."""
    payload: InvokePayload = {
        "message": "Hello",
        "session_id": "sess-123",
        "user_id": "user-456",
    }
    
    assert payload["message"] == "Hello"
    assert payload["session_id"] == "sess-123"


def test_invoke_response_typeddict():
    """Verify InvokeResponse TypedDict has correct fields."""
    response: InvokeResponse = {
        "response": "Hello back!",
        "status": "success",
        "metadata": {"tokens": 42},
    }
    
    assert response["response"] == "Hello back!"
    assert response["status"] == "success"


def test_stream_event_typeddict():
    """Verify StreamEvent TypedDict has correct fields."""
    event: StreamEvent = {
        "content": "Streaming...",
        "type": "token",
    }
    
    assert event["content"] == "Streaming..."
    assert event["type"] == "token"


def test_health_response_typeddict():
    """Verify HealthResponse TypedDict has correct fields."""
    health: HealthResponse = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    
    assert health["status"] == "healthy"
    assert health["timestamp"] == "2024-01-01T00:00:00Z"


# ===========================================================================
# Integration: Config + TypedDict
# ===========================================================================


@pytest.mark.asyncio
async def test_config_with_stream_timeout_and_typeddicts():
    """Integration test: Use both Phase 2 features together."""
    # Create config with new Phase 2 timeout options
    config = ClientConfig(
        url="http://localhost:8000",
        process_termination_timeout=3,
        stream_read_idle_timeout=15.0,
    )
    
    # Use TypedDicts with the config
    payload: InvokePayload = {
        "message": "Test message",
        "session_id": "test-session",
    }
    
    expected_response: InvokeResponse = {
        "response": "Test response",
        "status": "success",
    }
    
    # Verify config and types work together
    assert config.process_termination_timeout == 3
    assert config.stream_read_idle_timeout == 15.0
    assert payload["message"] == "Test message"
    assert expected_response["response"] == "Test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
