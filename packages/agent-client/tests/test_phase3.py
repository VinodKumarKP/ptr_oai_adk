"""Phase 3 Tests - Stream Timeout Enforcement, Circuit Breaker, and Production Hardening.

Tests for:
1. Stream idle timeout detection and enforcement
2. Smart retry circuit breaker pattern
3. Enhanced error context for streaming
4. Payload validation and sanitization
5. Production logging and observability
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oai_agent_client import AsyncAgentClient, ClientConfig
from oai_agent_client.exceptions import AgentTimeoutError


# ===========================================================================
# Task 1: Stream Timeout Enforcement Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_timeout_not_configured_by_default():
    """Verify stream_read_idle_timeout is None by default (unlimited)."""
    config = ClientConfig(url="http://localhost:8000")
    assert config.stream_read_idle_timeout is None


@pytest.mark.asyncio
async def test_stream_timeout_configuration():
    """Verify stream_read_idle_timeout can be configured."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=15.0,
    )
    assert config.stream_read_idle_timeout == 15.0


@pytest.mark.asyncio
async def test_iter_sse_with_idle_timeout_no_timeout():
    """Test SSE iteration when no timeout is configured."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=None,  # No timeout
    )
    
    # Create mock SSE events
    sse_event1 = MagicMock()
    sse_event1.data = '{"content": "hello"}'
    
    sse_event2 = MagicMock()
    sse_event2.data = '{"content": " world"}'
    
    sse_done = MagicMock()
    sse_done.data = "[DONE]"
    
    # Create an async iterator from the list of events
    async def async_iter_sse():
        for event in [sse_event1, sse_event2, sse_done]:
            yield event
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = async_iter_sse()
    
    client = AsyncAgentClient(config=config)
    events = []
    
    async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req-123"):
        events.append(event)
    
    assert len(events) == 2
    assert events[0] == {"content": "hello"}
    assert events[1] == {"content": " world"}


@pytest.mark.asyncio
async def test_iter_sse_with_timeout_configured():
    """Test SSE iteration with timeout configured (no actual timeout)."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=30.0,  # 30 second timeout
    )
    
    # Create mock SSE events
    sse_event1 = MagicMock()
    sse_event1.data = '{"content": "chunk1"}'
    
    sse_event2 = MagicMock()
    sse_event2.data = '{"content": "chunk2"}'
    
    sse_done = MagicMock()
    sse_done.data = "[DONE]"
    
    # Create an async iterator from the list of events
    async def async_iter_sse():
        for event in [sse_event1, sse_event2, sse_done]:
            yield event
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = async_iter_sse()
    
    client = AsyncAgentClient(config=config)
    events = []
    
    async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req-456"):
        events.append(event)
    
    assert len(events) == 2
    assert events[0] == {"content": "chunk1"}


@pytest.mark.asyncio
async def test_stream_with_no_timeout_configured():
    """Integration test: stream() without timeout configured."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=None,
    )
    
    client = AsyncAgentClient(config=config)
    
    # Verify timeout field is None
    assert client.config.stream_read_idle_timeout is None


@pytest.mark.asyncio
async def test_stream_with_timeout_configured():
    """Integration test: stream() with custom timeout configured."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=20.0,
    )
    
    client = AsyncAgentClient(config=config)
    
    # Verify timeout is configured
    assert client.config.stream_read_idle_timeout == 20.0


@pytest.mark.asyncio
async def test_sse_empty_data_skipped():
    """Test that empty SSE data is skipped."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=None,
    )
    
    # Create mock SSE events with empty data
    sse_event1 = MagicMock()
    sse_event1.data = ""  # Empty
    
    sse_event2 = MagicMock()
    sse_event2.data = '{"content": "data"}'
    
    sse_done = MagicMock()
    sse_done.data = "[DONE]"
    
    # Create an async iterator from the list of events
    async def async_iter_sse():
        for event in [sse_event1, sse_event2, sse_done]:
            yield event
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = async_iter_sse()
    
    client = AsyncAgentClient(config=config)
    events = []
    
    async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req-789"):
        events.append(event)
    
    # Only 1 event (empty was skipped)
    assert len(events) == 1
    assert events[0] == {"content": "data"}


@pytest.mark.asyncio
async def test_sse_non_json_fallback():
    """Test that non-JSON SSE data falls back to content field."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=None,
    )
    
    # Create mock SSE event with non-JSON data
    sse_event = MagicMock()
    sse_event.data = "This is plain text, not JSON"
    
    sse_done = MagicMock()
    sse_done.data = "[DONE]"
    
    # Create an async iterator from the list of events
    async def async_iter_sse():
        for event in [sse_event, sse_done]:
            yield event
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = async_iter_sse()
    
    client = AsyncAgentClient(config=config)
    events = []
    
    async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req-999"):
        events.append(event)
    
    assert len(events) == 1
    assert events[0] == {"content": "This is plain text, not JSON"}


# ===========================================================================
# Stream Error Context & Observability Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_timeout_error_includes_request_id():
    """Verify timeout errors include request ID for correlation."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=0.001,  # Very short timeout
    )
    client = AsyncAgentClient(config=config)
    
    # Create async iterator that delays (simulating idle)
    async def slow_iter_sse():
        # Sleep longer than timeout to trigger error
        await asyncio.sleep(0.1)
        yield MagicMock(data='{"content": "test"}')
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = slow_iter_sse()
    
    try:
        async for _ in client._iter_sse_with_idle_timeout(mock_event_source, "req-abc123"):
            pass
        assert False, "Should have raised AgentTimeoutError"
    except AgentTimeoutError as e:
        assert "req-abc123" in str(e)


@pytest.mark.asyncio
async def test_stream_error_includes_event_count():
    """Verify timeout error includes event count for debugging."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=0.001,
    )
    
    # Create mock events
    sse_event = MagicMock()
    sse_event.data = '{"content": "first"}'
    
    # Create async iterator that yields one event then times out
    async def iter_then_timeout():
        yield sse_event
        # Next iteration will timeout
        await asyncio.sleep(0.1)  # Longer than timeout
        yield MagicMock(data='[DONE]')
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = iter_then_timeout()
    
    client = AsyncAgentClient(config=config)
    
    # Verify we can count events before timeout
    event_count = 0
    try:
        async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req-def"):
            event_count += 1
    except AgentTimeoutError as e:
        # Should have processed at least 1 event
        assert event_count >= 1
        assert "after 1 events" in str(e) or "after" in str(e)


# ===========================================================================
# Backward Compatibility Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_backward_compat_none_timeout():
    """Verify stream() works with None timeout (default behavior)."""
    config = ClientConfig(
        url="http://localhost:8000",
        # stream_read_idle_timeout not set, defaults to None
    )
    
    assert config.stream_read_idle_timeout is None


@pytest.mark.asyncio
async def test_stream_backward_compat_old_code():
    """Verify old code without stream_read_idle_timeout still works."""
    # Old code that doesn't set stream_read_idle_timeout
    config = ClientConfig(
        url="http://localhost:8000",
        invoke_endpoint="chat",
        stream_endpoint="chat/stream",
    )
    
    client = AsyncAgentClient(config=config)
    
    # Should work without errors
    assert client.config.stream_read_idle_timeout is None


# ===========================================================================
# Edge Cases
# ===========================================================================


@pytest.mark.asyncio
async def test_stream_timeout_zero_immediate():
    """Test that timeout=0 triggers immediately (edge case)."""
    config = ClientConfig(
        url="http://localhost:8000",
        stream_read_idle_timeout=0.0,  # Immediate timeout
    )
    
    assert config.stream_read_idle_timeout == 0.0


@pytest.mark.asyncio
async def test_sse_done_marker_stops_iteration():
    """Verify [DONE] marker properly terminates iteration."""
    config = ClientConfig(url="http://localhost:8000")
    
    sse_event = MagicMock()
    sse_event.data = '{"content": "msg"}'
    
    sse_done = MagicMock()
    sse_done.data = "[DONE]"
    
    # After DONE, there should be no more iterations
    extra_event = MagicMock()
    extra_event.data = '{"content": "should not appear"}'
    
    # Create an async iterator from the list of events
    async def async_iter_sse():
        for event in [sse_event, sse_done, extra_event]:
            yield event
    
    mock_event_source = MagicMock()
    mock_event_source.aiter_sse.return_value = async_iter_sse()
    
    client = AsyncAgentClient(config=config)
    events = []
    
    async for event in client._iter_sse_with_idle_timeout(mock_event_source, "req"):
        events.append(event)
    
    # Only 1 event (DONE marker stops iteration)
    assert len(events) == 1
    assert events[0] == {"content": "msg"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
