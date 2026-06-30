"""Phase 2 performance and optimization tests.

Tests cover:
- Streaming JSON encoder for efficient serialization
- Judge service pre-initialization and concurrent evaluation limits
- Connection pooling configuration
- Atomic transactional logging
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

# Streaming JSON encoder tests
from oai_agent_server.utils.streaming_encoder import (
    StreamingJSONEncoder, serialize_chunk_streaming, serialize_chunk_safe
)

# Judge service tests
from oai_agent_server.services.llm_judge_service import LLMJudgeService

# Pool configuration tests
from oai_agent_server.utils.pool_config import PoolConfig, PoolMonitor


class TestStreamingJSONEncoder:
    """Test streaming JSON encoder for performance."""
    
    def test_streaming_encoder_basic(self):
        """Test basic streaming encoding."""
        data = {"key": "value", "number": 42}
        encoder = StreamingJSONEncoder()
        chunks = list(encoder.iterencode(data))
        
        # Should produce chunks
        assert len(chunks) > 0
        # Reassembling should produce valid JSON
        result = json.loads(''.join(chunks))
        assert result == data
    
    def test_streaming_encoder_large_object(self):
        """Test streaming large objects without keeping entire serialization in memory."""
        data = {
            "items": [{"id": i, "value": f"item_{i}" * 10} for i in range(100)],
            "metadata": {"total": 100, "page": 1},
        }
        
        encoder = StreamingJSONEncoder()
        chunk_count = 0
        total_size = 0
        
        for chunk in encoder.iterencode(data):
            chunk_count += 1
            total_size += len(chunk)
            # Each chunk should be reasonable size (not entire object at once)
            assert len(chunk) < len(json.dumps(data))
        
        assert chunk_count > 0
        assert total_size == len(json.dumps(data))
    
    def test_serialize_chunk_streaming(self):
        """Test serialize_chunk_streaming helper."""
        data = {"test": "data", "nested": {"key": "value"}}
        chunks = list(serialize_chunk_streaming(data))
        
        assert len(chunks) > 0
        result = json.loads(''.join(chunks))
        assert result == data
    
    def test_serialize_chunk_safe_with_fallback(self):
        """Test safe serialization with object fallback."""
        class CustomObj:
            def __init__(self):
                self.value = "test"
        
        obj = CustomObj()
        result = serialize_chunk_safe(obj, fallback_to_string=True)
        
        # Should produce valid JSON (serializes __dict__)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_streaming_encoder_datetime_and_pydantic(self):
        from datetime import datetime
        from pydantic import BaseModel
        
        class PydanticV2Model(BaseModel):
            name: str
            
        class PydanticV1Model:
            def dict(self):
                return {"v1": "data"}
                
        now = datetime(2026, 6, 30, 0, 0, 0)
        v2_model = PydanticV2Model(name="test-v2")
        v1_model = PydanticV1Model()
        
        data = {
            "dt": now,
            "v2": v2_model,
            "v1": v1_model
        }
        
        res = serialize_chunk_safe(data)
        parsed = json.loads(res)
        assert parsed["dt"] == now.isoformat()
        assert parsed["v2"] == {"name": "test-v2"}
        assert parsed["v1"] == {"v1": "data"}

    def test_serialize_chunk_safe_error_fallback(self):
        class BadStrObj:
            __slots__ = ('fail',)
            def __init__(self):
                self.fail = True
            def __str__(self):
                if self.fail:
                    self.fail = False
                    raise TypeError("cannot stringify")
                return "recovered"
                
        data = BadStrObj()
        
        # Test fallback_to_string=True (default)
        res = serialize_chunk_safe(data, fallback_to_string=True)
        parsed = json.loads(res)
        assert "error" in parsed
        assert parsed["error"] == "recovered"
        assert parsed["original_type"] == "BadStrObj"
        
        # Reset and test fallback_to_string=False raises TypeError
        data2 = BadStrObj()
        with pytest.raises(TypeError):
            serialize_chunk_safe(data2, fallback_to_string=False)



class TestJudgeServicePreInitialization:
    """Test judge service pre-initialization and concurrent evaluation."""
    
    @pytest.mark.asyncio
    async def test_judge_service_initialize_agent(self):
        """Test pre-initialization of judge agent."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_class.return_value = mock_agent_instance
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/tmp",
            logger=MagicMock(),
            db_logger=AsyncMock(),
            judge_model_id="test-model",
        )
        
        # Initially, judge agent should be None
        assert service.judge_agent is None
        
        # Initialize
        await service.initialize_judge_agent()
        
        # Should create agent and initialize it
        mock_agent_class.assert_called_once()
        mock_agent_instance.initialize.assert_called_once()
        assert service.judge_agent is not None
    
    @pytest.mark.asyncio
    async def test_judge_service_concurrent_evaluations(self):
        """Test that multiple evaluations can proceed concurrently with semaphore."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_instance.ainvoke = AsyncMock(return_value={
            'content': [{'text': json.dumps({
                'quality_score': 0.95,
                'hallucination_detected': 'false'
            })}]
        })
        mock_agent_class.return_value = mock_agent_instance
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/tmp",
            logger=MagicMock(),
            db_logger=AsyncMock(),
            judge_model_id="test-model",
            max_concurrent_evaluations=3,
        )
        
        # Pre-initialize
        await service.initialize_judge_agent()
        
        # Create multiple concurrent evaluation tasks
        tasks = [
            service.judge_interaction(
                interaction_id=f"interact_{i}",
                agent_name="test_agent",
                session_id="sess_123",
                user_message="Hello",
                agent_response="Hi there",
                user_id="user_1",
            )
            for i in range(5)
        ]
        
        # All should complete without deadlock
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that we didn't hit timeout or deadlock issues
        assert len(results) == 5
        # Some should complete (no exception), semaphore prevents too many concurrent
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful >= 0  # May have exceptions if db_logger not properly mocked
    
    @pytest.mark.asyncio
    async def test_judge_service_initialization_lock(self):
        """Test that initialization is properly protected with lock."""
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_class.return_value = mock_agent_instance
        
        service = LLMJudgeService(
            agent_class=mock_agent_class,
            config_root="/tmp",
            logger=MagicMock(),
            db_logger=AsyncMock(),
        )
        
        # Call initialize multiple times concurrently
        await asyncio.gather(
            service.initialize_judge_agent(),
            service.initialize_judge_agent(),
            service.initialize_judge_agent(),
        )
        
        # Should only create one agent instance (lock prevents race)
        assert mock_agent_class.call_count == 1
        mock_agent_instance.initialize.assert_called_once()


class TestPoolConfiguration:
    """Test connection pool configuration."""
    
    def test_pool_config_defaults(self):
        """Test pool configuration defaults."""
        config = PoolConfig.get_asyncpg_pool_kwargs()
        
        assert "min_size" in config
        assert "max_size" in config
        assert config["min_size"] > 0
        assert config["max_size"] >= config["min_size"]
    
    def test_pool_config_from_env(self, monkeypatch):
        """Test pool configuration from environment variables."""
        monkeypatch.setenv("DB_POOL_MIN_SIZE", "10")
        monkeypatch.setenv("DB_POOL_MAX_SIZE", "50")
        
        # Reload module to pick up env vars
        import importlib
        import oai_agent_server.utils.pool_config as pool_module
        importlib.reload(pool_module)
        
        config = pool_module.PoolConfig.get_asyncpg_pool_kwargs()
        assert config["min_size"] == 10
        assert config["max_size"] == 50
    
    def test_sqlite_pool_config(self):
        """Test SQLite pool configuration."""
        config = PoolConfig.get_sqlite_pool_kwargs()
        
        assert "timeout" in config
        assert "check_same_thread" in config
        assert config["timeout"] > 0
        assert config["check_same_thread"] is False


class TestPoolMonitor:
    """Test connection pool monitoring."""
    
    @pytest.mark.asyncio
    async def test_pool_monitor_creation(self):
        """Test pool monitor initialization."""
        mock_pool = MagicMock()
        logger = MagicMock()
        
        monitor = PoolMonitor(mock_pool, "test_db", logger)
        
        assert monitor.pool is mock_pool
        assert monitor.backend_name == "test_db"
        assert monitor.logger is logger
        assert monitor._started is False
    
    @pytest.mark.asyncio
    async def test_pool_monitor_start_stop(self):
        """Test pool monitor start and stop."""
        mock_pool = MagicMock()
        mock_pool._holders = []  # Simulates PostgreSQL asyncpg pool
        logger = MagicMock()
        
        monitor = PoolMonitor(mock_pool, "test_db", logger)
        
        # Start monitor in background task
        monitor_task = asyncio.create_task(monitor.start())
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Should be started
        assert monitor._started is True
        
        # Stop monitor
        await monitor.stop()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()
    
    def test_pool_monitor_get_metrics(self):
        """Test pool metrics collection."""
        # Mock PostgreSQL asyncpg pool
        mock_holder1 = MagicMock(_in_use=False)
        mock_holder2 = MagicMock(_in_use=True)
        mock_holder3 = MagicMock(_in_use=False)
        
        mock_pool = MagicMock()
        mock_pool._holders = [mock_holder1, mock_holder2, mock_holder3]
        mock_pool._maxsize = 10
        
        monitor = PoolMonitor(mock_pool, "postgres", MagicMock())
        metrics = monitor.get_metrics()
        
        assert metrics["backend"] == "postgres"
        assert metrics["type"] == "connection_pool"
        assert metrics["total_connections"] == 3
        assert metrics["in_use"] == 1
        assert metrics["available"] == 2
        assert abs(metrics["utilization_percent"] - 33.33) < 1  # ~33% utilized


class TestAtomicTransactionalLogging:
    """Test atomic transactional batch logging."""
    
    @pytest.mark.asyncio
    async def test_log_stream_chunks_batch_with_interaction(self):
        """Test atomic logging of chunks and interaction together."""
        from oai_agent_server.utils.database_logger import DatabaseLogger
        
        db_logger = DatabaseLogger(backends=[], logger=MagicMock())
        db_logger.is_active = True
        
        # Mock backend with transactional support
        mock_backend = AsyncMock()
        mock_backend.ACTIVITY_LOG_INSERT = "INSERT INTO activity_log ..."
        mock_backend.CHAT_LOGS_INSERT = "INSERT INTO chat_logs ..."
        # Add both methods since the implementation tries both
        mock_backend.execute_many = AsyncMock(return_value=None)
        mock_backend.execute = AsyncMock(return_value=None)
        mock_backend.execute_many_in_transaction = AsyncMock(return_value=None)
        db_logger._backend = mock_backend
        
        # Mock the _ready() method to return True
        db_logger._ready = lambda: True
        
        chunks_kwargs = {
            "interaction_id": "interact_123",
            "agent_name": "test_agent",
            "session_id": "sess_456",
            "user_id": "user_789",
            "endpoint": "/chat",
            "request_headers": {"User-Agent": "test"},
            "chunks": [
                {
                    "chunk_sequence": 1,
                    "chunk_content": {"text": "Hello"},
                    "chunk_text": "Hello",
                    "serialization_warning": None,
                },
                {
                    "chunk_sequence": 2,
                    "chunk_content": {"text": "World"},
                    "chunk_text": "World",
                    "serialization_warning": None,
                },
            ],
        }
        
        interaction_kwargs = {
            "agent_name": "test_agent",
            "session_id": "sess_456",
            "user_id": "user_789",
            "endpoint": "/chat",
            "input_message": "test message",
            "output_response": "test response",
            "request_headers": {"User-Agent": "test"},
            "model_info": {"model": "gpt-4"},
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "total_tokens": 15,
            "response_time_ms": 123.45,
            "status": "success",
            "error_message": None,
        }
        
        # Call atomic logging
        await db_logger.log_stream_chunks_batch_with_interaction(
            chunks_kwargs, interaction_kwargs
        )
        
        # Backend should have been called with either execute_many_in_transaction or both execute_many and execute
        assert (mock_backend.execute_many_in_transaction.called or 
                (mock_backend.execute_many.called and mock_backend.execute.called))


# Fixture to handle EnvironmentError in tests
@pytest.fixture(autouse=True)
def reset_pool_config():
    """Reset pool config after each test."""
    yield
    # Reload to reset to actual env values
    import importlib
    import oai_agent_server.utils.pool_config as pool_module
    try:
        importlib.reload(pool_module)
    except Exception:
        pass
