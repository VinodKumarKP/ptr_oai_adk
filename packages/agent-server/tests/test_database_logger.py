import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from datetime import datetime

# Mock asyncpg before importing DatabaseLogger
if 'asyncpg' not in sys.modules:
    asyncpg = MagicMock()
    sys.modules['asyncpg'] = asyncpg
    sys.modules['asyncpg.pool'] = MagicMock()

from oai_agent_server.utils.database_logger import DatabaseLogger

@pytest.fixture
def db_logger():
    logger = DatabaseLogger(logger=MagicMock())
    logger.pool = MagicMock()
    logger.pool.acquire.return_value.__aenter__.return_value = AsyncMock()
    # Reset state
    logger.is_active = False
    logger._db_logging_enabled = False
    return logger

@pytest.mark.asyncio
async def test_initialize_success(db_logger):
    with patch('oai_agent_server.utils.database_logger.ASYNCPG_AVAILABLE', True):
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            with patch.dict('os.environ', {'DB_LOGGING_ENABLED': 'true'}):
                await db_logger.initialize()
                
                mock_create_pool.assert_called_once()
                assert db_logger._db_logging_enabled is True

@pytest.mark.asyncio
async def test_initialize_disabled(db_logger):
    with patch('oai_agent_server.utils.database_logger.ASYNCPG_AVAILABLE', True):
        with patch.dict('os.environ', {'DB_LOGGING_ENABLED': 'false'}):
            await db_logger.initialize()
            assert db_logger.is_active is False
            assert db_logger._db_logging_enabled is False

@pytest.mark.asyncio
async def test_initialize_asyncpg_not_available(db_logger):
    with patch('oai_agent_server.utils.database_logger.ASYNCPG_AVAILABLE', False):
        await db_logger.initialize()
        assert db_logger.is_active is False
        db_logger.logger.warning.assert_called_with("asyncpg not installed. Database logging disabled.")

@pytest.mark.asyncio
async def test_initialize_exception(db_logger):
    with patch('oai_agent_server.utils.database_logger.ASYNCPG_AVAILABLE', True):
        with patch('asyncpg.create_pool', side_effect=Exception("Connection failed")):
            with patch.dict('os.environ', {'DB_LOGGING_ENABLED': 'true'}):
                await db_logger.initialize()
                assert db_logger.is_active is False
                db_logger.logger.warning.assert_called()

@pytest.mark.asyncio
async def test_create_table_exception(db_logger):
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.side_effect = Exception("Table error")
    
    with pytest.raises(Exception, match="Table error"):
        await db_logger._create_table()

@pytest.mark.asyncio
async def test_log_interaction(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    
    await db_logger.log_interaction(
        agent_name="test",
        session_id="sess1",
        user_id="user1",
        endpoint="/chat",
        input_message="hi",
        output_response="hello",
        token_usage={'total_tokens': 10}
    )
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.assert_called_once()

@pytest.mark.asyncio
async def test_log_interaction_disabled(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = False
    
    await db_logger.log_interaction("test", "s1", "u1", "/chat", "hi", "hello")
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.assert_not_called()

@pytest.mark.asyncio
async def test_log_interaction_exception(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.side_effect = Exception("DB Error")
    
    await db_logger.log_interaction("test", "s1", "u1", "/chat", "hi", "hello")
    db_logger.logger.warning.assert_called()

@pytest.mark.asyncio
async def test_log_stream_chunk(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    
    await db_logger.log_stream_chunk(
        agent_name="test",
        session_id="sess1",
        user_id="user1",
        endpoint="/stream",
        chunk_sequence=1,
        chunk_content="chunk"
    )
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.assert_called_once()

@pytest.mark.asyncio
async def test_log_stream_chunk_exception(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.execute.side_effect = Exception("DB Error")
    
    await db_logger.log_stream_chunk("test", "s1", "u1", "/stream", 1, "chunk")
    db_logger.logger.warning.assert_called()

@pytest.mark.asyncio
async def test_log_stream_chunks_batch(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    
    chunks = [{'chunk_sequence': 1, 'chunk_content': 'c1'}]
    await db_logger.log_stream_chunks_batch(
        agent_name="test",
        session_id="sess1",
        user_id="user1",
        endpoint="/stream",
        chunks=chunks
    )
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.executemany.assert_called_once()

@pytest.mark.asyncio
async def test_log_stream_chunks_batch_exception(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.executemany.side_effect = Exception("DB Error")
    
    await db_logger.log_stream_chunks_batch("test", "s1", "u1", "/stream", [{'chunk_sequence': 1}])
    db_logger.logger.warning.assert_called()

@pytest.mark.asyncio
async def test_get_logs_filters(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.fetch.return_value = []
    
    # Test with all filters
    await db_logger.get_logs(
        agent_name="test",
        session_id="s1",
        user_id="u1",
        endpoint="/chat",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow(),
        status="success"
    )
    conn.fetch.assert_called_once()
    # Check if query contains WHERE clauses
    query = conn.fetch.call_args[0][0]
    assert "agent_name =" in query
    assert "session_id =" in query
    assert "user_id =" in query

@pytest.mark.asyncio
async def test_get_logs_exception(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.fetch.side_effect = Exception("DB Error")
    
    logs = await db_logger.get_logs()
    assert logs == []
    db_logger.logger.error.assert_called()

@pytest.mark.asyncio
async def test_get_activity_logs(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    
    mock_row = {
        'id': 1, 'timestamp': datetime.utcnow(), 'agent_name': 'test',
        'session_id': 's1', 'user_id': 'u1', 'endpoint': '/stream',
        'chunk_sequence': 1, 'chunk_content': '"content"', 'chunk_text': 'text',
        'serialization_warning': None, 'request_headers': None,
        'created_at': datetime.utcnow()
    }
    conn.fetch.return_value = [mock_row]
    
    logs = await db_logger.get_activity_logs(session_id="s1")
    assert len(logs) == 1
    assert logs[0]['chunk_sequence'] == 1

@pytest.mark.asyncio
async def test_get_activity_logs_exception(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    conn.fetch.side_effect = Exception("DB Error")
    
    logs = await db_logger.get_activity_logs()
    assert logs == []
    db_logger.logger.error.assert_called()

@pytest.mark.asyncio
async def test_get_stats(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    mock_row = {
        'total_interactions': 10, 'unique_sessions': 5, 'unique_users': 2,
        'avg_response_time': 100.0, 'max_response_time': 200.0, 'min_response_time': 50.0,
        'total_tokens_sum': 1000, 'successful_interactions': 9, 'failed_interactions': 1
    }
    conn.fetchrow.return_value = mock_row
    
    stats = await db_logger.get_stats(agent_name="test", user_id="u1")
    assert stats['total_interactions'] == 10
    
    # Test exception
    conn.fetchrow.side_effect = Exception("DB Error")
    stats = await db_logger.get_stats()
    assert stats == {}

@pytest.mark.asyncio
async def test_get_user_stats(db_logger):
    db_logger.is_active = True
    db_logger._db_logging_enabled = True
    conn = db_logger.pool.acquire.return_value.__aenter__.return_value
    mock_row = {
        'user_id': 'u1',
        'total_interactions': 10, 'unique_sessions': 5,
        'avg_response_time': 100.0, 'max_response_time': 200.0, 'min_response_time': 50.0,
        'total_tokens_sum': 1000, 'successful_interactions': 9, 'failed_interactions': 1
    }
    conn.fetch.return_value = [mock_row]
    
    stats = await db_logger.get_user_stats(agent_name="test")
    assert len(stats) == 1
    assert stats[0]['user_id'] == 'u1'
    
    # Test exception
    conn.fetch.side_effect = Exception("DB Error")
    stats = await db_logger.get_user_stats()
    assert stats == []

@pytest.mark.asyncio
async def test_serialize_for_json(db_logger):
    # Test recursion limit
    deep_obj = {}
    curr = deep_obj
    for _ in range(15):
        curr['a'] = {}
        curr = curr['a']
    
    serialized = db_logger._serialize_for_json(deep_obj)
    assert isinstance(serialized, dict)

@pytest.mark.asyncio
async def test_close(db_logger):
    db_logger.is_active = True
    db_logger.pool.close = AsyncMock()
    await db_logger.close()
    db_logger.pool.close.assert_called_once()
    assert db_logger.is_active is False
