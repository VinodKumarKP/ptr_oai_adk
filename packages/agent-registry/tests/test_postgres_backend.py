import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from oai_agent_registry.services.db.database_logger import PostgresBackend

@pytest.mark.asyncio
async def test_postgres_backend_create_schema():
    backend = PostgresBackend()
    backend._pool = MagicMock()
    mock_conn = AsyncMock()
    
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    backend._pool.acquire = MagicMock(return_value=mock_acquire)
    
    logger = MagicMock()
    await backend._create_schema(logger=logger)
    
    assert mock_conn.execute.call_count > 0
    logger.info.assert_called()

@pytest.mark.asyncio
async def test_postgres_backend_create_schema_migration_error():
    backend = PostgresBackend()
    backend._pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = [None, None, Exception("Migration error"), None, None, None, None, None, None, None, None, None, None]
    
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock()
    backend._pool.acquire = MagicMock(return_value=mock_acquire)
    
    # Should not raise exception
    await backend._create_schema()
