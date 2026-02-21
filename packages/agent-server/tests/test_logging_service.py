import pytest
from unittest.mock import MagicMock, AsyncMock
from oai_agent_server.services.logging_service import LoggingService
from oai_agent_server.exceptions import DatabaseNotAvailableException

@pytest.fixture
def logging_service(mock_db_logger):
    return LoggingService(mock_db_logger, "test_agent")

@pytest.mark.asyncio
async def test_get_logs(logging_service):
    response = await logging_service.get_logs()
    assert response.status_code == 200
    logging_service.db_logger.get_logs.assert_called_once()

@pytest.mark.asyncio
async def test_get_logs_db_init_retry(logging_service):
    # Test retry logic when db is inactive
    logging_service.db_logger.is_active = False
    
    # First call fails, second call succeeds (simulated by side effect on is_active check?)
    # The code checks is_active, calls initialize, then checks is_active again.
    # We can mock initialize to set is_active = True
    async def mock_init():
        logging_service.db_logger.is_active = True
        
    logging_service.db_logger.initialize = AsyncMock(side_effect=mock_init)
    
    response = await logging_service.get_logs()
    assert response.status_code == 200
    logging_service.db_logger.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_get_session_logs(logging_service):
    logging_service.db_logger.get_activity_logs = AsyncMock(return_value=[])
    response = await logging_service.get_session_logs("sess1")
    assert response.status_code == 200
    logging_service.db_logger.get_activity_logs.assert_called_once()

@pytest.mark.asyncio
async def test_get_session_logs_db_not_available(logging_service):
    logging_service.db_logger.is_active = False
    logging_service.db_logger.initialize = AsyncMock() # Fails to set active
    
    with pytest.raises(DatabaseNotAvailableException):
        await logging_service.get_session_logs("sess1")

@pytest.mark.asyncio
async def test_get_log_stats(logging_service):
    response = await logging_service.get_log_stats()
    assert response.status_code == 200
    logging_service.db_logger.get_stats.assert_called_once()

@pytest.mark.asyncio
async def test_get_log_stats_db_not_available(logging_service):
    logging_service.db_logger.is_active = False
    logging_service.db_logger.initialize = AsyncMock()
    
    with pytest.raises(DatabaseNotAvailableException):
        await logging_service.get_log_stats()

@pytest.mark.asyncio
async def test_get_user_stats(logging_service):
    response = await logging_service.get_user_stats()
    assert response.status_code == 200
    logging_service.db_logger.get_user_stats.assert_called_once()

@pytest.mark.asyncio
async def test_get_user_stats_db_not_available(logging_service):
    logging_service.db_logger.is_active = False
    logging_service.db_logger.initialize = AsyncMock()
    
    with pytest.raises(DatabaseNotAvailableException):
        await logging_service.get_user_stats()

@pytest.mark.asyncio
async def test_get_user_stats_fallback(logging_service):
    # Test fallback if get_user_stats method doesn't exist on logger
    del logging_service.db_logger.get_user_stats
    response = await logging_service.get_user_stats()
    assert response.status_code == 200
    assert response.body.decode() == '{"agent_name":"test_agent","database_active":true,"user_statistics":{}}'
