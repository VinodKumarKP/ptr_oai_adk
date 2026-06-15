import pytest
import logging
import os
import time
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from oai_agent_core.utils.logger import Logger, LogLevel, LogFormat, PerformanceMetrics, get_logger

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"

@pytest.fixture
def logger(log_dir):
    return Logger(
        name="test_logger",
        log_dir=str(log_dir),
        log_level="DEBUG",
        enable_console=False,
        enable_file=True,
        enable_json=True
    )

def test_get_logger():
    # Patch _get_logger_from_config to None to test the original implementation
    with patch('oai_agent_core.utils.logger._get_logger_from_config', None):
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = get_logger()
            
            mock_get_logger.assert_called_with('agent_logger')
            mock_logger.handlers.clear.assert_called()
            assert mock_logger.propagate is False
            assert mock_logger.addHandler.call_count == 2 # File and Console

def test_logger_init(logger, log_dir):
    assert logger.name == "test_logger"
    assert logger.log_dir == log_dir
    assert (log_dir / "logging.log").exists()
    assert (log_dir / "logging.json").exists()

def test_log_levels(logger):
    with patch.object(logger.logger, 'log') as mock_log:
        logger.debug("debug msg")
        mock_log.assert_called_with(logging.DEBUG, "debug msg", exc_info=False)
        
        logger.info("info msg")
        mock_log.assert_called_with(logging.INFO, "info msg", exc_info=False)
        
        logger.warning("warning msg")
        mock_log.assert_called_with(logging.WARNING, "warning msg", exc_info=False)
        
        logger.error("error msg")
        mock_log.assert_called_with(logging.ERROR, "error msg", exc_info=False)
        
        logger.critical("critical msg")
        mock_log.assert_called_with(logging.CRITICAL, "critical msg", exc_info=False)

def test_exception_logging(logger):
    with patch.object(logger.logger, 'log') as mock_log:
        logger.exception("exception msg")
        mock_log.assert_called_with(logging.ERROR, "exception msg", exc_info=True)

def test_performance_metrics():
    metrics = PerformanceMetrics()
    metrics.record_duration("op1", 1.0)
    metrics.record_duration("op1", 2.0)
    
    stats = metrics.get_stats("op1")
    assert stats['count'] == 2
    assert stats['avg'] == 1.5
    assert stats['min'] == 1.0
    assert stats['max'] == 2.0
    assert stats['total'] == 3.0

def test_track_performance(logger):
    with logger.track_performance("test_op"):
        time.sleep(0.01)
        
    stats = logger.performance_metrics.get_stats("test_op")
    assert stats['count'] == 1
    assert stats['total'] > 0

def test_log_performance_stats(logger):
    with patch.object(logger, 'info') as mock_info:
        logger.performance_metrics.record_duration("op1", 1.0)
        logger.log_performance_stats("op1")
        mock_info.assert_called()
        assert "Performance stats for 'op1'" in mock_info.call_args[0][0]

def test_clear_old_logs(logger, log_dir):
    # Create an old log file
    old_file = log_dir / "old.log"
    old_file.touch()
    
    # Mock stat to return old time
    old_time = time.time() - (31 * 24 * 60 * 60) # 31 days ago
    
    # We need to patch Path.stat to return a mock object with st_mtime
    # But Path is used extensively.
    # Instead of patching Path.stat, let's use os.utime to set the file time
    os.utime(old_file, (old_time, old_time))
    
    # Verify it was set correctly (sanity check)
    assert old_file.stat().st_mtime == old_time
    
    logger.clear_old_logs(days_old=30)
    
    assert not old_file.exists()

def test_context_manager(logger):
    with logger as l:
        assert l == logger
    # Handlers flushed on exit

def test_json_formatter():
    from oai_agent_core.utils.logger import JsonFormatter
    formatter = JsonFormatter()
    record = logging.LogRecord("name", logging.INFO, "path", 1, "msg", (), None)
    json_str = formatter.format(record)
    data = json.loads(json_str)
    assert data['message'] == "msg"
    assert data['level'] == "INFO"
