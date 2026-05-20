
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock

from oai_platform_core.logging_utils import get_logger


def test_get_logger_default_creation():
    """Test that get_logger creates a logger with default settings."""
    logger_name = "test_default_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    with patch("oai_platform_core.logging_utils.RotatingFileHandler") as mock_file_handler, \
         patch("oai_platform_core.logging_utils.logging.StreamHandler") as mock_stream_handler:

        logger = get_logger(name=logger_name)

        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2
        mock_file_handler.assert_called_once()
        mock_stream_handler.assert_called_once()
    
    logger.handlers = []


def test_get_logger_custom_parameters():
    """Test that get_logger respects custom parameters."""
    logger_name = "test_custom_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    
    log_dir = tempfile.gettempdir()
    log_file = os.path.join(log_dir, "custom.log")

    with patch("oai_platform_core.logging_utils.RotatingFileHandler") as mock_file_handler:
        logger = get_logger(name=logger_name, level=logging.DEBUG, log_file=log_file, max_bytes=2048, backup_count=10)
        
        assert logger.level == logging.DEBUG
        mock_file_handler.assert_called_with(log_file, maxBytes=2048, backupCount=10)

    logger.handlers = []


def test_get_logger_idempotency():
    """Test that get_logger is idempotent and does not add duplicate handlers."""
    logger_name = "idempotent_test"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    with patch("oai_platform_core.logging_utils.RotatingFileHandler"), \
         patch("oai_platform_core.logging_utils.logging.StreamHandler"):
        logger1 = get_logger(name=logger_name)
        assert len(logger1.handlers) == 2

    logger2 = get_logger(name=logger_name)
    assert logger2 is logger1
    assert len(logger2.handlers) == 2
    
    logger.handlers = []


def test_logger_formatting():
    """Test that the correct formatter is applied to handlers."""
    logger_name = "test_formatting_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    mock_file_handler_instance = MagicMock()
    mock_stream_handler_instance = MagicMock()

    with patch("oai_platform_core.logging_utils.RotatingFileHandler", return_value=mock_file_handler_instance), \
         patch("oai_platform_core.logging_utils.logging.StreamHandler", return_value=mock_stream_handler_instance), \
         patch("oai_platform_core.logging_utils.logging.Formatter") as mock_formatter:

        get_logger(name=logger_name)

        mock_formatter.assert_called_with("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        mock_file_handler_instance.setFormatter.assert_called_once()
        mock_stream_handler_instance.setFormatter.assert_called_once()
        
    logger.handlers = []
