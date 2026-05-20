import logging
import os
import tempfile
from unittest.mock import MagicMock, patch
from oai_mcp_server_core.utils.logger_utils import get_logger

def test_get_logger_initialization():
    # Reset logger handlers for testing
    logger = logging.getLogger('app')
    logger.handlers = []
    
    # Patch the class where it is imported in the module under test
    with patch("oai_platform_core.logging_utils.RotatingFileHandler") as mock_file_handler:
        with patch("oai_platform_core.logging_utils.logging.StreamHandler") as mock_stream_handler:
            logger = get_logger()
            
            assert len(logger.handlers) == 2
            mock_file_handler.assert_called_once()
            mock_stream_handler.assert_called_once()
            
            # Verify file path
            temp_dir = tempfile.gettempdir()
            expected_path = os.path.join(temp_dir, 'app.log')
            mock_file_handler.assert_called_with(expected_path, maxBytes=1024 * 1024, backupCount=5)

def test_get_logger_idempotency():
    # Reset logger handlers
    logger = logging.getLogger('app')
    logger.handlers = []
    
    with patch("oai_platform_core.logging_utils.RotatingFileHandler"), patch("oai_platform_core.logging_utils.logging.StreamHandler"):
        # First call adds handlers
        logger1 = get_logger()
        assert len(logger1.handlers) == 2
        
        # Second call should return same logger without adding more handlers
        logger2 = get_logger()
        assert logger2 is logger1
        assert len(logger2.handlers) == 2

def test_logger_formatting():
    # Reset logger handlers
    logger = logging.getLogger('app')
    logger.handlers = []
    
    with patch("oai_platform_core.logging_utils.RotatingFileHandler"), patch("oai_platform_core.logging_utils.logging.StreamHandler"), patch("oai_platform_core.logging_utils.logging.Formatter") as mock_formatter:
        get_logger()
        mock_formatter.assert_called_with('%(asctime)s - %(name)s - %(levelname)s - %(message)s')