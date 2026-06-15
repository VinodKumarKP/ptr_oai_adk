"""Logging configuration and utilities for the OAI Agent Framework.

This module provides:
1. Centralized logging configuration
2. Logging best practices and guidelines
3. Ready-to-use logger setup
4. Structured logging utilities

Quick Start:
    >>> from oai_agent_core.core.logging_config import setup_logging, get_logger
    >>> 
    >>> # At application startup (main.py)
    >>> setup_logging(log_level=logging.INFO, log_file='logs/agent.log')
    >>> 
    >>> # In your modules
    >>> logger = get_logger(__name__)
    >>> logger.info("Agent initialized")
    >>> logger.error("Error occurred", exc_info=True)
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


__all__ = [
    'setup_logging',
    'get_logger',
    'configure_logging',
    'LoggingConfig',
    'StructuredLogger',
]


# ============================================================================
# Logging Levels & Configuration
# ============================================================================

class LoggingConfig:
    """Logging configuration constants and settings."""
    
    # Default logging level
    DEFAULT_LEVEL = logging.INFO
    
    # Log format with timestamp and level
    FORMAT_DETAILED = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Simplified format for console
    FORMAT_SIMPLE = '%(levelname)s - %(name)s - %(message)s'
    
    # Format for file logs (includes more detail)
    FORMAT_FILE = (
        '%(asctime)s | %(levelname)-8s | %(name)-20s | '
        '%(filename)s:%(funcName)s:%(lineno)d | %(message)s'
    )
    
    # Date format
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Log file settings
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5


# ============================================================================
# Logger Factory
# ============================================================================

_configured = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """Configure logging for the entire agent framework.
    
    Call this once at application startup to set up logging.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional path to log file
        console_output: Whether to output to console
        
    Example:
        >>> from oai_agent_core.core.logging_config import configure_logging
        >>> configure_logging(level=logging.DEBUG, log_file='agent.log')
    """
    global _configured
    
    if _configured:
        return
    
    root_logger = logging.getLogger('oai_agent_core')
    root_logger.setLevel(level)
    root_logger.propagate = False
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            LoggingConfig.FORMAT_DETAILED,
            datefmt=LoggingConfig.DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=LoggingConfig.LOG_FILE_MAX_BYTES,
            backupCount=LoggingConfig.LOG_FILE_BACKUP_COUNT
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            LoggingConfig.FORMAT_FILE,
            datefmt=LoggingConfig.DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from oai_agent_core.core.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Component initialized")
    """
    return logging.getLogger(name)


# ============================================================================
# Structured Logging Utilities
# ============================================================================

class StructuredLogger:
    """Wrapper for structured logging (JSON format).
    
    Provides methods for structured logging with consistent formatting.
    Useful for log aggregation and analysis.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> struct_logger = StructuredLogger(logger)
        >>> struct_logger.log_component_init('MyComponent', {'version': '1.0'})
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize structured logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_structured(
        self,
        level: int,
        event: str,
        **kwargs
    ) -> None:
        """Log a structured event.
        
        Args:
            level: Logging level
            event: Event name/description
            **kwargs: Additional fields to include
        """
        log_data = {
            'event': event,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.log(level, json.dumps(log_data))
    
    def log_component_init(self, component_name: str, details: dict = None) -> None:
        """Log component initialization."""
        self.log_structured(
            logging.INFO,
            'component_initialized',
            component=component_name,
            details=details or {}
        )
    
    def log_operation_start(self, operation: str, **context) -> None:
        """Log operation start."""
        self.log_structured(
            logging.DEBUG,
            'operation_started',
            operation=operation,
            context=context
        )
    
    def log_operation_complete(self, operation: str, duration_ms: float = 0, **context) -> None:
        """Log operation completion."""
        self.log_structured(
            logging.DEBUG,
            'operation_completed',
            operation=operation,
            duration_ms=duration_ms,
            context=context
        )
    
    def log_error_detail(self, error_type: str, message: str, **context) -> None:
        """Log error with details."""
        self.log_structured(
            logging.ERROR,
            'error_occurred',
            error_type=error_type,
            message=message,
            context=context
        )


# ============================================================================
# Logging Best Practices
# ============================================================================

"""
LOGGING BEST PRACTICES FOR AGENT-CORE
======================================

1. LOGGER NAMING
   - Use module name: logger = get_logger(__name__)
   - Enables filtering by module in logs

2. LOG LEVELS
   DEBUG   - Detailed diagnostic info (verbose, not for production)
   INFO    - General informational messages
   WARNING - Something unexpected but not critical
   ERROR   - Error that needs attention
   CRITICAL - System is unusable

3. WHAT TO LOG

   DEBUG LEVEL (Development/Troubleshooting):
   - Component initialization with config details
   - Decision points ("Choosing X over Y because...")
   - State changes in important variables
   - Function entry/exit (sparingly)
   - Configuration resolution steps
   
   Example:
   >>> logger.debug(f"Tool registry loading from {tool_config_path}")
   >>> logger.debug(f"Selected KB type '{kb_type}' based on config")

   INFO LEVEL (Normal Operation):
   - Component initialization status
   - Major operations (loaded N tools, initialized KB)
   - Configuration applied
   - Version information
   
   Example:
   >>> logger.info(f"✅ Successfully loaded {len(tools)} tools")
   >>> logger.info(f"Agent '{agent_name}' initialized with KB")

   WARNING LEVEL (Something Off):
   - Recoverable errors (one tool failed, continuing)
   - Deprecated usage
   - Configuration fallbacks applied
   - Expected but unusual conditions
   
   Example:
   >>> logger.warning(f"Failed to load optional skill: {skill_name}")
   >>> logger.warning(f"KB metadata missing, continuing with local discovery")

   ERROR LEVEL (Action Required):
   - Operations that failed
   - Invalid configuration
   - Missing required resources
   - Always use exc_info=True to include traceback
   
   Example:
   >>> logger.error(f"Failed to initialize KB: {e}", exc_info=True)

4. ERROR MESSAGES
   - Be specific about what failed and why
   - Include context (filenames, IDs, configs)
   - Use exc_info=True for exception context
   
   BAD:  logger.error("Error occurred")
   GOOD: logger.error(f"Failed to load tool '{tool_name}': {e}", exc_info=True)

5. AVOID
   - Don't use print() - use logger instead
   - Don't catch then log then raise (let caller log)
   - Don't log sensitive data (API keys, tokens)
   - Don't use f-strings with exc_info=True (use old-style)
   
   BAD:  except Exception as e:
             logger.error(f"Error: {e}")
             raise
   
   GOOD: except Exception as e:
             logger.error("Failed to process: %s", e, exc_info=True)
             raise

6. STRUCTURED LOGGING
   For complex logging needs, use StructuredLogger:
   
   >>> struct_logger = StructuredLogger(logger)
   >>> struct_logger.log_operation_start(
   ...     'load_tools',
   ...     tool_count=5,
   ...     framework='langchain'
   ... )

7. PERFORMANCE
   - Avoid logging in hot paths (loops, frequent calls)
   - Use lazy formatting: logger.debug("Value: %s", expensive_func())
   - Debug level is safe - it's disabled in production

8. INTEGRATION WITH OBSERVABILITY
   Logging integrates with OpenTelemetry:
   - Logs should be at appropriate level
   - ERROR logs become spans with status=ERROR
   - DEBUG logs help with distributed tracing context

Example Implementation:
-----------------------

def load_tool(self, tool_name: str, tool_config: dict) -> None:
    '''Load a tool from configuration.'''
    
    # Entry point
    self.logger.debug(f"Starting tool load: {tool_name}")
    self.logger.debug(f"Tool config: {tool_config}")
    
    try:
        # Decision points
        if 'custom_path' in tool_config:
            path = tool_config['custom_path']
            self.logger.debug(f"Using custom path for {tool_name}: {path}")
        else:
            path = self._resolve_default_path(tool_name)
            self.logger.debug(f"Using default path for {tool_name}: {path}")
        
        # Operations with logging
        module = self._import_module(path)
        self.logger.debug(f"Module imported: {module.__name__}")
        
        tool = module.get_tool()
        self.tools[tool_name] = tool
        
        # Success
        self.logger.info(f"✅ Tool loaded: {tool_name}")
        
    except FileNotFoundError as e:
        self.logger.error(f"Tool file not found for '{tool_name}': {path}")
    except ImportError as e:
        self.logger.error(f"Failed to import module for '{tool_name}': {e}", 
                         exc_info=True)
    except Exception as e:
        self.logger.error(f"Unexpected error loading '{tool_name}': {e}",
                         exc_info=True)
        raise
"""


# ============================================================================
# Application Initialization Helper
# ============================================================================

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    enable_json: bool = False
) -> None:
    """Initialize logging for the application.
    
    Call this ONCE at application startup before creating agents.
    
    Args:
        log_level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (created in logs/ directory if not absolute)
        console_output: Whether to output to console
        enable_json: Whether to enable JSON structured logging (future enhancement)
        
    Example:
        >>> from oai_agent_core.core.logging_config import setup_logging
        >>> 
        >>> if __name__ == '__main__':
        >>>     setup_logging(
        >>>         log_level=logging.INFO,
        >>>         log_file='logs/agent.log',
        >>>         console_output=True
        >>>     )
        >>>     
        >>>     # Now create and use agents
        >>>     agent = BaseAgent(...)
    """
    # Make log file path absolute if not already
    if log_file and not Path(log_file).is_absolute():
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / log_file)
    
    configure_logging(
        level=log_level,
        log_file=log_file,
        console_output=console_output
    )
    
    # Log startup
    logger = get_logger(__name__)
    logger.info(f"Logging initialized (level: {logging.getLevelName(log_level)})")
    if log_file:
        logger.info(f"Log file: {log_file}")


# ============================================================================
# Module-Level Initialization
# ============================================================================

# Configure root logger for agent-core on module import
configure_logging()
