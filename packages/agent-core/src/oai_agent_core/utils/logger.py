"""
Logger - Advanced logging solution with rotating files and structured output

This module provides a comprehensive logging solution designed for production
applications with features like rotating files, structured logging, and
performance monitoring.

Note: For basic logging setup, prefer using logging_config module:
    >>> from oai_agent_core.core.logging_config import setup_logging, get_logger
    >>> setup_logging(log_level=logging.INFO)
    
This module is useful for advanced features like performance tracking and JSON formatting.
"""

import json
import logging
import logging.handlers
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from logging.handlers import RotatingFileHandler

import sys
import time

# Import the standard logging setup from logging_config
try:
    from oai_agent_core.core.logging_config import get_logger as _get_logger_from_config
except ImportError:
    _get_logger_from_config = None


def get_logger() -> logging.Logger:
    """
    Get a configured logger instance with file and console handlers.
    
    This is a simple factory function for quick logger setup. For more advanced
    configuration, use the Logger class or create_agent_logger factory.
    
    Returns:
        logging.Logger: Configured logger instance.
        
    Deprecated:
        Use logging_config.get_logger() or logging_config.setup_logging() instead.
    """
    # Try to use the standard logging_config setup if available
    if _get_logger_from_config:
        return _get_logger_from_config('agent_logger')
    
    # Fallback to original implementation
    logger = logging.getLogger('agent_logger')  # Use named logger instead of root

    logger.handlers.clear()  # Clear existing handlers
    logger.propagate = False  # Prevent propagation to root logger
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    temp_dir = tempfile.gettempdir()
    log_file_path = os.path.join(temp_dir, 'app.log')

    # Create a file handler to write logs to a file
    file_handler = RotatingFileHandler(log_file_path, maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


class LogLevel(Enum):
    """Log level enumeration"""
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


class LogFormat(Enum):
    """Log format enumeration"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    CUSTOM = "custom"


class PerformanceMetrics:
    """Simple performance metrics collector"""

    def __init__(self):
        self._metrics = {}
        self._lock = threading.Lock()

    def record_duration(self, operation: str, duration: float):
        """Record operation duration"""
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        with self._lock:
            durations = self._metrics.get(operation, [])
            if not durations:
                return {}

            return {
                'count': len(durations),
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'total': sum(durations)
            }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all operation statistics"""
        with self._lock:
            return {op: self.get_stats(op) for op in self._metrics.keys()}

    def clear(self):
        """Clear all metrics"""
        with self._lock:
            self._metrics.clear()


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields if enabled
        if self.include_extra_fields and hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data

        return json.dumps(log_data, default=str, separators=(',', ':'))


class Logger:
    """
    Advanced logger with rotating files, structured output, and performance monitoring

    Features:
    - Multiple output destinations (console, file, rotating file)
    - Structured logging with JSON support
    - Performance monitoring and metrics
    - Context managers for operation tracking
    - Thread-safe operations
    - Configurable log levels and formats
    - Log file compression and cleanup
    """

    def __init__(self,
                 name: str = "root",
                 log_level: Union[LogLevel, str, int] = LogLevel.INFO,
                 log_dir: str = "logs",
                 log_file: str = "logging.log",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 log_format: LogFormat = LogFormat.DETAILED,
                 custom_format: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = False,
                 enable_performance_tracking: bool = True,
                 compress_backups: bool = True):
        """
        Initialize AgentLogger

        Args:
            name: Logger name
            log_level: Minimum log level to record
            log_dir: Directory to store log files
            log_file: Main log file name
            max_file_size: Maximum size for each log file in bytes
            backup_count: Number of backup files to keep
            log_format: Log format style
            custom_format: Custom format string (used when log_format is CUSTOM)
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON formatted file output
            enable_performance_tracking: Enable performance metrics
            compress_backups: Compress old log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.compress_backups = compress_backups

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.handlers.clear()  # Clear any existing handlers
        self._set_log_level(log_level)

        # Performance tracking
        self.performance_metrics = PerformanceMetrics() if enable_performance_tracking else None

        # Set up formatters
        self._setup_formatters(log_format, custom_format)

        # Set up handlers
        self._setup_handlers()

        # Thread safety
        self._lock = threading.Lock()

        # Log the logger initialization
        self.info(f"AgentLogger '{name}' initialized", extra_data={
            'log_dir': str(self.log_dir),
            'log_file': self.log_file,
            'log_level': self.logger.level,
            'handlers': [type(h).__name__ for h in self.logger.handlers]
        })

    def _set_log_level(self, log_level: Union[LogLevel, str, int]):
        """Set logger level"""
        if isinstance(log_level, LogLevel):
            level = log_level.value
        elif isinstance(log_level, str):
            level = getattr(logging, log_level.upper())
        else:
            level = log_level

        self.logger.setLevel(level)

    def _setup_formatters(self, log_format: LogFormat, custom_format: Optional[str]):
        """Set up log formatters"""
        if log_format == LogFormat.SIMPLE:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        elif log_format == LogFormat.DETAILED:
            format_str = ('%(asctime)s - %(name)s - %(levelname)s - '
                          '%(module)s:%(funcName)s:%(lineno)d - %(message)s')
        elif log_format == LogFormat.CUSTOM and custom_format:
            format_str = custom_format
        else:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        self.formatter = logging.Formatter(
            format_str,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.json_formatter = JsonFormatter()

    def _setup_handlers(self):
        """Set up log handlers"""
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if self.enable_file:
            file_path = self.log_dir / self.log_file
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # JSON file handler
        if self.enable_json:
            json_file_path = self.log_dir / f"{self.log_file.rsplit('.', 1)[0]}.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            json_handler.setFormatter(self.json_formatter)
            self.logger.addHandler(json_handler)

    def _log_with_extra(self, level: int, message: str,
                        extra_data: Optional[Dict[str, Any]] = None,
                        exc_info: bool = False):
        """Internal method to log with extra data"""
        if extra_data:
            # Create a custom LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (),
                exc_info=exc_info if exc_info else None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            self.logger.log(level, message, exc_info=exc_info)

    # Logging methods
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log_with_extra(logging.DEBUG, message, extra_data)

    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log_with_extra(logging.INFO, message, extra_data)

    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log_with_extra(logging.WARNING, message, extra_data)

    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
              exc_info: bool = False):
        """Log error message"""
        self._log_with_extra(logging.ERROR, message, extra_data, exc_info)

    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
                 exc_info: bool = False):
        """Log critical message"""
        self._log_with_extra(logging.CRITICAL, message, extra_data, exc_info)

    def exception(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log exception with traceback"""
        self._log_with_extra(logging.ERROR, message, extra_data, exc_info=True)

    # Performance tracking methods
    @contextmanager
    def track_performance(self, operation_name: str, log_result: bool = True):
        """Context manager for tracking operation performance"""
        if not self.performance_metrics:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.performance_metrics.record_duration(operation_name, duration)

            if log_result:
                self.debug(f"Operation '{operation_name}' completed", extra_data={
                    'operation': operation_name,
                    'duration_seconds': duration,
                    'duration_ms': duration * 1000
                })

    def log_performance_stats(self, operation: Optional[str] = None):
        """Log performance statistics"""
        if not self.performance_metrics:
            self.warning("Performance tracking not enabled")
            return

        if operation:
            stats = self.performance_metrics.get_stats(operation)
            if stats:
                self.info(f"Performance stats for '{operation}'", extra_data={
                    'operation': operation,
                    'statistics': stats
                })
            else:
                self.info(f"No performance data for operation '{operation}'")
        else:
            all_stats = self.performance_metrics.get_all_stats()
            if all_stats:
                self.info("All performance statistics", extra_data={
                    'all_statistics': all_stats
                })
            else:
                self.info("No performance data available")

    # Configuration methods
    def set_level(self, level: Union[LogLevel, str, int]):
        """Change log level"""
        self._set_log_level(level)
        self.info(f"Log level changed to {self.logger.level}")

    def add_custom_handler(self, handler: logging.Handler):
        """Add a custom handler"""
        self.logger.addHandler(handler)
        self.info(f"Added custom handler: {type(handler).__name__}")

    def remove_handler(self, handler: logging.Handler):
        """Remove a handler"""
        if handler in self.logger.handlers:
            self.logger.removeHandler(handler)
            self.info(f"Removed handler: {type(handler).__name__}")

    # Utility methods
    def get_log_files(self) -> list:
        """Get list of all log files"""
        log_files = []
        for file_path in self.log_dir.glob("*.log*"):
            log_files.append(str(file_path))
        return sorted(log_files)

    def clear_old_logs(self, days_old: int = 30):
        """Clear log files older than specified days"""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_files = []

        for file_path in self.log_dir.glob("*.log*"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                except OSError as e:
                    self.warning(f"Could not remove old log file {file_path}: {e}")

        if removed_files:
            self.info(f"Cleared {len(removed_files)} old log files", extra_data={
                'removed_files': removed_files,
                'days_old_threshold': days_old
            })

    def get_logger_info(self) -> Dict[str, Any]:
        """Get logger configuration information"""
        return {
            'name': self.name,
            'level': self.logger.level,
            'level_name': logging.getLevelName(self.logger.level),
            'handlers': [type(h).__name__ for h in self.logger.handlers],
            'log_dir': str(self.log_dir),
            'log_file': self.log_file,
            'max_file_size': self.max_file_size,
            'backup_count': self.backup_count,
            'performance_tracking_enabled': self.performance_metrics is not None
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            self.exception(f"Exception in logger context: {exc_val}")

        # Flush all handlers
        for handler in self.logger.handlers:
            handler.flush()


# Factory function for easy logger creation
def create_agent_logger(name: str = "AgentClient",
                        log_dir: str = "logs",
                        log_level: str = "INFO",
                        enable_json: bool = False,
                        enable_performance: bool = True) -> Logger:
    """
    Factory function to create a pre-configured AgentLogger.
    
    For application-wide logging setup, use logging_config.setup_logging() instead:
        >>> from oai_agent_core.core.logging_config import setup_logging
        >>> setup_logging(log_level=logging.INFO, log_file='logs/agent.log')

    Args:
        name: Logger name
        log_dir: Log directory
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Enable JSON logging
        enable_performance: Enable performance tracking

    Returns:
        Configured AgentLogger instance
        
    Example:
        >>> # For advanced features (performance tracking, JSON logging)
        >>> logger = create_agent_logger('MyAgent', enable_performance=True)
        >>> 
        >>> with logger.track_performance('my_operation'):
        >>>     # do work
        >>>     pass
        >>> logger.log_performance_stats('my_operation')
    """
    return Logger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        log_file=f"{name.lower()}.log",
        enable_json=enable_json,
        enable_performance_tracking=enable_performance,
        log_format=LogFormat.DETAILED
    )
