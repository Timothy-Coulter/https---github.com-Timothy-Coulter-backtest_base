"""
Logging Configuration System.

This module provides comprehensive logging functionality for the backtester
with support for file rotation, different log levels, and structured logging.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def __lt__(self, other):
        """Enable comparison between log levels."""
        if self.__class__ is other.__class__:
            order = [self.DEBUG, self.INFO, self.WARNING, self.ERROR, self.CRITICAL]
            return order.index(self) < order.index(other)
        return NotImplemented


class LogFormat(Enum):
    """Log format enumeration."""
    SIMPLE = "SIMPLE"
    STANDARD = "STANDARD"
    DETAILED = "DETAILED"
    JSON = "JSON"


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


# Utility functions for easy access
def get_backtester_logger(name: str = "backtester", **kwargs) -> logging.Logger:
    """Get a backtester logger with default settings.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for logger configuration
        
    Returns:
        Configured logger
    """
    return BacktesterLogger.get_logger(name, **kwargs)

class BacktesterLogger:
    """Backtester-specific logger class for compatibility with tests."""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    def __init__(self, name: str = "backtester", level: LogLevel = LogLevel.INFO,
                 format: LogFormat = LogFormat.STANDARD, log_file: Optional[str] = None) -> None:
        """Initialize the backtester logger.
        
        Args:
            name: Logger name
            level: Log level
            format: Log format
            log_file: Optional log file path
        """
        self.name = name
        self.level = level
        self.format = format
        self.log_file = log_file
        self._logger: Optional[logging.Logger] = None
    
    def set_level(self, level: LogLevel) -> None:
        """Set the log level."""
        if not isinstance(level, LogLevel):
            raise ValueError("Invalid log level")
        self.level = level
        if self._logger:
            self._logger.setLevel(getattr(logging, level.value))
    
    def set_format(self, format: LogFormat) -> None:
        """Set the log format."""
        self.format = format
    
    @classmethod
    def get_logger(cls, name: str, level: str = "INFO",
                   file_path: Optional[str] = None,
                   max_file_size: int = 10485760,  # 10MB
                   backup_count: int = 5,
                   console: bool = True,
                   structured: bool = False) -> logging.Logger:
        """Get or create a logger with the specified configuration."""
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        if structured:
            formatter: logging.Formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        # Add console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if specified
        if file_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Store logger
        cls._loggers[name] = logger
        return logger
    
    def _get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        if self._logger is None:
            self._logger = self._create_logger_instance()
        # Type cast to satisfy MyPy - logger is guaranteed to be non-None
        return self._logger
    
    def _create_logger_instance(self) -> logging.Logger:
        """Create a new logger instance (for test compatibility)."""
        return BacktesterLogger.get_logger(self.name)
    
    def info(self, message: str, extra_data: Optional[Dict] = None) -> None:
        """Log info message."""
        logger = self._get_logger()
        if extra_data:
            logger.info(f"{message} | Extra: {extra_data}")
        else:
            logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._get_logger().warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error message."""
        logger = self._get_logger()
        if exception:
            logger.error(f"{message} | Exception: {exception}")
        else:
            logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._get_logger().debug(message)
    
    def log_performance(self, performance_data: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.info(f"Performance: {performance_data}")
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade execution."""
        self.info(f"Trade: {trade_data}")
    
    def log_config(self, config_data: Dict[str, Any]) -> None:
        """Log configuration data."""
        self.info(f"Config: {config_data}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.info(f"Performance Metrics: {metrics}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass