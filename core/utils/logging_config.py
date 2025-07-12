#!/usr/bin/env python3
"""
Logging Configuration for ANNrun_code
Centralized logging setup with multiple handlers and formats
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File formatter (detailed)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console formatter (clean)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    handlers = []
    
    # File handler with rotation
    if file_output:
        log_file = log_path / f"annrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        logger.addHandler(console_handler)
    
    # Error file handler (separate file for errors)
    if file_output:
        error_log_file = log_path / f"annrun_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)
        logger.addHandler(error_handler)
    
    # Log the setup
    logger.info("=" * 60)
    logger.info("ANNrun_code Logging System Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log Directory: {log_path.absolute()}")
    logger.info(f"Console Output: {console_output}")
    logger.info(f"File Output: {file_output}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_system_info():
    """Log system information"""
    logger = get_logger(__name__)
    
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"  Current Working Directory: {os.getcwd()}")


def log_experiment_start(experiment_id: int, config: Dict[str, Any]):
    """Log experiment start information"""
    logger = get_logger(__name__)
    
    logger.info("🚀 EXPERIMENT START")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Started at: {datetime.now().isoformat()}")


def log_experiment_end(experiment_id: int, duration: float, success: bool):
    """Log experiment end information"""
    logger = get_logger(__name__)
    
    status = "✅ COMPLETED" if success else "❌ FAILED"
    logger.info(f"🏁 EXPERIMENT END - {status}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Ended at: {datetime.now().isoformat()}")


def setup_module_logger(module_name: str, log_level: str = "INFO") -> logging.Logger:
    """Setup logger for a specific module"""
    logger = logging.getLogger(module_name)
    
    # If root logger is already configured, use it
    if logging.getLogger().handlers:
        return logger
    
    # Otherwise setup basic logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logger.setLevel(numeric_level)
    
    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Performance logging helpers
class PerformanceLogger:
    """Helper class for logging performance metrics"""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = get_logger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"⏱️  Started: {operation}")
    
    def end_timer(self, operation: str):
        """End timing an operation and log duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"⏱️  Completed: {operation} in {duration:.3f}s")
            del self.start_times[operation]
            return duration
        else:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return None
    
    def log_memory_usage(self, operation: str = ""):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"💾 Memory usage{' for ' + operation if operation else ''}: {memory_mb:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory logging")


# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(
        log_dir="./test_logs",
        log_level="DEBUG",
        console_output=True,
        file_output=True
    )
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.start_timer("test_operation")
    
    import time
    time.sleep(1)  # Simulate work
    
    perf_logger.end_timer("test_operation")
    perf_logger.log_memory_usage("test")
    
    # Log system info
    log_system_info()
    
    print("Logging test completed!")
