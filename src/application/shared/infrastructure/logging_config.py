"""Centralized logging configuration module."""

import logging
import logging.handlers
import os
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = None,
    log_format: str = None,
    log_file_path: str = None,
    rotation_size: int = None,
    backup_count: int = None,
) -> None:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to LOG_LEVEL env var.
        log_format: Log format (json or console). Defaults to LOG_FORMAT env var.
        log_file_path: Path to log file. Defaults to LOG_FILE_PATH env var.
        rotation_size: Log file rotation size in bytes. Defaults to LOG_ROTATION_SIZE env var.
        backup_count: Number of backup log files. Defaults to LOG_BACKUP_COUNT env var.
    """
    # Get values from env vars with defaults
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_format = log_format or os.getenv("LOG_FORMAT", "json")
    log_file_path = log_file_path or os.getenv("LOG_FILE_PATH", "logs/app.json.log")
    rotation_size = rotation_size or int(os.getenv("LOG_ROTATION_SIZE", 10485760))  # 10MB default
    backup_count = backup_count or int(os.getenv("LOG_BACKUP_COUNT", 5))

    # Set log level
    log_level_value = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_value)

    if log_format.lower() == "json":
        # JSON formatter for console
        console_formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(logger)s %(message)s",
            timestamp=True,
        )
    else:
        # Human-readable formatter for console
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (JSON format)
    if log_file_path:
        # Create log directory if it doesn't exist
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=rotation_size,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level_value)

        # Always use JSON format for file
        # Use empty format to include all fields automatically (standard + extras)
        file_formatter = jsonlogger.JsonFormatter(
            fmt="%(timestamp)s %(levelname)s %(name)s %(message)s",
            timestamp=True,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
