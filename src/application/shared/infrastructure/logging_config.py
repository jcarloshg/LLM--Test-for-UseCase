"""Logging configuration for structured JSON logging using python-json-logger."""

import logging
import logging.handlers
import sys
from pathlib import Path

from pythonjsonlogger import jsonlogger

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z")
        log_record["service"] = "test-case-api"
        log_record["environment"] = ENVIRONMENT_CONFIG.ENVIRONMENT
        log_record["version"] = ENVIRONMENT_CONFIG.SERVICE_VERSION


def setup_logging() -> None:
    """
    Configure logging with JSON formatting for both stdout and file handlers.

    - stdout: JSON format for Docker logging driver
    - file: Rotating JSON log file with size-based rotation
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, ENVIRONMENT_CONFIG.LOG_LEVEL))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # JSON formatter
    json_formatter = CustomJsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s',
        rename_fields={"level": "severity"},
        static_fields={
            "service": "test-case-api",
            "environment": ENVIRONMENT_CONFIG.ENVIRONMENT,
            "version": ENVIRONMENT_CONFIG.SERVICE_VERSION,
        }
    )

    # Stdout handler (for Docker logging driver)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(json_formatter)
    logger.addHandler(stdout_handler)

    # File handler with rotation (if LOG_FILE_PATH specified)
    if ENVIRONMENT_CONFIG.LOG_FILE_PATH:
        try:
            log_file_path = Path(ENVIRONMENT_CONFIG.LOG_FILE_PATH)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=ENVIRONMENT_CONFIG.LOG_ROTATION_SIZE,
                backupCount=ENVIRONMENT_CONFIG.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(json_formatter)
            logger.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            # Log to stdout if file handler fails
            logger.debug(
                f"Could not create file handler for {ENVIRONMENT_CONFIG.LOG_FILE_PATH}: {e}. "
                "Using stdout only."
            )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
