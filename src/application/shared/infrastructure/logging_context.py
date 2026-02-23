"""Context management for correlation IDs in logging."""

import contextvars
from typing import Optional


# Context variable for storing correlation ID
_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current context.

    Args:
        correlation_id: Unique identifier for request tracing
    """
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get the correlation ID from the current context.

    Returns:
        Correlation ID or None if not set
    """
    return _correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id_var.set(None)
