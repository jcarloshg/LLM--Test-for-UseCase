"""Correlation ID middleware for request tracing."""

import contextvars
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Store correlation ID in context var for use throughout request lifecycle
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get the current request's correlation ID."""
    return correlation_id_var.get() or "unknown"


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and propagate correlation IDs for request tracing."""

    async def dispatch(self, request: Request, call_next):
        """Add correlation ID to request and response."""
        # Generate new correlation ID or use existing one from header
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4())
        )

        # Store in context var for access throughout request
        token = correlation_id_var.set(correlation_id)

        try:
            response = await call_next(request)
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            # Reset context var
            correlation_id_var.reset(token)
