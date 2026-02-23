"""FastAPI middleware for request/response logging with correlation IDs."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.application.shared.infrastructure.logging_context import set_correlation_id, get_correlation_id


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses with correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request/response with logging.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Generate or get correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        set_correlation_id(correlation_id)

        # Log request
        request_start_time = time.time()
        request_body_size = 0

        if request.method != "GET":
            try:
                body = await request.body()
                request_body_size = len(body)
            except Exception:
                pass

        logger.info(
            "Request received",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query_string": request.url.query,
                "endpoint": f"{request.method} {request.url.path}",
                "request_body_size": request_body_size,
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate latency
            latency = time.time() - request_start_time

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency": latency,
                    "endpoint": f"{request.method} {request.url.path}",
                },
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate latency
            latency = time.time() - request_start_time

            # Log error
            logger.error(
                "Request failed with exception",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency": latency,
                    "endpoint": f"{request.method} {request.url.path}",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True,
            )
            raise
