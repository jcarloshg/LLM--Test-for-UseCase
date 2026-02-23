"""HTTP request logging middleware."""

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.application.shared.middleware.correlation_middleware import get_correlation_id

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        correlation_id = get_correlation_id()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # Log request
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "request": {
                    "method": method,
                    "path": path,
                    "client_ip": client_ip
                }
            }
        )

        # Time the request
        start_time = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "response": {
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2)
                    },
                    "request": {
                        "method": method,
                        "path": path
                    }
                }
            )

            # Log slow requests as warnings
            if duration_ms > 5000:  # 5 seconds
                logger.warning(
                    "Slow request detected",
                    extra={
                        "correlation_id": correlation_id,
                        "request": {
                            "method": method,
                            "path": path
                        },
                        "response": {
                            "status_code": response.status_code,
                            "duration_ms": round(duration_ms, 2)
                        }
                    }
                )

            return response
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "request": {
                        "method": method,
                        "path": path
                    },
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "duration_ms": round(duration_ms, 2)
                    }
                }
            )
            raise
