# src/api/main.py

import logging
from fastapi import FastAPI
import uvicorn

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
from src.application.shared.infrastructure.logging_config import setup_logging
from src.application.shared.middleware.correlation_middleware import CorrelationMiddleware
from src.application.shared.middleware.logging_middleware import LoggingMiddleware
from src.presentation.routes import test_use_cases


# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Test Case Generator API",
    description="Generate structured test cases from user stories",
    version="1.0.0"
)

# Add middleware (reverse order: last added = first executed)
app.add_middleware(LoggingMiddleware)
app.add_middleware(CorrelationMiddleware)

app.include_router(test_use_cases)


@app.on_event("startup")
async def startup_event():
    """Log application startup."""
    logger.info(
        "Application startup",
        extra={
            "version": ENVIRONMENT_CONFIG.SERVICE_VERSION,
            "environment": ENVIRONMENT_CONFIG.ENVIRONMENT,
            "log_level": ENVIRONMENT_CONFIG.LOG_LEVEL
        }
    )


@app.get("/")
async def root():
    return {
        "service": "Test Case Generator",
        "version": "1.0.0",
        "working": True
    }


@app.get("/health")
async def health_check():
    """Check haealt"""
    return {
        "status": "healthy",
        "llm": "connected",
        "model": ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B
    }


@app.get("/metrics")
async def get_metrics():
    """Get aggregated metrics from MLflow"""
    # This would query MLflow for statistics
    return {
        "message": "View detailed metrics in MLflow UI at http://localhost:5000"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
