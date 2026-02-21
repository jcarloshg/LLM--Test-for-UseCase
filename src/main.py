# src/api/main.py

from fastapi import FastAPI
import uvicorn

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
from src.presentation.routes import test_use_cases


app = FastAPI(
    title="Test Case Generator API",
    description="Generate structured test cases from user stories",
    version="1.0.0"
)

app.include_router(test_use_cases)


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
