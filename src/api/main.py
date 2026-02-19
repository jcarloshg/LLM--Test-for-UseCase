# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from src.llm.client import LLMClient, LLMConfig
from src.llm.prompts import PromptBuilder
from src.validators.structure import StructureValidator
from src.validators.quality import QualityValidator
from src.mlflow_tracker import MLflowTracker
import uvicorn

app = FastAPI(
    title="Test Case Generator API",
    description="Generate structured test cases from user stories",
    version="1.0.0"
)

# Initialize components
llm_config = LLMConfig()
llm_client = LLMClient(llm_config)
prompt_builder = PromptBuilder()
structure_validator = StructureValidator()
quality_validator = QualityValidator(llm_client)
mlflow_tracker = MLflowTracker()

# Request/Response models


class GenerateRequest(BaseModel):
    user_story: str = Field(..., min_length=20, max_length=500)
    include_quality_check: bool = True
    model: Optional[str] = None


class TestCaseResponse(BaseModel):
    id: str
    title: str
    priority: str
    given: str
    when: str
    then: str


class GenerateResponse(BaseModel):
    user_story: str
    test_cases: List[TestCaseResponse]
    validation: dict
    quality_metrics: Optional[dict] = None
    metadata: dict


@app.get("/")
async def root():
    return {
        "service": "Test Case Generator",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate-test-cases",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test LLM connection
        result = llm_client.generate("test", "")
        llm_healthy = "error" not in result

        return {
            "status": "healthy" if llm_healthy else "degraded",
            "llm": "connected" if llm_healthy else "error",
            "model": llm_config.model
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/generate-test-cases", response_model=GenerateResponse)
async def generate_test_cases(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Generate test cases from user story"""

    # Build prompt
    prompts = prompt_builder.build(request.user_story)

    # Generate with LLM
    llm_result = llm_client.generate(
        prompts['user'],
        prompts['system']
    )

    if "error" in llm_result:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {llm_result['error']}"
        )

    # Parse JSON output
    try:
        output_text = llm_result['text'].strip()

        # Extract JSON if wrapped
        if '```json' in output_text:
            output_text = output_text.split('```json')[1].split('```')[0]
        elif '```' in output_text:
            output_text = output_text.split('```')[1].split('```')[0]

        output_json = json.loads(output_text)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM output as JSON: {str(e)}"
        )

    # Validate structure
    structure_validation = structure_validator.validate(output_json)

    if not structure_validation['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid test case structure: {structure_validation['errors']}"
        )

    test_cases = structure_validation['test_cases']

    # Quality validation (optional, can be slow)
    quality_metrics = None
    if request.include_quality_check:
        quality_metrics = quality_validator.evaluate_relevance(
            request.user_story,
            test_cases
        )

        coverage_metrics = quality_validator.evaluate_coverage(test_cases)
    else:
        coverage_metrics = {"passed": True}

    # Log to MLflow in background
    background_tasks.add_task(
        mlflow_tracker.log_generation,
        user_story=request.user_story,
        test_cases=test_cases,
        structure_validation=structure_validation,
        quality_metrics=quality_metrics or {},
        coverage_metrics=coverage_metrics,
        latency=llm_result['latency'],
        model_info={
            "model": llm_result['model'],
            "provider": llm_result['provider']
        }
    )

    return GenerateResponse(
        user_story=request.user_story,
        test_cases=[TestCaseResponse(**tc) for tc in test_cases],
        validation={
            "structure_valid": structure_validation['valid'],
            "count": structure_validation['count'],
            "quality_passed": quality_metrics['passed'] if quality_metrics else None,
            "coverage_passed": coverage_metrics['passed']
        },
        quality_metrics=quality_metrics,
        metadata={
            "latency": llm_result['latency'],
            "tokens": llm_result['tokens'],
            "model": llm_result['model']
        }
    )


@app.get("/metrics")
async def get_metrics():
    """Get aggregated metrics from MLflow"""
    # This would query MLflow for statistics
    return {
        "message": "View detailed metrics in MLflow UI at http://localhost:5000"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
