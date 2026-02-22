"""
Quality tracker response model for test case evaluation metrics.
Encapsulates all quality-related measurements from test case generation.
"""
from pydantic import BaseModel, Field


class QualityTrackerResponse(BaseModel):
    """
    Represents the complete quality assessment results from test case generation.

    This model encapsulates:
    - Structural quality: Test case structure completeness
    - Precondition quality: Setup requirement quality
    - JSON parsing success: Structural compliance on first attempt
    - Average quality scores: LLM-generated quality metrics
    - Retry metrics: Infrastructure efficiency indicators
    """

    quality_score: float = Field(
        ...,
        description="Average weighted quality score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    precondition_score: float = Field(
        ...,
        description="Average precondition quality score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    structure_score: float = Field(
        ...,
        description="Average structure quality score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    total_tests: int = Field(
        ...,
        description="Total number of test cases evaluated",
        ge=0
    )
    passing_tests: int = Field(
        ...,
        description="Number of tests meeting quality threshold",
        ge=0
    )
    passing_rate: float = Field(
        ...,
        description="Percentage of tests meeting quality threshold (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    json_parsing_success_rate: float = Field(
        ...,
        description="Percentage of responses parsed on first attempt (target >95%, 0.0 to 100.0)",
        ge=0.0,
        le=100.0
    )
    avg_quality_score: float = Field(
        ...,
        description="Average quality_score from generated test cases (0.0 to 10.0)",
        ge=0.0,
        le=10.0
    )
    retry_rate: float = Field(
        ...,
        description="Percentage of responses requiring retries (0.0 to 100.0)",
        ge=0.0,
        le=100.0
    )
    total_responses: int = Field(
        ...,
        description="Total number of responses evaluated",
        ge=0
    )

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True