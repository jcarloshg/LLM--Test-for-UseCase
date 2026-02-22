"""
Cost tracker response model for infrastructure efficiency metrics.
Encapsulates cost analysis and resource utilization measurements.
"""
from typing import List
from pydantic import BaseModel, Field


class ResourceUtilization(BaseModel):
    """
    Represents resource consumption metrics for Ollama container operations.

    Tracks actual and estimated resource usage during test case generation.
    """

    avg_latency: float = Field(
        ...,
        description="Average response latency in seconds",
        ge=0.0
    )
    throughput_requests_per_min: float = Field(
        ...,
        description="Estimated requests per minute capacity",
        ge=0.0
    )
    estimated_memory_per_request_mb: float = Field(
        ...,
        description="Estimated memory usage per request in MB",
        ge=0.0
    )
    estimated_cpu_usage_percent: float = Field(
        ...,
        description="Estimated CPU usage percentage (0.0 to 100.0)",
        ge=0.0,
        le=100.0
    )
    total_execution_time_seconds: float = Field(
        ...,
        description="Total cumulative execution time in seconds",
        ge=0.0
    )
    concurrent_capacity: int = Field(
        ...,
        description="Estimated number of concurrent requests the system can handle",
        ge=1
    )

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True


class CostTrackerResponse(BaseModel):
    """
    Represents the complete cost analysis and efficiency results.

    This model encapsulates:
    - Infrastructure cost per 1,000 requests
    - Server and capacity configuration
    - Resource utilization metrics
    - Cost efficiency scoring
    - Actionable optimization recommendations
    """

    cost_per_thousand_requests: float = Field(
        ...,
        description="Infrastructure cost per 1,000 requests in USD",
        ge=0.0
    )
    monthly_server_cost: float = Field(
        ...,
        description="Monthly infrastructure cost in USD",
        ge=0.0
    )
    max_requests_per_day: int = Field(
        ...,
        description="Maximum requests the system can handle per day without latency degradation",
        ge=1
    )
    resource_utilization: ResourceUtilization = Field(
        ...,
        description="Detailed resource utilization metrics"
    )
    cost_efficiency_score: float = Field(
        ...,
        description="Overall cost efficiency score (0.0-1.0, higher is better)",
        ge=0.0,
        le=1.0
    )
    recommendations: List[str] = Field(
        ...,
        description="List of optimization recommendations based on current metrics"
    )

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
