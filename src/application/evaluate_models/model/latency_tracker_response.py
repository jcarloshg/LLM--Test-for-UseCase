# latency_tracker_response.py
from pydantic import BaseModel


class LatencyTrackerResponse(BaseModel):
    """Latency statistics response model"""
    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    std_dev: float
    sla_met: bool = False  # Optional field indicating if SLA is met
