"""
Latency tracking and performance analytics module.
Provides utilities to calculate comprehensive latency statistics and validate against SLAs.
"""
from typing import List

import numpy as np

from src.application.evaluate_models.model.latency_tracker_response import LatencyTrackerResponse


class LatencyTracker:
    """
    Best Practice: Track P50, P95, P99 latencies, not just averages
    """

    @staticmethod
    def calculate_latency_stats(latencies: List[float], sla_p95_ms: float = None) -> LatencyTrackerResponse:
        """
        Calculate comprehensive latency statistics and return as LatencyTrackerResponse.

        Args:
            latencies: List of latency values in seconds
            sla_p95_ms: Optional SLA threshold in milliseconds for P95 latency

        Returns:
            LatencyTrackerResponse: Structured response containing all latency metrics
        """
        latencies_array = np.array(latencies)

        stats = {
            "mean": float(np.mean(latencies_array)),
            "median": float(np.median(latencies_array)),
            "p50": float(np.percentile(latencies_array, 50)),
            "p95": float(np.percentile(latencies_array, 95)),
            "p99": float(np.percentile(latencies_array, 99)),
            "min": float(np.min(latencies_array)),
            "max": float(np.max(latencies_array)),
            "std_dev": float(np.std(latencies_array))
        }

        # Check if SLA is met
        sla_met = False
        if sla_p95_ms is not None:
            sla_met = LatencyTracker.meets_sla(latencies, sla_p95_ms)

        return LatencyTrackerResponse(**stats, sla_met=sla_met)

    @staticmethod
    def meets_sla(latencies: List[float], sla_p95_ms: float) -> bool:
        """
        Best Practice: Define SLAs based on percentiles

        Example: "95% of requests must complete within 3 seconds"
        """
        p95 = np.percentile(latencies, 95)
        return p95 * 1000 <= sla_p95_ms  # Convert to ms
