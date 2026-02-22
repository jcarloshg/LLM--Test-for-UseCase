"""
Latency tracking and performance analytics module.
Provides utilities to calculate comprehensive latency statistics and validate against SLAs.
"""
from typing import List, Dict, Any

import numpy as np

from src.application.evaluate_models.model.latency_tracker_response import LatencyTrackerResponse
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


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

    @staticmethod
    def calculate_from_execution_responses(
        execution_responses: List[ExecutableChainResponse],
        sla_p95_ms: float = None
    ) -> Dict[str, Any]:
        """
        Calculate latency statistics from ExecutableChainResponse objects.

        Args:
            execution_responses: List of ExecutableChainResponse objects from chain executions
            sla_p95_ms: Optional SLA threshold in milliseconds for P95 latency

        Returns:
            Dict containing:
                - latency_stats: LatencyTrackerResponse with all metrics
                - total_executions: Number of executions analyzed
                - sla_met: Boolean indicating if SLA is met
        """
        if not execution_responses:
            return {
                "latency_stats": LatencyTrackerResponse(
                    mean=0.0,
                    median=0.0,
                    p50=0.0,
                    p95=0.0,
                    p99=0.0,
                    min=0.0,
                    max=0.0,
                    std_dev=0.0,
                    sla_met=False
                ),
                "total_executions": 0,
                "sla_met": False
            }

        # Extract latencies from execution responses
        latencies = []
        for response in execution_responses:
            if hasattr(response, "latency") and response.latency is not None:
                latencies.append(response.latency)

        if not latencies:
            return {
                "latency_stats": LatencyTrackerResponse(
                    mean=0.0,
                    median=0.0,
                    p50=0.0,
                    p95=0.0,
                    p99=0.0,
                    min=0.0,
                    max=0.0,
                    std_dev=0.0,
                    sla_met=False
                ),
                "total_executions": len(execution_responses),
                "sla_met": False
            }

        # Calculate statistics
        latency_stats = LatencyTracker.calculate_latency_stats(latencies, sla_p95_ms)

        return {
            "latency_stats": latency_stats,
            "total_executions": len(execution_responses),
            "sla_met": latency_stats.sla_met
        }
