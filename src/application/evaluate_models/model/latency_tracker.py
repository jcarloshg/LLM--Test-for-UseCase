# performance_tracker.py
from typing import List, Dict

import numpy as np


class LatencyTracker:
    """
    Best Practice: Track P50, P95, P99 latencies, not just averages
    """

    @staticmethod
    def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive latency statistics
        """

        latencies_array = np.array(latencies)

        return {
            "mean": float(np.mean(latencies_array)),
            "median": float(np.median(latencies_array)),
            "p50": float(np.percentile(latencies_array, 50)),
            "p95": float(np.percentile(latencies_array, 95)),
            "p99": float(np.percentile(latencies_array, 99)),
            "min": float(np.min(latencies_array)),
            "max": float(np.max(latencies_array)),
            "std_dev": float(np.std(latencies_array))
        }

    @staticmethod
    def meets_sla(latencies: List[float], sla_p95_ms: float) -> bool:
        """
        Best Practice: Define SLAs based on percentiles

        Example: "95% of requests must complete within 3 seconds"
        """
        p95 = np.percentile(latencies, 95)
        return p95 * 1000 <= sla_p95_ms  # Convert to ms
