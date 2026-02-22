"""
Cost tracking and infrastructure efficiency module for test case generation.
Measures operational costs and resource utilization for scalability analysis.
"""
from typing import List, Dict, Any
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


class CostTracker:
    """
    Tracks and evaluates infrastructure costs and resource efficiency.

    Cost considerations for Ollama-based inference:
    - Ollama API inference cost: $0 (self-hosted)
    - Cost shift to infrastructure: Compute resources, memory, storage
    - Key metrics: Cost per 1,000 requests, resource utilization
    """

    @staticmethod
    def calculate_cost_per_thousand_requests(
        monthly_server_cost: float,
        max_requests_per_day: int
    ) -> float:
        """
        Calculate infrastructure cost per 1,000 requests.

        Args:
            monthly_server_cost: Monthly cost of server/cloud instance in USD
            max_requests_per_day: Maximum requests the machine can handle per day without degrading latency

        Returns:
            float: Cost per 1,000 requests in USD
        """
        if max_requests_per_day <= 0:
            raise ValueError("max_requests_per_day must be greater than 0")

        # Calculate requests per month (assuming 30 days)
        requests_per_month = max_requests_per_day * 30

        if requests_per_month == 0:
            return 0.0

        # Cost per request
        cost_per_request = monthly_server_cost / requests_per_month

        # Cost per 1,000 requests
        cost_per_thousand = cost_per_request * 1000

        return round(cost_per_thousand, 4)

    @staticmethod
    def estimate_resource_utilization(
        responses: List[ExecutableChainResponse],
        container_memory_limit_gb: float = 8.0,
        container_cpu_cores: int = 2
    ) -> Dict[str, Any]:
        """
        Estimate resource utilization based on execution responses.

        Args:
            responses: List of ExecutableChainResponse objects
            container_memory_limit_gb: Total memory allocated to Ollama container (default: 8GB)
            container_cpu_cores: Number of CPU cores allocated (default: 2)

        Returns:
            Dict containing:
                - avg_latency: Average response latency in seconds
                - throughput_requests_per_min: Estimated requests per minute
                - estimated_memory_per_request_mb: Estimated memory usage per request
                - estimated_cpu_usage_percent: Estimated CPU usage percentage
                - total_execution_time_seconds: Total time for all requests
                - concurrent_capacity: Estimated concurrent request capacity
        """
        if not responses:
            return {
                "avg_latency": 0.0,
                "throughput_requests_per_min": 0.0,
                "estimated_memory_per_request_mb": 0.0,
                "estimated_cpu_usage_percent": 0.0,
                "total_execution_time_seconds": 0.0,
                "concurrent_capacity": 0
            }

        # Extract latencies from responses
        latencies = [res.latency for res in responses if hasattr(res, "latency")]

        if not latencies:
            return {
                "avg_latency": 0.0,
                "throughput_requests_per_min": 0.0,
                "estimated_memory_per_request_mb": 0.0,
                "estimated_cpu_usage_percent": 0.0,
                "total_execution_time_seconds": 0.0,
                "concurrent_capacity": 0
            }

        # Calculate latency metrics
        avg_latency = sum(latencies) / len(latencies)
        total_execution_time = sum(latencies)

        # Calculate throughput (requests per minute)
        throughput_requests_per_min = (60.0 / avg_latency) if avg_latency > 0 else 0.0

        # Estimate memory per request
        # Assumption: Base model memory (e.g., 4GB for quantized model) + overhead per request (~50MB)
        base_model_memory_gb = 4.0  # Conservative estimate for 4-bit quantized model
        memory_overhead_per_request_mb = 50
        estimated_memory_per_request_mb = (
            (base_model_memory_gb * 1024) / len(responses) + memory_overhead_per_request_mb
        )

        # Estimate CPU usage percentage
        # Assumption: CPU usage correlates with latency and number of cores
        # Normalize latency to 0-100 scale (assuming max reasonable latency of 10 seconds)
        max_reasonable_latency = 10.0
        base_cpu_usage = (avg_latency / max_reasonable_latency) * 100
        # Adjust for number of CPU cores (more cores = lower per-core usage)
        estimated_cpu_usage_percent = min(
            100.0,
            base_cpu_usage * (2.0 / max(1, container_cpu_cores))  # Normalize to 2-core baseline
        )

        # Calculate concurrent capacity (how many requests can run simultaneously)
        # Based on available memory divided by memory per request
        available_memory_mb = container_memory_limit_gb * 1024
        concurrent_capacity = max(
            1,
            int(available_memory_mb / (base_model_memory_gb * 1024))
        )

        return {
            "avg_latency": round(avg_latency, 4),
            "throughput_requests_per_min": round(throughput_requests_per_min, 2),
            "estimated_memory_per_request_mb": round(estimated_memory_per_request_mb, 2),
            "estimated_cpu_usage_percent": round(estimated_cpu_usage_percent, 2),
            "total_execution_time_seconds": round(total_execution_time, 2),
            "concurrent_capacity": concurrent_capacity
        }

    @staticmethod
    def calculate_cost_analysis(
        execution_responses: List[ExecutableChainResponse],
        monthly_server_cost: float = 100.0,
        max_requests_per_day: int = 5000,
        container_memory_limit_gb: float = 8.0,
        container_cpu_cores: int = 2
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive cost analysis for infrastructure.

        Args:
            execution_responses: List of ExecutableChainResponse objects
            monthly_server_cost: Monthly infrastructure cost in USD (default: $100)
            max_requests_per_day: Maximum requests without latency degradation (default: 5000)
            container_memory_limit_gb: Total container memory in GB (default: 8GB)
            container_cpu_cores: Number of CPU cores (default: 2)

        Returns:
            Dict containing:
                - cost_per_thousand_requests: Cost per 1,000 requests in USD
                - monthly_server_cost: Monthly infrastructure cost
                - max_requests_per_day: Maximum requests per day
                - resource_utilization: Resource utilization metrics
                - cost_efficiency_score: Overall efficiency score (0.0-1.0)
                - recommendations: List of optimization recommendations
        """
        # Calculate cost per 1,000 requests
        cost_per_thousand = CostTracker.calculate_cost_per_thousand_requests(
            monthly_server_cost=monthly_server_cost,
            max_requests_per_day=max_requests_per_day
        )

        # Calculate resource utilization
        resource_util = CostTracker.estimate_resource_utilization(
            responses=execution_responses,
            container_memory_limit_gb=container_memory_limit_gb,
            container_cpu_cores=container_cpu_cores
        )

        # Calculate cost efficiency score
        # Lower latency and higher throughput = better efficiency
        throughput = resource_util.get("throughput_requests_per_min", 0)
        avg_latency = resource_util.get("avg_latency", 0)

        # Efficiency score: 1.0 is ideal, lower values indicate room for improvement
        # Target: <2s latency, >30 req/min throughput
        latency_efficiency = max(0, 1 - (avg_latency / 5.0))  # Penalize latency > 5s
        throughput_efficiency = min(1.0, throughput / 30.0)  # Normalize to 30 req/min target
        cost_efficiency_score = (latency_efficiency + throughput_efficiency) / 2

        # Generate recommendations
        recommendations = []
        if avg_latency > 3.0:
            recommendations.append(
                "High latency detected. Consider using a quantized model (4-bit) or upgrading hardware."
            )
        if throughput < 20:
            recommendations.append(
                "Low throughput. Increase CPU cores or optimize prompt engineering."
            )
        if cost_per_thousand > 10:
            recommendations.append(
                f"High cost per 1,000 requests (${cost_per_thousand:.2f}). Review server capacity vs. cost."
            )
        if resource_util.get("estimated_cpu_usage_percent", 0) > 80:
            recommendations.append(
                "High CPU usage. Consider load balancing or upgrading compute resources."
            )
        if not recommendations:
            recommendations.append("Infrastructure utilization is within acceptable parameters.")

        return {
            "cost_per_thousand_requests": cost_per_thousand,
            "monthly_server_cost": monthly_server_cost,
            "max_requests_per_day": max_requests_per_day,
            "resource_utilization": resource_util,
            "cost_efficiency_score": round(cost_efficiency_score, 4),
            "recommendations": recommendations
        }
