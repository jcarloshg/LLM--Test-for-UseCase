# cost_analysis.py
from typing import Dict

from src.application.model_evaluation.models.model_configs import ModelConfig


class CostAnalyzer:
    """
    Best Practice: Calculate full production cost estimates
    """

    @staticmethod
    def estimate_monthly_cost(
        model_config: ModelConfig,
        avg_input_tokens: int,
        avg_output_tokens: int,
        requests_per_day: int
    ) -> Dict[str, float]:
        """
        Estimate monthly costs based on expected usage
        """

        # Calculate per-request cost
        per_request_cost = (
            (avg_input_tokens / 1000) * model_config.cost_per_1k_input +
            (avg_output_tokens / 1000) * model_config.cost_per_1k_output
        )

        # Monthly estimates
        requests_per_month = requests_per_day * 30
        monthly_cost = per_request_cost * requests_per_month

        # Infrastructure costs for self-hosted
        infra_cost = 0.0
        if model_config.provider == "ollama":
            # Estimate GPU costs (e.g., A10G at $0.75/hr)
            infra_cost = 0.75 * 24 * 30  # $540/month

        return {
            "per_request": per_request_cost,
            "daily": per_request_cost * requests_per_day,
            "monthly_api": monthly_cost,
            "monthly_infra": infra_cost,
            "monthly_total": monthly_cost + infra_cost,
            "yearly_total": (monthly_cost + infra_cost) * 12
        }

    @staticmethod
    def cost_benefit_score(
        accuracy: float,
        monthly_cost: float,
        max_budget: float
    ) -> float:
        """
        Best Practice: Calculate value score (accuracy / cost ratio)

        Returns score where higher = better value
        """
        if monthly_cost > max_budget:
            return 0.0  # Over budget

        # Normalize cost to 0-1 (lower is better)
        cost_score = 1 - (monthly_cost / max_budget)

        # Combined score: accuracy weighted by cost efficiency
        return accuracy * (0.7) + cost_score * (0.3)
