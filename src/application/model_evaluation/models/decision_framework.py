# decision_framework.py
from typing import List, Dict
import pandas as pd


class ModelDecisionFramework:
    """
    Best Practice: Systematic decision-making with weighted criteria
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Define importance weights for each criterion

        Default weights:
        - accuracy: 40%
        - cost: 25%
        - latency: 20%
        - reliability: 15%
        """
        self.weights = weights or {
            "accuracy": 0.40,
            "cost": 0.25,
            "latency": 0.20,
            "reliability": 0.15
        }

    def score_models(self, evaluations: List[Dict]) -> pd.DataFrame:
        """
        Best Practice: Normalize and weight all criteria

        Args:
            evaluations: List of dicts with model evaluation results

        Returns:
            DataFrame with scores and ranking
        """

        df = pd.DataFrame(evaluations)

        # Normalize each metric to 0-1
        df['accuracy_norm'] = self._normalize(df['accuracy'])
        df['cost_norm'] = 1 - \
            self._normalize(df['monthly_cost'])  # Lower is better
        df['latency_norm'] = 1 - \
            self._normalize(df['p95_latency'])  # Lower is better
        df['reliability_norm'] = self._normalize(df['success_rate'])

        # Calculate weighted score
        df['weighted_score'] = (
            df['accuracy_norm'] * self.weights['accuracy'] +
            df['cost_norm'] * self.weights['cost'] +
            df['latency_norm'] * self.weights['latency'] +
            df['reliability_norm'] * self.weights['reliability']
        )

        # Rank models
        df['rank'] = df['weighted_score'].rank(ascending=False)

        return df.sort_values('weighted_score', ascending=False)

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Normalize to 0-1 range"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([1.0] * len(series))
        return (series - min_val) / (max_val - min_val)

    def generate_recommendation(self, scored_df: pd.DataFrame) -> str:
        """
        Best Practice: Provide clear recommendation with reasoning
        """

        winner = scored_df.iloc[0]

        recommendation = f"""
## Model Selection Recommendation

**Selected Model: {winner['model_name']}**

### Key Metrics:
- Accuracy: {winner['accuracy']:.2%}
- Monthly Cost: ${winner['monthly_cost']:.2f}
- P95 Latency: {winner['p95_latency']:.2f}s
- Success Rate: {winner['success_rate']:.2%}

### Weighted Score: {winner['weighted_score']:.3f}

### Reasoning:
"""

        # Add reasoning based on strengths
        if winner['accuracy_norm'] > 0.8:
            recommendation += "\n- ✅ Excellent accuracy performance"
        if winner['cost_norm'] > 0.7:
            recommendation += "\n- ✅ Cost-effective for expected usage"
        if winner['latency_norm'] > 0.7:
            recommendation += "\n- ✅ Meets latency requirements"

        # Add runner-up comparison
        if len(scored_df) > 1:
            runner_up = scored_df.iloc[1]
            recommendation += f"\n\n### Runner-up: {runner_up['model_name']}"
            recommendation += f"\n- Score: {runner_up['weighted_score']:.3f}"
            recommendation += f"\n- Key difference: "

            if winner['accuracy'] > runner_up['accuracy']:
                diff = (winner['accuracy'] - runner_up['accuracy']) * 100
                recommendation += f"{diff:.1f}% higher accuracy"
            elif winner['monthly_cost'] < runner_up['monthly_cost']:
                diff = runner_up['monthly_cost'] - winner['monthly_cost']
                recommendation += f"${diff:.2f}/month cheaper"

        return recommendation
