# detailed_evaluation.py
from typing import Any, Dict, List

from src.application.generate_test.models.structure import TestCase
from src.application.model_evaluation.models.evaluators import EvaluationMetrics


class DetailedEvaluator:
    """
    Best Practice: Break down performance by test case characteristics
    """

    def __init__(self, evaluator: EvaluationMetrics):
        self.evaluator = evaluator

    def evaluate_detailed(
        self,
        test_cases: List[TestCase],
        predictions: list
    ) -> Dict[str, Any]:
        """
        Evaluate with breakdown by difficulty, category, etc.
        """

        results = {
            "overall": {},
            "by_difficulty": {},
            "by_category": {},
            "failures": []
        }

        # Overall metrics
        expected = [tc.expected_output for tc in test_cases]
        results["overall"] = self.evaluator.evaluate(predictions, expected)

        # By difficulty
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_cases = [
                tc for tc in test_cases if tc.difficulty == difficulty]
            if difficulty_cases:
                diff_predictions = [predictions[i] for i, tc in enumerate(test_cases)
                                    if tc.difficulty == difficulty]
                diff_expected = [tc.expected_output for tc in difficulty_cases]

                results["by_difficulty"][difficulty] = self.evaluator.evaluate(
                    diff_predictions,
                    diff_expected
                )

        # By category
        categories = set(tc.category for tc in test_cases)
        for category in categories:
            cat_cases = [tc for tc in test_cases if tc.category == category]
            cat_predictions = [predictions[i] for i, tc in enumerate(test_cases)
                               if tc.category == category]
            cat_expected = [tc.expected_output for tc in cat_cases]

            results["by_category"][category] = self.evaluator.evaluate(
                cat_predictions,
                cat_expected
            )

        # Track failures
        for i, (tc, pred) in enumerate(zip(test_cases, predictions)):
            if pred.strip().lower() != tc.expected_output.strip().lower():
                results["failures"].append({
                    "test_id": tc.id,
                    "difficulty": tc.difficulty,
                    "expected": tc.expected_output,
                    "predicted": pred,
                    "input": tc.input[:100]
                })

        return results
