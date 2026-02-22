"""
Quality tracking and assessment module for test case artifacts.
Measures quality through heuristic rules and structural validity,
not traditional classification metrics.
"""
from typing import List, Dict, Any

from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


class QualityTracker:
    """
    Tracks and evaluates test case generation quality.

    Quality is measured through:
    - Precondition completeness (not empty)
    - Test case structural validity (minimum logical steps)
    - Overall artifact usefulness and structure
    """

    @staticmethod
    def evaluate_preconditions(preconditions: Any) -> float:
        """
        Evaluate preconditions quality.

        Args:
            preconditions: Preconditions data (string, list, or dict)

        Returns:
            float: Score between 0.0 and 1.0
                - 1.0: Well-formed and non-empty
                - 0.5: Partially formed or minimal content
                - 0.0: Empty or invalid
        """
        if preconditions is None:
            return 0.0

        if isinstance(preconditions, str):
            if not preconditions.strip():
                return 0.0
            if len(preconditions.strip()) < 10:
                return 0.5
            return 1.0

        if isinstance(preconditions, (list, dict)):
            if len(preconditions) == 0:
                return 0.0
            if len(preconditions) < 3:
                return 0.5
            return 1.0

        return 0.5

    @staticmethod
    def count_logical_steps(test_case: Dict[str, Any]) -> int:
        """
        Count logical steps in a test case.

        Args:
            test_case: Test case object

        Returns:
            int: Number of identifiable logical steps
        """
        steps = 0

        # Count steps from explicit fields (primary indicators)
        if "steps" in test_case and isinstance(test_case["steps"], list):
            steps += len(test_case["steps"])

        if "actions" in test_case and isinstance(test_case["actions"], list):
            steps += len(test_case["actions"])

        if "assertions" in test_case and isinstance(test_case["assertions"], list):
            steps += len(test_case["assertions"])

        # Count description sentences only if no explicit steps found
        if steps == 0 and "description" in test_case:
            desc = test_case["description"]
            if isinstance(desc, str):
                sentences = desc.split(".")
                steps += len([s for s in sentences if s.strip()])

        return steps

    @staticmethod
    def evaluate_test_case_structure(test_case: Dict[str, Any], min_steps: int = 3) -> float:
        """
        Evaluate test case structure quality.

        Args:
            test_case: Test case object to evaluate
            min_steps: Minimum required logical steps (default: 3)

        Returns:
            float: Score between 0.0 and 1.0
                - 1.0: Has minimum steps and valid structure
                - 0.5: Has fewer steps but valid structure
                - 0.0: Invalid or missing structure
        """
        if not isinstance(test_case, dict):
            return 0.0

        required_fields = ["description", "expected_output"]
        has_required = all(field in test_case for field in required_fields)

        if not has_required:
            return 0.0

        # Check for empty descriptions
        description = test_case.get("description", "")
        if isinstance(description, str) and not description.strip():
            return 0.0

        # Check for empty expected_output
        expected_output = test_case.get("expected_output", "")
        if isinstance(expected_output, str) and not expected_output.strip():
            return 0.0

        # Count logical steps
        logical_steps = QualityTracker.count_logical_steps(test_case)

        if logical_steps >= min_steps:
            return 1.0
        if logical_steps > 0:
            return 0.5
        return 0.0

    @staticmethod
    def calculate_quality_score(
        execution_responses: List[ExecutableChainResponse],
        min_test_steps: int = 2,
        passing_threshold: float = 0.7,
        structure_weight: float = 0.6,
        precondition_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Calculate quality score for a batch of test case generation executions.

        Args:
            execution_responses: List of ExecutableChainResponse objects from test generation
            min_test_steps: Minimum required logical steps per test case (default: 3)
            passing_threshold: Quality threshold to consider test as passing (default: 0.7)
            structure_weight: Weight for structure quality (default: 0.6, must sum to 1.0 with precondition_weight)
            precondition_weight: Weight for precondition quality (default: 0.4)

        Returns:
            Dict containing:
                - quality_score: Average weighted quality score (0.0 to 1.0)
                - precondition_score: Average precondition quality
                - structure_score: Average structure quality
                - total_tests: Number of tests evaluated
                - passing_tests: Number of tests meeting quality threshold
                - passing_rate: Percentage of tests meeting threshold
        """
        # Validate weights
        if abs((structure_weight + precondition_weight) - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {structure_weight + precondition_weight}"
            )

        if not execution_responses:
            return {
                "quality_score": 0.0,
                "precondition_score": 0.0,
                "structure_score": 0.0,
                "total_tests": 0,
                "passing_tests": 0,
                "passing_rate": 0.0
            }

        quality_scores = []
        precondition_scores = []
        structure_scores = []

        for response in execution_responses:
            if not hasattr(response, "result"):
                continue

            result = response.result
            if not isinstance(result, dict):
                continue

            # Extract test cases from result with validation
            test_cases = result.get("test_cases", [])
            if isinstance(test_cases, dict):
                test_cases = [test_cases]
            elif not isinstance(test_cases, list):
                continue

            for test_case in test_cases:
                if not isinstance(test_case, dict):
                    continue

                # Evaluate preconditions
                preconditions = test_case.get("preconditions")
                precond_score = QualityTracker.evaluate_preconditions(
                    preconditions)

                # Evaluate structure
                struct_score = QualityTracker.evaluate_test_case_structure(
                    test_case,
                    min_steps=min_test_steps
                )

                # Overall quality using weighted average
                overall_score = (
                    (struct_score * structure_weight) +
                    (precond_score * precondition_weight)
                )

                quality_scores.append(overall_score)
                precondition_scores.append(precond_score)
                structure_scores.append(struct_score)

        # Calculate averages
        avg_quality = sum(quality_scores) / \
            len(quality_scores) if quality_scores else 0.0
        avg_precondition = (
            sum(precondition_scores) / len(precondition_scores)
            if precondition_scores else 0.0
        )
        avg_structure = (
            sum(structure_scores) / len(structure_scores)
            if structure_scores else 0.0
        )

        # Count passing tests based on configurable threshold
        passing_tests = sum(
            1 for score in quality_scores if score >= passing_threshold)
        passing_rate = passing_tests / \
            len(quality_scores) if quality_scores else 0.0

        return {
            "quality_score": avg_quality,
            "precondition_score": avg_precondition,
            "structure_score": avg_structure,
            "total_tests": len(quality_scores),
            "passing_tests": passing_tests,
            "passing_rate": passing_rate
        }
