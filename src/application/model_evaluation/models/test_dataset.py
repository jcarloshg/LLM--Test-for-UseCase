# test_dataset.py
from abc import ABC, abstractmethod
from typing import List
import json

from .test_case import TestCase


class EvaluationDataset(ABC):
    """
    Best Practice: Stratified test set covering:
    - Easy/Medium/Hard cases
    - Different input lengths
    - Edge cases and failure modes
    """

    @staticmethod
    def create_stratified_dataset() -> List[TestCase]:
        """Create balanced test dataset"""

        # ─────────────────────────────────────
        # TODO - set the path for examples
        # ─────────────────────────────────────

        return [
            # Easy cases (30%)
            TestCase(
                id="easy_001",
                input="Classify: This product is amazing!",
                expected_output="positive",
                category="classification",
                difficulty="easy",
                metadata={"length": "short"}
            ),

            # Medium cases (50%)
            TestCase(
                id="medium_001",
                input="Classify: The product works but shipping was slow",
                expected_output="mixed",
                category="classification",
                difficulty="medium",
                metadata={"length": "medium", "nuance": "mixed_sentiment"}
            ),

            # Hard cases (20%)
            TestCase(
                id="hard_001",
                input="Classify: I don't NOT dislike it, though the quality isn't bad",
                expected_output="neutral",
                category="classification",
                difficulty="hard",
                metadata={"length": "medium", "nuance": "double_negative"}
            ),

            # Edge cases
            TestCase(
                id="edge_001",
                input="Classify: 👍👍👍",
                expected_output="positive",
                category="classification",
                difficulty="hard",
                metadata={"type": "emoji_only"}
            )
        ]

    @staticmethod
    def save_dataset_locally(dataset: List[TestCase], filepath: str):
        """Save with versioning locally"""
        with open(filepath, 'w') as f:
            json.dump([tc.dict() for tc in dataset], f, indent=2)

    @abstractmethod
    def save_dataset_platform(self, dataset: List[TestCase]):
        """Log data on platform for example mlflow"""
        pass
