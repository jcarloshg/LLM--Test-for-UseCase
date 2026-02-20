# test_dataset.py
from abc import ABC, abstractmethod
from typing import List
import json
import random
import logging
import traceback

from .test_case import TestCase

logger = logging.getLogger(__name__)


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

    @staticmethod
    def load_stories_for_test(num_easy: int = 1, num_medium: int = 2, num_hard: int = 1) -> List[TestCase]:
        """Load stratified user stories with configurable difficulty distribution.

        Args:
            num_easy: Number of easy test cases to load (default: 1)
            num_medium: Number of medium test cases to load (default: 2)
            num_hard: Number of hard test cases to load (default: 1)

        Returns:
            List of TestCase objects with specified difficulty distribution
        """
        try:
            with open('data/test/user_stories.json') as f:
                data = json.load(f)
                examples = data.get('examples', [])

            # Separate stories by difficulty
            easy_stories = [ex for ex in examples if ex.get('difficulty') == 'easy']
            medium_stories = [ex for ex in examples if ex.get('difficulty') == 'medium']
            hard_stories = [ex for ex in examples if ex.get('difficulty') == 'hard']

            # Randomly select based on parameters
            selected_easy = random.sample(
                easy_stories, min(num_easy, len(easy_stories)))
            selected_medium = random.sample(
                medium_stories, min(num_medium, len(medium_stories)))
            selected_hard = random.sample(
                hard_stories, min(num_hard, len(hard_stories)))

            # Create TestCase objects
            test_cases = []
            for story in selected_easy:
                test_cases.append(TestCase(
                    id=story['id'],
                    input=story['user_story'],
                    expected_output=json.dumps(story['expected_output']),
                    category="test_generation",
                    difficulty="easy",
                    metadata={"source": "user_stories"}
                ))

            for story in selected_medium:
                test_cases.append(TestCase(
                    id=story['id'],
                    input=story['user_story'],
                    expected_output=json.dumps(story['expected_output']),
                    category="test_generation",
                    difficulty="medium",
                    metadata={"source": "user_stories"}
                ))

            for story in selected_hard:
                test_cases.append(TestCase(
                    id=story['id'],
                    input=story['user_story'],
                    expected_output=json.dumps(story['expected_output']),
                    category="test_generation",
                    difficulty="hard",
                    metadata={"source": "user_stories"}
                ))

            return test_cases
        except FileNotFoundError as e:
            print(f"="*60)
            print(f"FileNotFoundError {str(e)}")
            print(f"="*60)
            return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"="*60)
            print(f"json.JSONDecodeError, KeyError {str(e)}")
            print(f"="*60)
            return []
        except Exception as e:
            print(f"="*60)
            print(f"Exception {str(e)}")
            print(f"="*60)
            return []
