# test_dataset.py
from abc import ABC, abstractmethod
from typing import List
import json
import random

from src.application.evaluate_models.model.test_case import TestCase


class EvaluationDataset(ABC):
    """
    Best Practice: Stratified test set covering:
    - Easy/Medium/Hard cases
    - Different input lengths
    - Edge cases and failure modes
    """

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
                examples = data.get('user_stories', data.get('examples', []))

            # Randomly select stories by difficulty
            selected_stories = []
            for difficulty, num_samples in [('easy', num_easy), ('medium', num_medium), ('hard', num_hard)]:
                difficulty_stories = [
                    ex for ex in examples if ex.get('difficulty') == difficulty
                ]
                selected = random.sample(
                    difficulty_stories, min(num_samples, len(difficulty_stories)))
                selected_stories.extend(selected)

            # Create TestCase objects
            test_cases = [
                TestCase(
                    id=story['id'],
                    user_story=story['user_story'],
                    difficulty=story.get('difficulty')
                )
                for story in selected_stories
            ]

            return test_cases

        except FileNotFoundError as e:
            print("="*60)
            print(f"[EvaluationDataset] - FileNotFoundError {str(e)}")
            print("="*60)
            return []
        except (json.JSONDecodeError, KeyError) as e:
            print("="*60)
            print(
                f"[EvaluationDataset] - json.JSONDecodeError, KeyError {str(e)}")
            print("="*60)
            return []
        except Exception as e:
            print("="*60)
            print(f"[EvaluationDataset] - Exception {str(e)}")
            print("="*60)
            return []
