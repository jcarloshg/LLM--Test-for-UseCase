from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    """Abstract base class for model evaluation with MLflow logging"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name

    @abstractmethod
    def evaluate_model_suite(self, models: list, test_data: list):
        """
        Best Practice: Use parent run for the evaluation session,
        child runs for each model
        """
        pass

    @abstractmethod
    def _evaluate_single_model(self, model_config: dict, test_data: list):
        """
        Evaluate a single model against test data.

        Args:
            model_config: Configuration dictionary for the model
            test_data: Test dataset

        Returns:
            Evaluation results for the model
        """
        pass

    @abstractmethod
    def _log_comparison_summary(self, results: list):
        """
        Log summary comparison of all evaluated models.

        Args:
            results: List of evaluation results from all models
        """
        pass
