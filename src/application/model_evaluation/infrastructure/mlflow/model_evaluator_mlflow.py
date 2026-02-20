from datetime import datetime
import mlflow

from ...models.model_evaluator import ModelEvaluator
from .mlflow_config import MLflowConfig


class ModelEvaluatorMLflow(ModelEvaluator):
    """MLflow-based implementation of ModelEvaluator"""

    def __init__(self, experiment_name: str):
        super().__init__(experiment_name)
        MLflowConfig.setup_experiment(experiment_name)

    def evaluate_model_suite(self, models: list, test_data: list):
        """
        Best Practice: Use parent run for the evaluation session,
        child runs for each model
        """
        with mlflow.start_run(run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M')}") as parent_run:

            # Log evaluation metadata
            mlflow.log_param("num_models", len(models))
            mlflow.log_param("test_size", len(test_data))
            mlflow.log_param("evaluation_date", datetime.now().isoformat())

            results = []

            for model_config in models:
                # Nested run for each model
                with mlflow.start_run(
                    run_name=f"{model_config['name']}",
                    nested=True
                ) as child_run:

                    result = self._evaluate_single_model(model_config, test_data)
                    results.append(result)

            # Log comparison summary in parent run
            self._log_comparison_summary(results)

            return results

    def _evaluate_single_model(self, model_config: dict, test_data: list):
        """
        Evaluate a single model against test data.

        Args:
            model_config: Configuration dictionary for the model
            test_data: Test dataset

        Returns:
            Evaluation results for the model
        """
        # Log model configuration
        mlflow.log_param("model_name", model_config.get("name", "unknown"))
        mlflow.log_param("model_type", model_config.get("type", "unknown"))

        # Placeholder for actual model evaluation logic
        # This should be overridden or extended in subclasses
        result = {
            "name": model_config.get("name"),
            "type": model_config.get("type"),
            "metrics": {}
        }

        return result

    def _log_comparison_summary(self, results: list):
        """
        Log summary comparison of all evaluated models.

        Args:
            results: List of evaluation results from all models
        """
        mlflow.log_param("total_models_evaluated", len(results))

        # Log summary information for each model
        for idx, result in enumerate(results):
            model_name = result.get("name", f"model_{idx}")
            mlflow.log_param(f"evaluated_model_{idx}", model_name)
