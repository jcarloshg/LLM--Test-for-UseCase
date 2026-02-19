# src/mlflow_tracker.py
import mlflow
import mlflow.pyfunc
from datetime import datetime
import json


class MLflowTracker:
    """Track experiments and metrics in MLflow"""

    def __init__(self, experiment_name: str = "test-case-generation"):
        mlflow.set_experiment(experiment_name)

    def log_generation(
        self,
        user_story: str,
        test_cases: list,
        structure_validation: dict,
        quality_metrics: dict,
        coverage_metrics: dict,
        latency: float,
        model_info: dict
    ):
        """Log a single test case generation run"""

        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model", model_info.get('model'))
            mlflow.log_param("provider", model_info.get('provider'))
            mlflow.log_param("user_story", user_story[:100])

            # Log metrics
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("structure_valid",
                              1.0 if structure_validation['valid'] else 0.0)
            mlflow.log_metric("test_case_count", structure_validation['count'])

            if quality_metrics:
                mlflow.log_metric("relevance_score",
                                  quality_metrics['relevance'])
                mlflow.log_metric("coverage_score",
                                  quality_metrics['coverage'])
                mlflow.log_metric("clarity_score", quality_metrics['clarity'])
                mlflow.log_metric("overall_quality",
                                  quality_metrics['overall'])

            if coverage_metrics:
                mlflow.log_metric("coverage_score",
                                  coverage_metrics['coverage_score'])
                mlflow.log_metric("priority_diversity",
                                  coverage_metrics['priority_diversity'])

            # Log artifacts
            mlflow.log_dict({
                "user_story": user_story,
                "test_cases": test_cases,
                "validations": {
                    "structure": structure_validation,
                    "quality": quality_metrics,
                    "coverage": coverage_metrics
                }
            }, "generation_result.json")

            # Log tags
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            mlflow.set_tag("passed_validation", structure_validation['valid'])

            run_id = mlflow.active_run().info.run_id

        return run_id
