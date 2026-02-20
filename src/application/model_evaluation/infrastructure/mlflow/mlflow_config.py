import mlflow
from datetime import datetime
from mlflow.exceptions import MlflowException


class MLflowConfig:
    """Centralized MLflow configuration"""

    @staticmethod
    def setup_experiment(experiment_name: str = "llm-model-evaluation"):
        """
        Create or get experiment with proper naming

        Best Practice: Use descriptive names with timestamps for iterations
        """
        # Set tracking URI (local or remote)
        # Or "http://mlflow-server:5000"
        mlflow.set_tracking_uri("file:./mlruns")

        created_date: str = datetime.now().isoformat()
        experiment_name = f"{experiment_name}-{created_date}"

        # Create experiment with metadata
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags={
                    "project": "model-evaluation",
                    "phase": "phase-3-evaluation",
                    "created_date": created_date
                }
            )
        except MlflowException:
            experiment_id = mlflow.get_experiment_by_name(
                name=experiment_name
            ).experiment_id

        mlflow.set_experiment(experiment_name)
        return experiment_id
