# dataset_versioning.py
import mlflow
import hashlib
from typing import List
import json

from src.application.generate_test.models.structure import TestCase


class DatasetVersioning:
    """Track dataset versions in MLflow"""

    @staticmethod
    def log_dataset_version(dataset: List[TestCase], version: str):
        """
        Best Practice: Log dataset hash and metadata
        """

        # Calculate dataset hash
        dataset_json = json.dumps([tc.dict()
                                  for tc in dataset], sort_keys=True)
        dataset_hash = hashlib.md5(dataset_json.encode()).hexdigest()

        # Log to MLflow
        with mlflow.start_run(run_name=f"dataset-{version}"):
            mlflow.log_param("dataset_version", version)
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("num_samples", len(dataset))

            # Log distribution
            difficulties = {}
            for tc in dataset:
                difficulties[tc.difficulty] = difficulties.get(
                    tc.difficulty, 0) + 1

            for diff, count in difficulties.items():
                mlflow.log_metric(f"count_{diff}", count)

            # Save dataset as artifact
            filepath = f"datasets/test_dataset_{version}.json"
            with open(filepath, 'w') as f:
                json.dump([tc.dict() for tc in dataset], f, indent=2)

            mlflow.log_artifact(filepath)

        return dataset_hash
