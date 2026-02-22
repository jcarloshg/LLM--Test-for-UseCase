# evaluate_models_application.py
from datetime import datetime
from typing import List, Dict, Any
import json
import mlflow

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.infra.executable_chain.executable_chain_v1 import ExecutableChainV1
from src.application.create_tests.infra.vectorstores.faiss_vectorstore import load_faiss_vectorstore
from src.application.create_tests.models import RAG_PROMPT
from src.application.evaluate_models.model.model_configs import ModelConfig, ModelRegistry
from src.application.evaluate_models.model.test_case import TestCase
from src.application.evaluate_models.model.test_dataset import EvaluationDataset

from src.application.evaluate_models.infra.mlflow_config import MLflowConfig


class EvaluateModelsApplication:

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_id = MLflowConfig.setup_experiment(experiment_name)

    def run_complete_evaluation(
        self,
        models_to_test: List[ModelConfig],
        test_dataset: List[TestCase],
        executable_chain: ExecutableChain,
        expected_requests_per_day: int = 1000
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline with MLflow tracking
        """

        # Start parent run
        with mlflow.start_run(
            run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M')}"
        ) as parent_run:

            # Log evaluation parameters
            mlflow.log_param("num_models", len(models_to_test))
            mlflow.log_param("test_size", len(test_dataset))
            mlflow.log_param("expected_requests_per_day",
                             expected_requests_per_day)
            mlflow.log_param("evaluation_date", datetime.now().isoformat())

            all_results = []

            # Evaluate each model
            for model_config in models_to_test:
                print(f"\n{'='*60}")
                print(f"Evaluating: {model_config.name}")
                print(f"{'='*60}")

                result = self._evaluate_single_model(
                    model_config,
                    test_dataset,
                    expected_requests_per_day,
                    executable_chain=executable_chain
                )

        return {}

    def _evaluate_single_model(
        self,
        model_config: ModelConfig,
        test_dataset: List[TestCase],
        expected_requests_per_day: int,
        executable_chain: ExecutableChain,
    ) -> Dict[str, Any]:
        """Evaluate single model with complete metrics"""

        # Log model configuration
        mlflow.log_params({
            "model_name": model_config.name,
            "model_provider": model_config.provider,
            "model_id": model_config.model_id,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens
        })

        print(f"Running {len(test_dataset)} test cases...")

        for i, test_case in enumerate(test_dataset, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_dataset)}")

            print("\n")
            print(test_case.user_story)
            user_story = test_case.user_story
            current_llm = model_config.llm
            executable_chain.update_llm(current_llm)
            executable_chain_response = executable_chain.execute(
                prompt=user_story
            )
            print(executable_chain_response)

        return {}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EvaluateModelsApplication("model-evaluation")

    # ─────────────────────────────────────
    # Load FAISS vectorstore
    # ─────────────────────────────────────
    try:
        retriever = load_faiss_vectorstore()
    except FileNotFoundError as e:
        print(f"="*60)
        print(f"[creaet_test_controller] - FileNotFoundError {str(e)} ")
        print(f"="*60)
        raise Exception("Something was wriong")

    # ─────────────────────────────────────
    # update the chain events
    # ─────────────────────────────────────
    executable_chain_v1 = ExecutableChainV1(
        retriever=retriever,
        prompt_emplate=RAG_PROMPT
    )

    test_cases = EvaluationDataset.load_stories_for_test(
        num_easy=1,
        num_medium=1,
        num_hard=1,
    )

    model_registry = ModelRegistry()
    models_for_evaluation = model_registry.get_models_to_compare()

    # Run evaluation
    results = pipeline.run_complete_evaluation(
        expected_requests_per_day=5000,  # Adjust based on expected traffic
        test_dataset=test_cases,
        models_to_test=models_for_evaluation,
        executable_chain=executable_chain_v1
    )

    # # Generate report
    # pipeline.generate_report()

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review MLflow dashboard: http://localhost:5000")
    print("2. Read recommendation.md for decision rationale")
    print("3. Check model_comparison.csv for detailed metrics")
