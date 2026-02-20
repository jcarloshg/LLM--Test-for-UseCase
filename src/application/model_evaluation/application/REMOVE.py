# complete_evaluation_pipeline.py
from datetime import datetime
from typing import List, Dict, Any
import mlflow

from src.application.generate_test.infrastructure.anthropic.llm_client_anthropic import LLMClientAnthropic
from src.application.generate_test.infrastructure.ollama.llm_client_ollama import LLMClientOllama
from src.application.generate_test.infrastructure.prompt_builder_cla import PromptBuilderCla
from src.application.generate_test.models.prompt_builder import PromptBuilder

from src.application.model_evaluation.infrastructure.mlflow.mlflow_config import MLflowConfig
from src.application.model_evaluation.models.cost_analysis import CostAnalyzer
from src.application.model_evaluation.models.decision_framework import ModelDecisionFramework
from src.application.model_evaluation.models.detailed_evaluation import DetailedEvaluator
from src.application.model_evaluation.models.evaluators import EvaluationMetrics
from src.application.model_evaluation.models.model_configs import ModelConfig, ModelRegistry
from src.application.model_evaluation.models.performance_tracker import PerformanceTracker
from src.application.model_evaluation.models.test_case import TestCase
from src.application.model_evaluation.models.test_dataset import EvaluationDataset


class MLflowModelEvaluationPipeline:
    """
    Complete pipeline for model evaluation with MLflow tracking

    Usage:
        pipeline = MLflowModelEvaluationPipeline("model-selection-v1")
        results = pipeline.run_complete_evaluation()
        pipeline.generate_report()
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_id = MLflowConfig.setup_experiment(experiment_name)

        # Initialize components
        self.evaluator = EvaluationMetrics()
        self.detailed_evaluator = DetailedEvaluator(self.evaluator)
        self.cost_analyzer = CostAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.decision_framework = ModelDecisionFramework()
        self.prompt_builder: PromptBuilder = PromptBuilderCla()

    def run_complete_evaluation(
        self,
        models_to_test: List[ModelConfig] = None,
        test_dataset: List[TestCase] = None,
        expected_requests_per_day: int = 1000
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline with MLflow tracking
        """

        # Use defaults if not provided
        if models_to_test is None:
            models_to_test = ModelRegistry.get_models_to_compare()

        if test_dataset is None:
            test_dataset = EvaluationDataset.load_stories_for_test()

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
                    expected_requests_per_day
                )

                all_results.append(result)

            # Generate comparison and recommendation
            if all_results:
                comparison_df = self.decision_framework.score_models(all_results)
                recommendation = self.decision_framework.generate_recommendation(comparison_df)

                # Log comparison summary
                self._log_comparison_summary(comparison_df, recommendation)

                return {
                    "parent_run_id": parent_run.info.run_id,
                    "results": all_results,
                    "comparison": comparison_df.to_dict('records'),
                    "recommendation": recommendation
                }
            else:
                print("\n⚠️  All model evaluations failed. No results to compare.")
                return {
                    "parent_run_id": parent_run.info.run_id,
                    "results": all_results,
                    "comparison": [],
                    "recommendation": "Unable to generate recommendation due to evaluation failures."
                }

    def _evaluate_single_model(
        self,
        model_config: ModelConfig,
        test_dataset: List[TestCase],
        expected_requests_per_day: int
    ) -> Dict[str, Any]:
        """Evaluate single model with complete metrics"""

        with mlflow.start_run(
            run_name=model_config.name,
            nested=True
        ) as run:

            # Log model configuration
            mlflow.log_params({
                "model_name": model_config.name,
                "model_provider": model_config.provider,
                "model_id": model_config.model_id,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens
            })

            # Initialize appropriate client based on provider
            # Use the llm_config from model_config
            if model_config.provider.lower() == "anthropic":
                client = LLMClientAnthropic(model_config.llm_config)
            elif model_config.provider.lower() == "ollama":
                client = LLMClientOllama(model_config.llm_config)
            else:
                raise ValueError(
                    f"Unsupported provider: {model_config.provider}")

            # Run predictions
            predictions = []
            latencies = []
            token_counts = []
            errors = 0

            print(f"Running {len(test_dataset)} test cases...")

            for i, test_case in enumerate(test_dataset, 1):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(test_dataset)}")

                prompts = self.prompt_builder.build(test_case.input)

                try:
                    result = client.generate(
                        prompt=prompts.get("user", ""),
                        system_prompt=prompts.get("system", "")
                    )
                    predictions.append(result.text)
                    latencies.append(result.latency)
                    token_counts.append(result.tokens)
                except Exception as e:
                    print(f"  Error on test case {i}: {e}")
                    predictions.append("")
                    errors += 1

            # Check if we have any successful predictions
            if not latencies:
                print(
                    f"  ⚠️  All {len(test_dataset)} test cases failed for {model_config.name}")
                return {
                    "model_name": model_config.name,
                    "run_id": run.info.run_id,
                    "accuracy": 0.0,
                    "semantic_similarity": 0.0,
                    "f1_score": 0.0,
                    "p95_latency": 0.0,
                    "mean_latency": 0.0,
                    "monthly_cost": 0.0,
                    "per_request_cost": 0.0,
                    "success_rate": 0.0,
                    "total_tests": len(test_dataset),
                    "easy_accuracy": 0.0,
                    "hard_accuracy": 0.0
                }

            # Calculate metrics
            # 1. Quality metrics
            quality_metrics = self.detailed_evaluator.evaluate_detailed(
                test_dataset,
                predictions
            )

            # 2. Performance metrics
            latency_stats = self.performance_tracker.calculate_latency_stats(
                latencies)

            # 3. Cost analysis
            avg_input_tokens = int(sum(token_counts) /
                                   len(token_counts)) if token_counts else 0
            avg_output_tokens = int(
                sum(token_counts) / len(token_counts)) if token_counts else 0

            cost_estimates = self.cost_analyzer.estimate_monthly_cost(
                model_config,
                avg_input_tokens,
                avg_output_tokens,
                expected_requests_per_day
            )

            # Log to MLflow
            self._log_metrics_to_mlflow(
                quality_metrics,
                latency_stats,
                cost_estimates,
                errors,
                len(test_dataset)
            )

            # Save predictions as artifact
            self._log_predictions_artifact(test_dataset, predictions)

            # Compile result
            result = {
                "model_name": model_config.name,
                "run_id": run.info.run_id,
                "accuracy": quality_metrics['overall'].get('exact_match', 0),
                "semantic_similarity": quality_metrics['overall'].get('semantic_similarity', 0),
                "f1_score": quality_metrics['overall'].get('f1_score', 0),
                "p95_latency": latency_stats['p95'],
                "mean_latency": latency_stats['mean'],
                "monthly_cost": cost_estimates['monthly_total'],
                "per_request_cost": cost_estimates['per_request'],
                "success_rate": 1 - (errors / len(test_dataset)),
                "total_tests": len(test_dataset),
                "easy_accuracy": quality_metrics['by_difficulty'].get('easy', {}).get('exact_match', 0),
                "hard_accuracy": quality_metrics['by_difficulty'].get('hard', {}).get('exact_match', 0)
            }

            print(f"\n✅ {model_config.name} Evaluation Complete:")
            print(f"   Accuracy: {result['accuracy']:.2%}")
            print(f"   P95 Latency: {result['p95_latency']:.2f}s")
            print(f"   Monthly Cost: ${result['monthly_cost']:.2f}")

            return result

    def _log_metrics_to_mlflow(
        self,
        quality_metrics: dict,
        latency_stats: dict,
        cost_estimates: dict,
        errors: int,
        total: int
    ):
        """Log all metrics to MLflow"""

        # Quality metrics
        for key, value in quality_metrics['overall'].items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"quality_{key}", value)

        # By difficulty
        for difficulty, metrics in quality_metrics.get('by_difficulty', {}).items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{difficulty}_{key}", value)

        # Latency metrics
        for key, value in latency_stats.items():
            mlflow.log_metric(f"latency_{key}", value)

        # Cost metrics
        for key, value in cost_estimates.items():
            mlflow.log_metric(f"cost_{key}", value)

        # Reliability
        mlflow.log_metric("error_rate", errors / total)
        mlflow.log_metric("success_rate", 1 - (errors / total))

    def _log_predictions_artifact(self, test_dataset: List[TestCase], predictions: list):
        """Save predictions as MLflow artifact"""

        import json

        results = []
        for tc, pred in zip(test_dataset, predictions):
            results.append({
                "test_id": tc.id,
                "input": tc.input,
                "expected": tc.expected_output,
                "predicted": pred,
                "match": pred.strip().lower() == tc.expected_output.strip().lower()
            })

        filepath = "predictions.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(filepath, artifact_path="predictions")

    def _log_comparison_summary(self, comparison_df, recommendation: str):
        """Log final comparison to MLflow"""

        # Save comparison table
        comparison_df.to_csv("model_comparison.csv", index=False)
        mlflow.log_artifact("model_comparison.csv", artifact_path="comparison")

        # Save recommendation
        with open("recommendation.md", 'w') as f:
            f.write(recommendation)
        mlflow.log_artifact("recommendation.md", artifact_path="comparison")

        # Log winner info
        winner = comparison_df.iloc[0]
        mlflow.log_param("selected_model", winner['model_name'])
        mlflow.log_metric("winner_score", winner['weighted_score'])

    def generate_report(self, run_id: str = None):
        """
        Generate HTML report from MLflow run

        Best Practice: Create shareable reports
        """

        if run_id is None:
            # Get latest run
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(
                    f"⚠️  Could not find experiment '{self.experiment_name}'")
                return

            runs = mlflow.search_runs([experiment.experiment_id])
            if runs.empty:
                print(
                    f"⚠️  No runs found for experiment '{self.experiment_name}'")
                return

            run_id = runs.iloc[0]['run_id']

        # Load run data
        try:
            run = mlflow.get_run(run_id)
        except Exception as e:
            print(f"⚠️  Could not load run {run_id}: {e}")
            return

        # Generate HTML report
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .winner {{ background-color: #d4edda; }}
    </style>
</head>
<body>
    <h1>Model Evaluation Report</h1>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p><strong>Run ID:</strong> {run_id}</p>

    <h2>Summary</h2>
    <p>View detailed results in MLflow UI: <a href="http://localhost:5000">MLflow Dashboard</a></p>

    <h2>Artifacts</h2>
    <ul>
        <li><a href="mlruns/{experiment.experiment_id}/{run_id}/artifacts/comparison/model_comparison.csv">Model Comparison CSV</a></li>
        <li><a href="mlruns/{experiment.experiment_id}/{run_id}/artifacts/comparison/recommendation.md">Recommendation</a></li>
    </ul>
</body>
</html>
"""

        with open("evaluation_report.html", 'w') as f:
            f.write(html)

        print(f"\n📊 Report generated: evaluation_report.html")
        print(f"📊 MLflow UI: http://localhost:5000")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLflowModelEvaluationPipeline("model-evaluation")

    # Run evaluation
    results = pipeline.run_complete_evaluation(
        expected_requests_per_day=5000  # Adjust based on expected traffic
    )

    # Generate report
    pipeline.generate_report()

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review MLflow dashboard: http://localhost:5000")
    print("2. Read recommendation.md for decision rationale")
    print("3. Check model_comparison.csv for detailed metrics")
