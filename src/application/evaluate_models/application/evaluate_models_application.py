# evaluate_models_application.py
import json
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import mlflow

from src.application.create_tests.infra.executable_chain.executable_chain_v1 import ExecutableChainV1
from src.application.create_tests.infra.vectorstores.faiss_vectorstore import load_faiss_vectorstore
from src.application.create_tests.models import RAG_PROMPT
from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse
from src.application.evaluate_models.model.latency_tracker import LatencyTracker
from src.application.evaluate_models.model.model_configs import ModelConfig, ModelRegistry
from src.application.evaluate_models.model.quality_tracker import QualityTracker
from src.application.evaluate_models.model.cost_tracker import CostTracker
from src.application.evaluate_models.model.test_case import TestCase
from src.application.evaluate_models.model.test_dataset import EvaluationDataset

from src.application.evaluate_models.infra.mlflow_config import MLflowConfig


class EvaluateModelsApplication:
    """
    Complete evaluation application for comparing LLM models.

    Orchestrates evaluation of multiple models across quality, latency, and cost metrics,
    with comprehensive MLflow tracking for experiment management and comparison.
    """

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
        Run complete evaluation pipeline with MLflow tracking.

        Evaluates multiple models and logs all metrics (quality, latency, cost)
        to MLflow for comprehensive comparison and analysis.
        """

        # Start parent run
        with mlflow.start_run(
            run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M')}"
        ):
            # ─────────────────────────────────────
            # Log global evaluation parameters
            # ─────────────────────────────────────
            mlflow.log_params({
                "num_models": len(models_to_test),
                "test_dataset_size": len(test_dataset),
                "expected_requests_per_day": expected_requests_per_day,
                "evaluation_timestamp": datetime.now().isoformat()
            })

            all_results = []

            # ─────────────────────────────────────
            # Evaluate each model with nested runs
            # ─────────────────────────────────────
            for idx, model_config in enumerate(models_to_test, 1):
                print(f"\n{'='*60}")
                print(
                    f"[{idx}/{len(models_to_test)}] Evaluating: {model_config.name}")
                print(f"{'='*60}")

                result = self._evaluate_single_model(
                    model_config,
                    test_dataset,
                    expected_requests_per_day,
                    executable_chain=executable_chain
                )

                all_results.append(result)

            # ─────────────────────────────────────
            # Log comparison summary
            # ─────────────────────────────────────
            if all_results:
                # Extract quality scores for comparison
                quality_scores = [
                    (r["model_name"], r["quality_metrics"]["quality_score"])
                    for r in all_results
                ]

                # Find best model
                best_model = max(quality_scores, key=lambda x: x[1])
                mlflow.log_metric("best_model_quality_score",
                                  float(best_model[1]))
                mlflow.log_param("best_performing_model", best_model[0])

                # Log comparison summary as text artifact
                summary_text = self._generate_evaluation_summary(all_results)
                mlflow.log_text(
                    summary_text, artifact_file="evaluation_summary.txt")

                # Log all results as JSON artifact for detailed analysis
                results_artifact = {
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "num_models": len(all_results),
                    "best_model": best_model[0],
                    "best_quality_score": float(best_model[1]),
                    "models": all_results
                }
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(results_artifact, f, indent=2, default=str)
                    results_file = f.name
                mlflow.log_artifact(
                    results_file, artifact_path="evaluation_results")
                os.unlink(results_file)

                # Log model comparison metrics as JSON
                comparison_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model_comparison": []
                }
                for result in all_results:
                    latency = result["latency_metrics"]
                    comparison_data["model_comparison"].append({
                        "model_name": result["model_name"],
                        "quality_score": result["quality_metrics"]["quality_score"],
                        "avg_latency": float(latency.get("mean", 0)),
                        "p95_latency": float(latency.get("p95", 0)),
                        "cost_per_thousand": result["cost_metrics"]["cost_per_thousand_requests"],
                        "cost_efficiency": result["cost_metrics"]["cost_efficiency_score"]
                    })
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(comparison_data, f, indent=2)
                    comparison_file = f.name
                mlflow.log_artifact(
                    comparison_file, artifact_path="evaluation_results")
                os.unlink(comparison_file)

            return {
                "evaluation_status": "completed",
                "num_models_evaluated": len(models_to_test),
                "total_test_cases": len(test_dataset),
                "results": all_results
            }

    def _evaluate_single_model(
        self,
        model_config: ModelConfig,
        test_dataset: List[TestCase],
        expected_requests_per_day: int,
        executable_chain: ExecutableChain,
    ) -> Dict[str, Any]:
        """Evaluate single model with complete metrics"""

        with mlflow.start_run(run_name=model_config.name, nested=True):
            # ─────────────────────────────────────
            # Log model configuration parameters
            # ─────────────────────────────────────
            mlflow.log_params({
                "model_name": model_config.name,
                "model_provider": model_config.provider,
                "model_id": model_config.model_id,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
                "test_dataset_size": len(test_dataset)
            })

            # ─────────────────────────────────────
            # Execute test cases and collect responses
            # ─────────────────────────────────────
            responses: list[ExecutableChainResponse] = []

            print(f"Running {len(test_dataset)} test cases...")

            for i, test_case in enumerate(test_dataset, 1):
                user_story = test_case.user_story
                current_llm = model_config.llm
                executable_chain.update_llm(current_llm)
                response = executable_chain.execute(prompt=user_story)
                responses.append(response)

                if i % 5 == 0 or i == len(test_dataset):
                    print(f"  Progress: {i}/{len(test_dataset)} completed")

            # ─────────────────────────────────────
            # 1. Quality Metrics (Accuracy & Quality)
            # ─────────────────────────────────────
            quality = QualityTracker.calculate_quality_score(
                execution_responses=responses
            )
            print(f"="*60)
            print("Quality Metrics:")
            print(quality)
            print(f"="*60)

            # Log quality metrics to MLflow
            mlflow.log_metrics({
                "quality_score": float(quality.quality_score),
                "precondition_score": float(quality.precondition_score),
                "structure_score": float(quality.structure_score),
                "passing_rate": float(quality.passing_rate),
                "json_parsing_success_rate": float(quality.json_parsing_success_rate),
                "avg_quality_score": float(quality.avg_quality_score),
                "retry_rate": float(quality.retry_rate)
            })

            mlflow.log_params({
                "total_tests": quality.total_tests,
                "passing_tests": quality.passing_tests,
                "total_responses": quality.total_responses
            })

            # ─────────────────────────────────────
            # 2. Latency & Performance Metrics
            # ─────────────────────────────────────
            latencies = [res.latency for res in responses]
            latency_stats = LatencyTracker.calculate_latency_stats(
                latencies=latencies
            )
            print(f"="*60)
            print("Latency Metrics:")
            print(latency_stats)
            print(f"="*60)

            # Log latency metrics to MLflow
            mlflow.log_metrics({
                "latency_avg": float(latency_stats.mean),
                "latency_min": float(latency_stats.min),
                "latency_max": float(latency_stats.max),
                "latency_p50": float(latency_stats.p50),
                "latency_p95": float(latency_stats.p95),
                "latency_p99": float(latency_stats.p99),
                "latency_std": float(latency_stats.std_dev)
            })

            # ─────────────────────────────────────
            # 3. Cost & Infrastructure Efficiency
            # ─────────────────────────────────────
            cost_analysis = CostTracker.calculate_cost_analysis(
                execution_responses=responses,
                monthly_server_cost=100.0,
                max_requests_per_day=expected_requests_per_day
            )
            print(f"="*60)
            print("Cost Analysis:")
            print(cost_analysis)
            print(f"="*60)

            # Log cost metrics to MLflow
            mlflow.log_metrics({
                "cost_per_thousand_requests": float(cost_analysis.cost_per_thousand_requests),
                "cost_efficiency_score": float(cost_analysis.cost_efficiency_score),
                "resource_avg_latency": float(cost_analysis.resource_utilization.avg_latency),
                "resource_throughput_req_per_min": float(cost_analysis.resource_utilization.throughput_requests_per_min),
                "resource_memory_per_request_mb": float(cost_analysis.resource_utilization.estimated_memory_per_request_mb),
                "resource_cpu_usage_percent": float(cost_analysis.resource_utilization.estimated_cpu_usage_percent),
                "resource_total_execution_time": float(cost_analysis.resource_utilization.total_execution_time_seconds),
                "resource_concurrent_capacity": float(cost_analysis.resource_utilization.concurrent_capacity)
            })

            mlflow.log_params({
                "monthly_server_cost": cost_analysis.monthly_server_cost,
                "max_requests_per_day": cost_analysis.max_requests_per_day
            })

            # ─────────────────────────────────────
            # Log artifacts for comprehensive tracking
            # ─────────────────────────────────────

            # 1. Log quality metrics as JSON artifact
            quality_artifact = {
                "model": model_config.name,
                "quality_metrics": quality.model_dump(),
                "timestamp": datetime.now().isoformat()
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(quality_artifact, f, indent=2)
                quality_file = f.name
            mlflow.log_artifact(quality_file, artifact_path="metrics")
            os.unlink(quality_file)

            # 2. Log latency metrics as JSON artifact
            latency_artifact = {
                "model": model_config.name,
                "latency_metrics": latency_stats.model_dump(),
                "timestamp": datetime.now().isoformat()
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(latency_artifact, f, indent=2)
                latency_file = f.name
            mlflow.log_artifact(latency_file, artifact_path="metrics")
            os.unlink(latency_file)

            # 3. Log cost analysis as JSON artifact
            cost_artifact = {
                "model": model_config.name,
                "cost_metrics": cost_analysis.model_dump(),
                "recommendations": cost_analysis.recommendations,
                "timestamp": datetime.now().isoformat()
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(cost_artifact, f, indent=2)
                cost_file = f.name
            mlflow.log_artifact(cost_file, artifact_path="metrics")
            os.unlink(cost_file)

            # 4. Log optimization recommendations as text artifact
            if cost_analysis.recommendations:
                recommendations_text = "\n".join(
                    [f"{i+1}. {rec}" for i,
                        rec in enumerate(cost_analysis.recommendations)]
                )
                mlflow.log_text(recommendations_text,
                                artifact_file="recommendations.txt")

            # 5. Generate and log model evaluation report
            model_report = self._generate_model_report(
                model_config.name,
                quality,
                latency_stats,
                cost_analysis
            )
            mlflow.log_text(
                model_report, artifact_file="model_evaluation_report.txt")

            # ─────────────────────────────────────
            # Compile and return comprehensive results
            # ─────────────────────────────────────
            results = {
                "model_name": model_config.name,
                "quality_metrics": quality.model_dump(),
                "latency_metrics": latency_stats.model_dump(),
                "cost_metrics": cost_analysis.model_dump(),
                "total_test_cases": len(test_dataset),
                "total_responses_processed": len(responses)
            }

            # Log summary metrics for easy comparison
            mlflow.log_metric("overall_quality_score",
                              float(quality.quality_score))
            mlflow.log_metric("overall_latency_p95", float(latency_stats.p95))
            mlflow.log_metric("overall_cost_efficiency", float(
                cost_analysis.cost_efficiency_score))

            return results

    @staticmethod
    def _generate_evaluation_summary(results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive evaluation summary report.

        Args:
            results: List of evaluation results for each model

        Returns:
            Formatted summary text
        """
        summary = "=" * 80 + "\n"
        summary += "COMPREHENSIVE MODEL EVALUATION SUMMARY\n"
        summary += "=" * 80 + "\n\n"

        for result in results:
            model_name = result["model_name"]
            quality = result["quality_metrics"]
            latency = result["latency_metrics"]
            cost = result["cost_metrics"]

            summary += f"Model: {model_name}\n"
            summary += "-" * 80 + "\n"

            summary += "Quality Metrics:\n"
            summary += f"  • Overall Quality Score: {quality['quality_score']:.4f}\n"
            summary += f"  • Precondition Score: {quality['precondition_score']:.4f}\n"
            summary += f"  • Structure Score: {quality['structure_score']:.4f}\n"
            summary += f"  • JSON Parsing Success Rate: {quality['json_parsing_success_rate']:.2f}%\n"
            summary += f"  • Average Quality Score: {quality['avg_quality_score']:.2f}/10\n"
            summary += f"  • Passing Rate: {quality['passing_rate']:.2f}%\n"
            summary += f"  • Retry Rate: {quality['retry_rate']:.2f}%\n\n"

            summary += "Latency Metrics (seconds):\n"
            summary += f"  • Average: {latency.get('mean', 0):.4f}s\n"
            summary += f"  • Min: {latency.get('min', 0):.4f}s\n"
            summary += f"  • Max: {latency.get('max', 0):.4f}s\n"
            summary += f"  • P50 (Median): {latency.get('p50', 0):.4f}s\n"
            summary += f"  • P95: {latency.get('p95', 0):.4f}s\n"
            summary += f"  • P99: {latency.get('p99', 0):.4f}s\n\n"

            summary += "Cost & Efficiency:\n"
            summary += f"  • Cost per 1,000 Requests: ${cost['cost_per_thousand_requests']:.2f}\n"
            summary += f"  • Monthly Server Cost: ${cost['monthly_server_cost']:.2f}\n"
            summary += f"  • Cost Efficiency Score: {cost['cost_efficiency_score']:.4f}\n"
            summary += f"  • Throughput: {cost['resource_utilization']['throughput_requests_per_min']:.2f} req/min\n"
            summary += f"  • Memory per Request: {cost['resource_utilization']['estimated_memory_per_request_mb']:.2f} MB\n"
            summary += f"  • CPU Usage: {cost['resource_utilization']['estimated_cpu_usage_percent']:.2f}%\n"
            summary += f"  • Concurrent Capacity: {cost['resource_utilization']['concurrent_capacity']} requests\n\n"

            summary += "Recommendations:\n"
            for idx, rec in enumerate(cost['recommendations'], 1):
                summary += f"  {idx}. {rec}\n"

            summary += "\n" + ("=" * 80) + "\n\n"

        return summary

    @staticmethod
    def _generate_model_report(
        model_name: str,
        quality,
        latency_stats: Dict[str, Any],
        cost_analysis
    ) -> str:
        """
        Generate detailed evaluation report for a single model.

        Args:
            model_name: Name of the model
            quality: QualityTrackerResponse object
            latency_stats: Dictionary of latency metrics
            cost_analysis: CostTrackerResponse object

        Returns:
            Formatted report text
        """
        report = "=" * 100 + "\n"
        report += f"MODEL EVALUATION REPORT: {model_name}\n"
        report += f"Generated: {datetime.now().isoformat()}\n"
        report += "=" * 100 + "\n\n"

        # Quality Section
        report += "1. QUALITY METRICS\n"
        report += "-" * 100 + "\n"
        report += f"Overall Quality Score:        {quality.quality_score:.4f} (0.0-1.0)\n"
        report += f"Precondition Quality:         {quality.precondition_score:.4f} (0.0-1.0)\n"
        report += f"Structure Quality:            {quality.structure_score:.4f} (0.0-1.0)\n"
        report += f"Passing Test Rate:            {quality.passing_rate*100:.2f}%\n"
        report += f"JSON Parsing Success Rate:    {quality.json_parsing_success_rate:.2f}% (Target: >95%)\n"
        report += f"Average Quality Score:        {quality.avg_quality_score:.2f}/10.0\n"
        report += f"Retry Rate:                   {quality.retry_rate:.2f}%\n"
        report += f"Total Tests:                  {quality.total_tests}\n"
        report += f"Passing Tests:                {quality.passing_tests}\n"
        report += f"Total Responses:              {quality.total_responses}\n\n"

        # Latency Section
        report += "2. LATENCY & PERFORMANCE\n"
        report += "-" * 100 + "\n"
        report += f"Average Latency:              {latency_stats.mean:.4f}s\n"
        report += f"Minimum Latency:              {latency_stats.min:.4f}s\n"
        report += f"Maximum Latency:              {latency_stats.max:.4f}s\n"
        report += f"Median Latency (P50):         {latency_stats.p50:.4f}s\n"
        report += f"95th Percentile (P95):        {latency_stats.p95:.4f}s\n"
        report += f"99th Percentile (P99):        {latency_stats.p99:.4f}s\n"
        report += f"Standard Deviation:           {latency_stats.std_dev:.4f}s\n\n"

        # Cost & Efficiency Section
        report += "3. COST & INFRASTRUCTURE EFFICIENCY\n"
        report += "-" * 100 + "\n"
        report += f"Cost per 1,000 Requests:      ${cost_analysis.cost_per_thousand_requests:.2f}\n"
        report += f"Monthly Server Cost:          ${cost_analysis.monthly_server_cost:.2f}\n"
        report += f"Max Requests per Day:         {cost_analysis.max_requests_per_day:,}\n"
        report += f"Cost Efficiency Score:        {cost_analysis.cost_efficiency_score:.4f} (0.0-1.0)\n\n"

        # Resource Utilization Section
        report += "4. RESOURCE UTILIZATION\n"
        report += "-" * 100 + "\n"
        util = cost_analysis.resource_utilization
        report += f"Average Latency:              {util.avg_latency:.4f}s\n"
        report += f"Throughput:                   {util.throughput_requests_per_min:.2f} req/min\n"
        report += f"Memory per Request:           {util.estimated_memory_per_request_mb:.2f} MB\n"
        report += f"CPU Usage:                    {util.estimated_cpu_usage_percent:.2f}%\n"
        report += f"Total Execution Time:         {util.total_execution_time_seconds:.2f}s\n"
        report += f"Concurrent Capacity:          {util.concurrent_capacity} requests\n\n"

        # Recommendations Section
        report += "5. OPTIMIZATION RECOMMENDATIONS\n"
        report += "-" * 100 + "\n"
        if cost_analysis.recommendations:
            for idx, rec in enumerate(cost_analysis.recommendations, 1):
                report += f"{idx}. {rec}\n"
        else:
            report += "No optimization recommendations at this time.\n"

        report += "\n" + "=" * 100 + "\n"

        return report


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
        num_medium=0,
        num_hard=0,
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
