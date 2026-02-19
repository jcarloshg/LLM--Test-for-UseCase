import json
from src.llm.client import LLMClient, LLMConfig
from src.llm.prompts import PromptBuilder
from src.validators.structure import StructureValidator
from src.validators.quality import QualityValidator
from src.mlflow_tracker import MLflowTracker


def run_evaluation():
    """Run evaluation on test dataset"""

    # Initialize
    client = LLMClient(LLMConfig())
    builder = PromptBuilder()
    struct_validator = StructureValidator()
    quality_validator = QualityValidator(client)
    tracker = MLflowTracker(experiment_name="evaluation")

    # Load test dataset
    with open('data/validation/test_dataset.json') as f:
        test_data = json.load(f)

    results = []

    print(f"\n{'='*60}")
    print("RUNNING EVALUATION")
    print(f"{'='*60}\n")

    for i, test_case in enumerate(test_data, 1):
        print(f"Test {i}/{len(test_data)}: {test_case['id']}")

        # Generate
        prompts = builder.build(test_case['user_story'])
        llm_result = client.generate(prompts['user'], prompts['system'])

        # Parse
        try:
            output_text = llm_result['text'].strip()
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            output_json = json.loads(output_text)
        except Exception as e:
            print(f"  ❌ Parse error: {e}")
            results.append({
                "id": test_case['id'],
                "passed": False,
                "error": "parse_error"
            })
            continue

        # Validate
        struct_val = struct_validator.validate(output_json)

        if not struct_val['valid']:
            print(f"  ❌ Structure invalid")
            results.append({
                "id": test_case['id'],
                "passed": False,
                "error": "structure_invalid"
            })
            continue

        # Quality check
        quality = quality_validator.evaluate_relevance(
            test_case['user_story'],
            struct_val['test_cases']
        )

        coverage = quality_validator.evaluate_coverage(
            struct_val['test_cases'],
            test_case.get('expected_test_count', 3)
        )

        passed = (
            struct_val['valid'] and
            quality['passed'] and
            coverage['passed']
        )

        # Log to MLflow
        tracker.log_generation(
            user_story=test_case['user_story'],
            test_cases=struct_val['test_cases'],
            structure_validation=struct_val,
            quality_metrics=quality,
            coverage_metrics=coverage,
            latency=llm_result['latency'],
            model_info={
                "model": llm_result['model'],
                "provider": llm_result['provider']
            }
        )

        print(f"  {'✅' if passed else '❌'} Pass: {passed}")
        print(f"     Quality: {quality['overall']:.2f}")
        print(f"     Coverage: {coverage['coverage_score']:.2f}")

        results.append({
            "id": test_case['id'],
            "passed": passed,
            "count": struct_val['count'],
            "quality": quality['overall'],
            "coverage": coverage['coverage_score'],
            "latency": llm_result['latency']
        })

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for r in results if r.get('passed'))
    avg_quality = sum(r.get('quality', 0) for r in results) / total
    avg_coverage = sum(r.get('coverage', 0) for r in results) / total
    avg_latency = sum(r.get('latency', 0) for r in results) / total

    print(f"Pass Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Avg Quality Score: {avg_quality:.2f}")
    print(f"Avg Coverage Score: {avg_coverage:.2f}")
    print(f"Avg Latency: {avg_latency:.2f}s")

    # Save results
    with open('reports/evaluation_results.json', 'w') as f:
        json.dump({
            "summary": {
                "total": total,
                "passed": passed,
                "pass_rate": passed/total,
                "avg_quality": avg_quality,
                "avg_coverage": avg_coverage,
                "avg_latency": avg_latency
            },
            "results": results
        }, f, indent=2)

    print(f"\n✅ Results saved to reports/evaluation_results.json")
    print(f"📊 View MLflow dashboard: http://localhost:5000\n")


if __name__ == "__main__":
    import os
    os.makedirs('reports', exist_ok=True)
    run_evaluation()
