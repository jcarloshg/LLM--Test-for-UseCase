# tests/test_prompts.py
from src.llm.client import LLMClient, LLMConfig
from src.llm.prompts import PromptBuilder
import json


def test_prompt_quality():
    """Test prompt on sample user stories"""

    client = LLMClient(LLMConfig())
    builder = PromptBuilder()

    test_stories = [
        "As a user, I want to reset my password so that I can regain access",
        "As an admin, I want to export user data so that I can analyze trends",
        "As a customer, I want to track my order so that I know when it arrives"
    ]

    results = []

    for story in test_stories:
        print(f"\n{'='*60}")
        print(f"Testing: {story}")
        print(f"{'='*60}")

        prompts = builder.build(story)
        response = client.generate(
            prompts['user'],
            prompts['system']
        )

        print(f"\nLatency: {response['latency']:.2f}s")
        print(f"Tokens: {response['tokens']}")

        # Try to parse JSON
        try:
            output_text = response['text'].strip()

            # Extract JSON if wrapped in markdown
            if '```json' in output_text:
                output_text = output_text.split('```json')[1].split('```')[0]
            elif '```' in output_text:
                output_text = output_text.split('```')[1].split('```')[0]

            test_cases = json.loads(output_text)

            print(f"✅ Valid JSON")
            print(
                f"Test cases generated: {len(test_cases.get('test_cases', []))}")

            # Display first test case
            if test_cases.get('test_cases'):
                tc = test_cases['test_cases'][0]
                print(f"\nSample test case:")
                print(f"  Title: {tc.get('title')}")
                print(f"  Priority: {tc.get('priority')}")
                print(f"  Given: {tc.get('given')}")

            results.append({
                "story": story,
                "success": True,
                "count": len(test_cases.get('test_cases', []))
            })

        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw output: {response['text'][:200]}")
            results.append({
                "story": story,
                "success": False,
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r.get('success'))
    print(f"Success rate: {successful}/{len(results)}")

    return results


if __name__ == "__main__":
    test_prompt_quality()
