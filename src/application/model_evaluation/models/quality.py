# from src.llm.client import LLMClient
import json

from src.application.generate_test.models.llm_client import LlmClient


class QualityValidator:
    """Evaluate test case quality"""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def evaluate_relevance(self, user_story: str, test_cases: list) -> dict:
        """Use LLM-as-judge to score relevance"""

        # Build test case summary to avoid JSON parsing issues
        test_summary = f"Number of test cases: {len(test_cases)}\n"
        for tc in test_cases[:5]:  # Show first 5 test cases
            test_summary += f"- {tc.get('title', 'Untitled')}: {tc.get('type', 'unknown')}\n"

        # Use a simpler prompt for better compatibility with smaller models
        judge_prompt = f"""Evaluate the quality of test cases for this user story.

USER STORY:
{user_story}

TEST CASES SUMMARY:
{test_summary}

Rate these test cases 0-10 on relevance, coverage, clarity, structure, and priority balance.

Output ONLY valid JSON:
{{"overall_score": 8, "scores": {{"relevance": 8, "coverage": 7, "clarity": 8, "structure": 9, "priority_balance": 8}}, "missing_scenarios": [], "strengths": ["good"], "improvements_needed": [], "recommendation": "pass", "reasoning": "summary"}}
"""

        result = self.llm_client.generate(judge_prompt)

        try:
            # Parse response - result is ILlmClientResponse, access text as attribute
            text = str(result.text).strip()

            # Remove markdown code blocks if present
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            # Try to find and extract JSON if wrapped in text
            if not text.startswith('{'):
                # Try to find JSON object in the text
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx+1]

            try:
                scores = json.loads(text)
            except json.JSONDecodeError as je:
                print(f"="*60)
                print(f"[QualityValidator] - JSON Parse Error: {str(je)}")
                print(f"Text (first 500 chars): {text[:500]}")
                print(f"="*60)

                # Try to repair common JSON issues
                try:
                    # Remove trailing commas before } or ]
                    import re
                    text_fixed = re.sub(r',(\s*[}\]])', r'\1', text)
                    # Remove newlines inside strings that shouldn't have them
                    text_fixed = text_fixed.replace('\n', ' ')
                    scores = json.loads(text_fixed)
                    print(f"[QualityValidator] - Successfully repaired JSON")
                except:
                    raise ValueError(f"Failed to parse JSON: {str(je)}")

            # Extract scores from comprehensive evaluation format
            score_obj = scores.get('scores', {})
            overall_score = scores.get('overall_score', 0) / 10

            return {
                "relevance": score_obj.get('relevance', 0) / 10,
                "coverage": score_obj.get('coverage', 0) / 10,
                "clarity": score_obj.get('clarity', 0) / 10,
                "structure": score_obj.get('structure', 0) / 10,
                "priority_balance": score_obj.get('priority_balance', 0) / 10,
                "overall": overall_score,
                "test_case_feedback": scores.get('test_case_feedback', []),
                "missing_scenarios": scores.get('missing_scenarios', []),
                "strengths": scores.get('strengths', []),
                "improvements_needed": scores.get('improvements_needed', []),
                "recommendation": scores.get('recommendation', 'needs_revision'),
                "reasoning": scores.get('reasoning', ''),
                "passed": overall_score >= 0.7
            }
        except Exception as e:
            print(f"="*60)
            print(f"[QualityValidator] - Exception {str(e)}")
            print(f"="*60)
            return {
                "relevance": 0.0,
                "coverage": 0.0,
                "clarity": 0.0,
                "structure": 0.0,
                "priority_balance": 0.0,
                "overall": 0.0,
                "test_case_feedback": [],
                "missing_scenarios": [],
                "strengths": [],
                "improvements_needed": [],
                "recommendation": "fail",
                "reasoning": f"Failed to parse judge response: {str(e)}",
                "passed": False
            }

    def evaluate_coverage(self, test_cases: list, min_count: int = 3) -> dict:
        """Check test case count and diversity"""

        count = len(test_cases)
        priorities = set(tc.get('priority') for tc in test_cases)

        # Check for positive and negative scenarios
        titles_lower = [tc.get('title', '').lower() for tc in test_cases]
        has_positive = any(
            'successful' in t or 'valid' in t for t in titles_lower)
        has_negative = any(
            'fail' in t or 'invalid' in t or 'error' in t for t in titles_lower)

        coverage_score = 0.0
        if count >= min_count:
            coverage_score += 0.4
        if len(priorities) >= 2:
            coverage_score += 0.3
        if has_positive:
            coverage_score += 0.15
        if has_negative:
            coverage_score += 0.15

        return {
            "count": count,
            "min_count_met": count >= min_count,
            "priority_diversity": len(priorities),
            "has_positive_cases": has_positive,
            "has_negative_cases": has_negative,
            "coverage_score": coverage_score,
            "passed": coverage_score >= 0.7
        }
