from src.llm.client import LLMClient, LLMConfig
import json


class QualityValidator:
    """Evaluate test case quality"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate_relevance(self, user_story: str, test_cases: list) -> dict:
        """Use LLM-as-judge to score relevance"""

        judge_prompt = f"""You are evaluating test cases for quality.

User Story:
{user_story}

Generated Test Cases:
{json.dumps(test_cases, indent=2)}

Rate these test cases on:
1. Relevance to user story (0-10)
2. Coverage of scenarios (0-10)
3. Clarity of steps (0-10)

Respond with ONLY this JSON format:
{{
  "relevance_score": <0-10>,
  "coverage_score": <0-10>,
  "clarity_score": <0-10>,
  "reasoning": "<brief explanation>"
}}
"""

        result = self.llm_client.generate(judge_prompt)

        try:
            # Parse response
            text = result['text'].strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]

            scores = json.loads(text)

            # Normalize to 0-1
            return {
                "relevance": scores.get('relevance_score', 0) / 10,
                "coverage": scores.get('coverage_score', 0) / 10,
                "clarity": scores.get('clarity_score', 0) / 10,
                "overall": (
                    scores.get('relevance_score', 0) +
                    scores.get('coverage_score', 0) +
                    scores.get('clarity_score', 0)
                ) / 30,
                "reasoning": scores.get('reasoning', ''),
                "passed": (scores.get('relevance_score', 0) / 10) >= 0.7
            }
        except:
            return {
                "relevance": 0.0,
                "coverage": 0.0,
                "clarity": 0.0,
                "overall": 0.0,
                "reasoning": "Failed to parse judge response",
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
