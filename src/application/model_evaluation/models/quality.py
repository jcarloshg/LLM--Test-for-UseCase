# from src.llm.client import LLMClient
import json

from src.application.generate_test.models.llm_client import LlmClient


class QualityValidator:
    """Evaluate test case quality"""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def evaluate_relevance(self, user_story: str, test_cases: list) -> dict:
        """Use LLM-as-judge to score relevance using comprehensive evaluation rubric"""

        # Prepare variables for the prompt
        user_story_id = "US-AUTO"
        ac_section = ""  # Acceptance criteria not available
        test_cases_json = json.dumps(test_cases, indent=2)

        # Comprehensive evaluation prompt with detailed rubric
        judge_prompt = f"""You are a Senior QA Lead evaluating auto-generated test cases for quality and completeness.

---
## EVALUATION CONTEXT
---

### User Story

**ID:** {user_story_id}

{user_story}
{ac_section}

### Test Cases to Evaluate
```json
{test_cases_json}
```

---
## EVALUATION RUBRIC
---

Score each dimension from 0-10 using these criteria:

### RELEVANCE (Weight: 25%)
Do test cases directly address the user story?

| Score | Criteria |
|-------|----------|
| 9-10  | All tests directly relevant to user story requirements |
| 7-8   | Most tests relevant; 1-2 tangential cases |
| 5-6   | Partially relevant; some tests miss the point |
| 3-4   | Weak connection; many irrelevant tests |
| 0-2   | Tests don't address the user story |

### COVERAGE (Weight: 30%)
Are all important scenarios tested?

| Score | Criteria |
|-------|----------|
| 9-10  | Happy path + all edge cases + errors + boundaries |
| 7-8   | Good coverage; missing 1-2 scenarios |
| 5-6   | Basic coverage; missing edge cases |
| 3-4   | Minimal; only happy path OR only errors |
| 0-2   | Major scenarios missing |

### CLARITY (Weight: 20%)
Are test steps specific and actionable?

| Score | Criteria |
|-------|----------|
| 9-10  | Steps specific, unambiguous; any tester can execute |
| 7-8   | Clear steps; minor ambiguities |
| 5-6   | Understandable but vague in places |
| 3-4   | Confusing; requires interpretation |
| 0-2   | Cannot be executed as written |

### STRUCTURE (Weight: 15%)
Are test cases well-formed?

| Score | Criteria |
|-------|----------|
| 9-10  | All fields complete; proper Gherkin syntax |
| 7-8   | Minor field issues or inconsistencies |
| 5-6   | Some missing fields or formatting problems |
| 3-4   | Multiple structural issues |
| 0-2   | Malformed or incomplete test cases |

### PRIORITY BALANCE (Weight: 10%)
Are priorities correctly assigned?

| Score | Criteria |
|-------|----------|
| 9-10  | Priorities reflect actual importance; critical tests identified |
| 7-8   | Mostly correct; minor adjustments needed |
| 5-6   | Some mismatches |
| 3-4   | Poor priority assignment |
| 0-2   | Random or incorrect priorities |

---
## VERDICT CRITERIA
---

- **pass**: Overall ≥ 7.5, no critical gaps
- **needs_revision**: Overall 5.0-7.4, or fixable issues exist
- **fail**: Overall < 5.0, or fundamental problems

---
## OUTPUT INSTRUCTIONS
---

Return ONLY valid JSON. No markdown fences. No text before or after.

Start your response with the opening brace character.

Required JSON structure:

{{
  "overall_score": <number 0-10>,
  "scores": {{
    "relevance": <number 0-10>,
    "coverage": <number 0-10>,
    "clarity": <number 0-10>,
    "structure": <number 0-10>,
    "priority_balance": <number 0-10>
  }},
  "per_test_case": [
    {{
      "id": "<test case id>",
      "score": <number 0-10>,
      "issues": ["<specific issue>"],
      "suggestion": "<improvement>"
    }}
  ],
  "missing_scenarios": ["<scenario not covered>"],
  "strengths": ["<what's done well>"],
  "improvements_needed": ["<what to fix>"],
  "recommendation": "<pass | needs_revision | fail>",
  "reasoning": "<2-3 sentence summary>"
}}
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
                    import re
                    text_fixed = text

                    # 1. Replace all newlines with spaces to remove line breaks in strings
                    text_fixed = text_fixed.replace('\n', ' ')
                    text_fixed = text_fixed.replace('\r', ' ')

                    # 2. Remove trailing commas before } or ]
                    text_fixed = re.sub(r',(\s*[}\]])', r'\1', text_fixed)

                    # 3. Fix multiple spaces
                    text_fixed = re.sub(r'\s+', ' ', text_fixed)

                    # 4. Ensure proper spacing in JSON structure
                    text_fixed = re.sub(r'{\s*', '{ ', text_fixed)
                    text_fixed = re.sub(r'\s*}', ' }', text_fixed)
                    text_fixed = re.sub(r'\[\s*', '[ ', text_fixed)
                    text_fixed = re.sub(r'\s*]', ' ]', text_fixed)

                    # 5. Fix colon spacing
                    text_fixed = re.sub(r':\s*', ': ', text_fixed)

                    # 6. Fix comma spacing
                    text_fixed = re.sub(r',\s*', ', ', text_fixed)

                    # 7. Remove duplicate commas
                    text_fixed = re.sub(r',+', ',', text_fixed)

                    # 8. Clean up excessive spaces
                    text_fixed = re.sub(r'\s+', ' ', text_fixed)

                    scores = json.loads(text_fixed)
                    print(f"[QualityValidator] - Successfully repaired JSON")
                except Exception as repair_err:
                    print(f"[QualityValidator] - Repair failed: {str(repair_err)}")
                    print(f"[QualityValidator] - Using partial parse fallback")
                    # Last resort: try to extract just the main score fields
                    try:
                        overall_match = re.search(r'"overall_score"\s*:\s*(\d+(?:\.\d+)?)', text)
                        scores = {
                            "overall_score": float(overall_match.group(1)) if overall_match else 5,
                            "scores": {
                                "relevance": 5,
                                "coverage": 5,
                                "clarity": 5,
                                "structure": 5,
                                "priority_balance": 5
                            },
                            "per_test_case": [],
                            "missing_scenarios": [],
                            "strengths": ["evaluation attempted"],
                            "improvements_needed": ["retry evaluation"],
                            "recommendation": "needs_revision",
                            "reasoning": "JSON parsing failed, using fallback"
                        }
                        print(f"[QualityValidator] - Fallback parse succeeded")
                    except:
                        raise ValueError(f"Failed to parse JSON: {str(je)}")

            # Extract scores from comprehensive evaluation format
            # Scores from LLM are 0-10, normalize to 0-1
            score_obj = scores.get('scores', {})
            overall_score_raw = scores.get('overall_score', 0)
            overall_score = overall_score_raw / 10 if overall_score_raw else 0

            # Determine pass/fail based on recommendation or overall score
            recommendation = scores.get('recommendation', 'needs_revision')
            passed = recommendation == 'pass' or overall_score >= 0.75

            return {
                "relevance": score_obj.get('relevance', 0) / 10,
                "coverage": score_obj.get('coverage', 0) / 10,
                "clarity": score_obj.get('clarity', 0) / 10,
                "structure": score_obj.get('structure', 0) / 10,
                "priority_balance": score_obj.get('priority_balance', 0) / 10,
                "overall": overall_score,
                "per_test_case": scores.get('per_test_case', []),
                "missing_scenarios": scores.get('missing_scenarios', []),
                "strengths": scores.get('strengths', []),
                "improvements_needed": scores.get('improvements_needed', []),
                "recommendation": recommendation,
                "reasoning": scores.get('reasoning', ''),
                "passed": passed
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
                "per_test_case": [],
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
