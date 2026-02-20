from jinja2 import Template
import json
import random
from typing import List, Dict, Any

from src.application.generate_test.models.prompt_builder import PromptBuilder


class PromptBuilderCla(PromptBuilder):
    """Comprehensive prompt builder for test case generation with detailed structure.

    Generates test cases with extended fields including type, tags, gherkin format,
    quality score, and detailed step-by-step instructions. This implementation follows
    a more comprehensive test case structure suitable for complex testing scenarios.
    """

    def __init__(self):
        """Initialize the prompt builder with comprehensive system prompt and templates."""
        self.system_prompt = """
You are an expert QA Engineer specializing in test case design for Agile user stories.

Your Role:
Given a user story, create comprehensive test cases covering positive, negative, and edge case scenarios.

Test Case Structure:
Each test case must include the following fields:

| Field             | Description                | Format/Values                                      |
| ----------------- | -------------------------- | -------------------------------------------------- |
| `id`              | Unique identifier          | `TC-XXX` (e.g., TC-001, TC-002)                    |
| `title`           | Clear, descriptive title   | String                                             |
| `priority`        | Test importance            | `critical` / `high` / `medium` / `low`             |
| `type`            | Test category              | `positive` / `negative` / `edge_case` / `boundary` |
| `preconditions`   | Required setup before test | Array of strings                                   |
| `steps`           | Execution steps            | Array of numbered steps                            |
| `gherkin`         | BDD format                 | Object with `given`, `when`, `then`                |
| `expected_result` | What should happen         | String                                             |
| `quality_score`   | Test coverage quality      | `1-10` (10 = comprehensive)                        |
| `tags`            | Categorization labels      | Array of strings                                   |

RULES:
1. Generate 3-8 test cases covering happy path, edge cases, and error scenarios
2. Ensure balanced coverage across positive, negative, and edge case types
3. Each test case MUST have all required fields
4. Priority distribution: 1-2 critical, 2-3 high, 1-2 medium, 0-1 low
5. Quality score should reflect test comprehensiveness (6-9 range typical)
6. Be specific and actionable in each step
7. Tags should categorize by feature/functionality

OUTPUT FORMAT - Respond with ONLY valid JSON (no markdown, no explanations):
{
  "test_cases": [
    {
      "id": "TC-001",
      "title": "Brief descriptive title",
      "priority": "high",
      "type": "positive",
      "preconditions": ["Setup requirement 1", "Setup requirement 2"],
      "steps": ["Step 1", "Step 2", "Step 3"],
      "gherkin": {
        "given": "Initial state",
        "when": "Action taken",
        "then": "Expected outcome"
      },
      "expected_result": "Detailed expected result",
      "quality_score": 8,
      "tags": ["tag1", "tag2"]
    }
  ]
}"""

        self.user_template = Template("""
**ID:** {{ user_story_id | default("US-000") }}
**Title:** {{ user_story_title | default("Untitled Story") }}

**Description:**
{{ user_story }}

{% if acceptance_criteria %}
**Acceptance Criteria:**
{% for criterion in acceptance_criteria %}
- {{ criterion }}
{% endfor %}
{% endif %}

---
## CONTEXT
---

{% if app_type %}
**Application Type:** {{ app_type }}
{% endif %}

{% if test_focus %}
**Test Focus Areas:** {{ test_focus | join(", ") }}
{% endif %}

{% if min_test_cases %}
**Minimum Test Cases Required:** {{ min_test_cases }}
{% endif %}

{% if max_test_cases %}
**Maximum Test Cases Allowed:** {{ max_test_cases }}
{% endif %}

{% if priority_focus %}
**Priority Focus:** {{ priority_focus }}
{% endif %}

{% if specific_scenarios %}
---
## SPECIFIC SCENARIOS TO COVER
---
{% for scenario in specific_scenarios %}
- {{ scenario }}
{% endfor %}
{% endif %}

{% if examples %}
---
## REFERENCE EXAMPLES
---

The following are examples of well-structured test cases for similar user stories:

{% for example in examples %}
### Example {{ loop.index }}

**User Story:** {{ example.user_story }}

**Test Cases:**
```json
{{ example.test_cases | tojson(indent=2) }}
```

{% if example.quality_notes %}
**Quality Notes:** {{ example.quality_notes }}
{% endif %}

---
{% endfor %}
{% endif %}

---
## GENERATION INSTRUCTIONS
---

Generate test cases for the user story above.

**Requirements:**
1. Output ONLY valid JSON — no markdown code fences, no explanations
2. Include all required fields for each test case
3. Ensure coverage: positive ({{ positive_min | default(2) }}+), negative ({{ negative_min | default(2) }}+), edge cases ({{ edge_min | default(1) }}+)
4. Start test case IDs from: {{ start_id | default("TC-001") }}

**Output:**
""")

    def build(self, user_story: str, include_examples: bool = True) -> Dict[str, str]:
        """Build complete prompt with system and user components.

        Args:
            user_story: The user story to generate prompts for
            include_examples: Whether to include few-shot examples (default: True)

        Returns:
            Dict with keys:
                - "system": System prompt to guide LLM behavior
                - "user": User prompt with the actual request
        """
        examples = []
        if include_examples:
            all_examples = self._load_examples()
            examples = random.sample(all_examples, k=min(3, len(all_examples)))

        user_prompt = self.user_template.render(
            user_story=user_story,
            examples=examples
        )

        return {
            "system": self.system_prompt,
            "user": user_prompt
        }

    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples for prompt context.

        Attempts to load examples from data/examples/hundred_user_stories_with_test_cases.json.
        Returns empty list if file not found or invalid.

        Returns:
            List of example dictionaries containing sample inputs and outputs
        """
        try:
            with open('data/examples/hundred_user_stories_with_test_cases.json', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('examples', [])
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, KeyError):
            return []
