# src/llm/prompts.py
from jinja2 import Template
import json


class PromptBuilder:
    def __init__(self):
        self.system_prompt = """You are an expert QA engineer who creates comprehensive test cases from user stories.

Your task is to generate structured test cases in Given-When-Then format.

RULES:
1. Generate 3-6 test cases covering happy path, edge cases, and error scenarios
2. Each test case MUST have: id, title, priority, given, when, then
3. Priority must be one of: critical, high, medium, low
4. Be specific and actionable in each step
5. Cover positive AND negative scenarios

OUTPUT FORMAT - Respond with ONLY valid JSON (no markdown, no explanations):
{
  "test_cases": [
    {
      "id": "TC_001",
      "title": "Brief descriptive title",
      "priority": "high",
      "given": "Preconditions",
      "when": "Action taken",
      "then": "Expected result"
    }
  ]
}"""

        self.user_template = Template("""
User Story:
{{ user_story }}

{% if examples %}
Examples of good test cases:

{% for example in examples %}
User Story: {{ example.user_story }}
Test Cases Generated:
{{ example.test_cases | tojson(indent=2) }}

{% endfor %}
{% endif %}

Now generate test cases for the user story above. Remember: output ONLY valid JSON.
""")

    def build(self, user_story: str, include_examples: bool = True) -> dict:
        """Build complete prompt"""

        examples = []
        if include_examples:
            examples = self._load_examples()[:2]  # Use 2 examples

        user_prompt = self.user_template.render(
            user_story=user_story,
            examples=examples
        )

        return {
            "system": self.system_prompt,
            "user": user_prompt
        }

    def _load_examples(self):
        """Load few-shot examples"""
        try:
            with open('data/examples/user_stories.json') as f:
                data = json.load(f)
                return data['examples']
        except:
            return []
