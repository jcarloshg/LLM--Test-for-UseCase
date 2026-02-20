from jinja2 import Template
import json
from typing import List, Dict, Any

from src.application.generate_test.models.prompt_builder import IPromptBuilder


class PromptBuilder(IPromptBuilder):
    """Concrete implementation of prompt builder for test case generation.

    Builds system and user prompts for generating test cases from user stories.
    Uses Jinja2 templates for flexible prompt construction with optional examples.
    """

    def __init__(self):
        """Initialize the prompt builder with system prompt and templates."""
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
            examples = self._load_examples()[:2]  # Use 2 examples

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

        Attempts to load examples from data/examples/user_stories.json.
        Returns empty list if file not found or invalid.

        Returns:
            List of example dictionaries containing sample inputs and outputs
        """
        try:
            with open('data/examples/user_stories.json') as f:
                data = json.load(f)
                return data.get('examples', [])
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, KeyError):
            return []
        except Exception:
            return []
