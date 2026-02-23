"""Robust JSON output parser for handling LLM responses with markdown formatting."""

import json
import re
from typing import Dict, Any

from langchain_core.output_parsers import BaseOutputParser


class RobustJsonOutputParser(BaseOutputParser):
    """JSON parser that handles markdown-wrapped JSON output from LLMs."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.

        Handles cases where LLM wraps JSON in markdown:
        ```json
        {...}
        ```

        Args:
            text: Raw text output from LLM that may contain JSON

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If the text cannot be parsed as valid JSON
        """
        # Strip markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            # Remove opening markdown block
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            # Remove closing markdown block
            text = re.sub(r'\n```\s*$', '', text)

        # Try to extract JSON object if there's extra text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        return json.loads(text)
