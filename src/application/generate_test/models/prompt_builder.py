"""
asfasd
"""

# ─────────────────────────────────────
# TODO delete this
# ─────────────────────────────────────

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Defines the interface for building prompts for LLM interactions.
    Subclasses must implement methods to construct system and user prompts.
    """

    @abstractmethod
    def build(self, user_story: str, include_examples: bool = True, context: str = "") -> Dict[str, str]:
        """Build complete prompt with system and user components.

        Args:
            user_story: The user story or input to create prompts for
            include_examples: Whether to include few-shot examples (default: True)
            context: Optional context or documentation to include in the prompt (default: "")

        Returns:
            Dict with keys:
                - "system": System prompt to guide LLM behavior
                - "user": User prompt with the actual request

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples for prompt context.

        Returns:
            List of example dictionaries containing sample inputs and outputs

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    def validate_prompt(self, prompt: Dict[str, str]) -> bool:
        """Validate that prompt has required structure.

        Args:
            prompt: Prompt dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        required_keys = {"system", "user"}
        return all(key in prompt and isinstance(prompt[key], str) for key in required_keys)
