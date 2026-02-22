"""Executable chain implementation with direct prompting (no RAG)."""

import time
import json
import re
from typing import Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from pydantic import ValidationError

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse
from src.application.create_tests.infra.executable_chain.test_case_structure import TestCaseStructure, TestCasesResponse

# ─────────────────────────────────────
# TODO: create a file called RobustJsonOutputParser
# TODO create a class for PromptTemplate to save metadata
# ─────────────────────────────────────


class RobustJsonOutputParser(BaseOutputParser):
    """JSON parser that handles markdown-wrapped JSON output from LLMs."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.

        Handles cases where LLM wraps JSON in markdown:
        ```json
        {...}
        ```
        """
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


class ExecutableChainPrompting(ExecutableChain):
    """Executable chain implementation with direct prompting.

    Provides a simple implementation that executes prompts directly without
    retrieval-augmented generation. Suitable for tasks that don't require context retrieval.
    """

    def __init__(self, prompt_emplate: PromptTemplate, llm: Optional[any] = None):
        """Initialize the prompting chain.

        Args:
            prompt_emplate: The prompt template to use for chain execution
            llm: Optional language model instance (can be set later with update_llm)
        """
        super().__init__(prompt_emplate, llm)

    def execute(self, prompt: str, max_retries: int = 3) -> ExecutableChainResponse:
        """Execute the prompting chain with the given prompt.

        Invokes the chain with the prompt input without retrieval context.
        Validates test case structure and retries if validation fails.
        The response is guaranteed to be valid JSON parsed output with correct structure.

        Args:
            prompt: The user prompt/question to process
            max_retries: Maximum number of retries for structure validation (default: 3)

        Returns:
            ExecutableChainResponse: Response containing generated JSON (Dict), latency, tokens, model, and provider

        Raises:
            ValueError: If LLM is not configured or structure validation fails
            Exception: If chain execution fails or JSON parsing fails
        """
        try:
            # ─────────────────────────────────────
            # Validate LLM is configured
            # ─────────────────────────────────────
            if self.llm is None:
                raise ValueError(
                    "LLM is not configured. Use update_llm() to set an LLM instance.")

            # ─────────────────────────────────────
            # Execute chain with retry logic
            # ─────────────────────────────────────
            return self._execute_with_validation(prompt, max_retries, attempt=1)

        except ValueError as e:
            print("=" * 60)
            print(f"[ExecutableChainPrompting] - ValueError: {str(e)}")
            print("=" * 60)
            raise
        except TypeError as e:
            print("=" * 60)
            print(
                f"[ExecutableChainPrompting] - TypeError (Invalid JSON response): {str(e)}")
            print("=" * 60)
            raise
        except Exception as e:
            print("=" * 60)
            print(f"[ExecutableChainPrompting] - Exception: {str(e)}")
            print("=" * 60)
            raise Exception(f"Failed to execute chain: {str(e)}")

    def _execute_with_validation(
        self,
        prompt: str,
        max_retries: int,
        attempt: int = 1
    ) -> ExecutableChainResponse:
        """Execute chain with test case structure validation and retry logic.

        Args:
            prompt: The prompt to execute
            max_retries: Maximum number of retry attempts
            attempt: Current attempt number

        Returns:
            ExecutableChainResponse with validated test case structure
        """
        # ─────────────────────────────────────
        # Create direct prompting chain (no RAG)
        # ─────────────────────────────────────
        chain = (
            self.prompt_emplate
            | self.llm
            | RobustJsonOutputParser()
        )

        # ─────────────────────────────────────
        # Invoke the chain with the prompt
        # ─────────────────────────────────────
        start_time = time.time()
        result = chain.invoke({"question": prompt})
        latency = time.time() - start_time

        # ─────────────────────────────────────
        # Validate JSON response is dictionary
        # ─────────────────────────────────────
        if not isinstance(result, dict):
            raise TypeError(
                f"Expected JSON dict response, got {type(result).__name__}. "
                f"Ensure the prompt template instructs the model to return valid JSON.")

        # ─────────────────────────────────────
        # Validate test case structure
        # ─────────────────────────────────────
        try:
            TestCasesResponse(**result)
        except ValidationError as e:
            if attempt < max_retries:
                print(
                    f"[ExecutableChainPrompting] - Structure validation failed on attempt {attempt}/{max_retries}")
                print(f"[ExecutableChainPrompting] - Validation errors: {e}")
                print(f"[ExecutableChainPrompting] - Re-invoking LLM...")
                return self._execute_with_validation(prompt, max_retries, attempt + 1)
            raise ValueError(
                f"Test case structure validation failed after {max_retries} attempts. "
                f"Response must contain 'test_cases' key with list of test cases. "
                f"Each test case must have: {', '.join(TestCaseStructure.model_fields.keys())}. "
                f"Got fields: {list(result.keys())}. "
                f"Validation errors: {str(e)}"
            )

        # ─────────────────────────────────────
        # Return as ExecutableChainResponse
        # ─────────────────────────────────────
        return ExecutableChainResponse(
            result=result,
            latency=latency,
            tokens=0,
            model=getattr(self.llm, 'model_name', 'unknown'),
            provider=getattr(self.llm, '__class__', 'unknown').__name__,
            attempt=attempt,
        )
