"""Executable chain implementation with direct prompting (no RAG)."""

import time
import logging
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse
from src.application.create_tests.infra.executable_chain.test_case_structure import (
    TestCaseStructure,
    TestCasesResponse,
)

logger = logging.getLogger(__name__)


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
            logger.error("ValueError: %s", str(e))
            raise
        except TypeError as e:
            logger.error("TypeError (Invalid JSON response): %s", str(e))
            raise
        except Exception as e:
            logger.error("Exception: %s", str(e))
            raise RuntimeError(f"Failed to execute chain: {str(e)}") from e

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
            | JsonOutputParser()
        )

        # ─────────────────────────────────────
        # Invoke the chain with the prompt
        # ─────────────────────────────────────
        start_time = time.time()
        # Get the first input variable name from the template
        input_var = self.prompt_emplate.input_variables[0]
        result = chain.invoke({input_var: prompt})
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
                logger.warning(
                    "Structure validation failed on attempt %d/%d",
                    attempt,
                    max_retries,
                )
                logger.debug("Validation errors: %s", e)
                logger.info("Re-invoking LLM...")
                return self._execute_with_validation(prompt, max_retries, attempt + 1)
            error_msg = (
                f"Test case structure validation failed after {max_retries} attempts. "
                f"Response must contain 'test_cases' key with list of test cases. "
                f"Each test case must have: "
                f"{', '.join(TestCaseStructure.model_fields.keys())}. "
                f"Got fields: {list(result.keys())}. "
                f"Validation errors: {str(e)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from e

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
