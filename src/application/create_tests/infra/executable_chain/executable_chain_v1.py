import time
import json
from typing import Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse
from src.application.create_tests.infra.executable_chain.test_case_structure import TestCaseStructure, TestCasesResponse


def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


class ExecutableChainV1(ExecutableChain):
    """Executable chain implementation with RAG support.

    Provides a RAG (Retrieval-Augmented Generation) pattern implementation
    that executes a prompt using a language model with JSON output parsing.
    """

    def __init__(self, prompt_emplate: PromptTemplate, retriever: VectorStoreRetriever, llm: Optional[any] = None):
        super().__init__(prompt_emplate, retriever, llm)

    def execute(self, prompt: str, max_retries: int = 3) -> ExecutableChainResponse:
        """Execute the RAG chain with the given prompt.

        Invokes the chain with the prompt input and measures execution latency.
        Validates test case structure and retries if validation fails.
        The response is guaranteed to be valid JSON parsed output with correct structure.

        Args:
            prompt: The user prompt/question to process
            max_retries: Maximum number of retries for structure validation (default: 2)

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
            print(f"="*60)
            print(f"[ExecutableChainV1] - ValueError: {str(e)}")
            print(f"="*60)
            raise
        except TypeError as e:
            print(f"="*60)
            print(
                f"[ExecutableChainV1] - TypeError (Invalid JSON response): {str(e)}")
            print(f"="*60)
            raise
        except Exception as e:
            print(f"="*60)
            print(f"[ExecutableChainV1] - Exception: {str(e)}")
            print(f"="*60)
            raise Exception(f"Failed to execute chain: {str(e)}")

    def _execute_with_validation(
        self,
        prompt: str,
        max_retries: int,
        attempt: int = 1
    ) -> ExecutableChainResponse:
        """
        Execute chain with test case structure validation and retry logic.

        Args:
            prompt: The prompt to execute
            max_retries: Maximum number of retry attempts
            attempt: Current attempt number

        Returns:
            ExecutableChainResponse with validated test case structure
        """
        # ─────────────────────────────────────
        # Create RAG chain with JSON output parser
        # ─────────────────────────────────────
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | self.retriever | format_docs,
                "question": RunnableLambda(lambda x: x["question"])
            }
            | self.prompt_emplate
            | self.llm
            | JsonOutputParser()
        )

        # ─────────────────────────────────────
        # Invoke the RAG chain with the prompt
        # ─────────────────────────────────────
        start_time = time.time()
        result = rag_chain.invoke({"question": prompt})
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
                    f"[ExecutableChainV1] - Structure validation failed on attempt {attempt}/{max_retries}")
                print(f"[ExecutableChainV1] - Validation errors: {e}")
                print(f"[ExecutableChainV1] - Re-invoking LLM...")
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
        )
