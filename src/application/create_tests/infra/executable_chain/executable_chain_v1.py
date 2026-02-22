import time
import json
import re
import asyncio
from typing import Optional, Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse
from src.application.create_tests.infra.executable_chain.test_case_structure import TestCaseStructure, TestCasesResponse
from src.application.create_tests.infra.executable_chain.rag_cache import RAGCache


def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


class RobustJsonOutputParser(BaseOutputParser):
    """JSON parser that handles markdown-wrapped JSON output from LLMs."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks.

        Handles cases where LLM wraps JSON in markdown:
        ```json
        {...}
        ```
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


class ExecutableChainV1(ExecutableChain):
    """Executable chain implementation with RAG support.

    Provides a RAG (Retrieval-Augmented Generation) pattern implementation
    that executes a prompt using a language model with JSON output parsing.
    """

    def __init__(self, prompt_emplate: PromptTemplate, retriever: VectorStoreRetriever, llm: Optional[any] = None):
        super().__init__(prompt_emplate, llm)
        self.retriever = retriever
        self._rag_cache = RAGCache(max_cache_size=100)

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

    def _cached_retrieve(self, question: str) -> str:
        """Retrieve context with caching.

        Args:
            question: The user question/prompt

        Returns:
            Formatted context string
        """
        # Try to get from cache first
        cached_docs = self._rag_cache.get(question)
        if cached_docs is not None:
            return format_docs(cached_docs)

        # If not cached, retrieve and cache
        docs = self.retriever.invoke(question)
        self._rag_cache.set(question, docs)
        return format_docs(docs)

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
        # Create RAG chain with caching and robust JSON output parser
        # ─────────────────────────────────────
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: self._cached_retrieve(x["question"])),
                "question": RunnableLambda(lambda x: x["question"])
            }
            | self.prompt_emplate
            | self.llm
            | RobustJsonOutputParser()
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
        # Log cache statistics
        # ─────────────────────────────────────
        cache_stats = self._rag_cache.get_stats()
        print(f"[ExecutableChainV1] - RAG Cache Stats: {cache_stats}")

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

    def get_cache_stats(self) -> dict:
        """Get RAG cache statistics.

        Returns:
            Dict with cache_size, max_size, and total_accesses
        """
        return self._rag_cache.get_stats()

    def clear_cache(self) -> None:
        """Clear RAG cache."""
        self._rag_cache.clear()
        print("[ExecutableChainV1] - RAG cache cleared")

    async def execute_async(
        self, prompts: List[str], max_concurrent: int = 3, max_retries: int = 3
    ) -> List[ExecutableChainResponse]:
        """Execute multiple prompts concurrently with rate limiting.

        Args:
            prompts: List of prompts to execute
            max_concurrent: Maximum concurrent requests (default: 3)
            max_retries: Maximum retries per prompt (default: 3)

        Returns:
            List of ExecutableChainResponse objects in the same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(prompt: str) -> ExecutableChainResponse:
            async with semaphore:
                return await self._execute_async_internal(prompt, max_retries)

        # Execute all prompts concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(prompt) for prompt in prompts],
            return_exceptions=False
        )

        return results

    async def _execute_async_internal(
        self, prompt: str, max_retries: int
    ) -> ExecutableChainResponse:
        """Internal async execution wrapper.

        Args:
            prompt: The prompt to execute
            max_retries: Maximum retries on validation failure

        Returns:
            ExecutableChainResponse with results and metrics
        """
        loop = asyncio.get_event_loop()

        # Run synchronous execute in thread pool to avoid blocking
        return await loop.run_in_executor(
            None, lambda: self.execute(prompt, max_retries)
        )
