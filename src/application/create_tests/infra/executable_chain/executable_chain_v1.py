import time
import json
from typing import Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda

from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


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

    def execute(self, prompt: str) -> ExecutableChainResponse:
        """Execute the RAG chain with the given prompt.

        Invokes the chain with the prompt input and measures execution latency.
        The response is guaranteed to be valid JSON parsed output.

        Args:
            prompt: The user prompt/question to process

        Returns:
            ExecutableChainResponse: Response containing generated JSON (Dict), latency, tokens, model, and provider

        Raises:
            ValueError: If LLM is not configured
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
            # Validate JSON response
            # ─────────────────────────────────────
            if not isinstance(result, dict):
                raise TypeError(
                    f"Expected JSON dict response, got {type(result).__name__}. "
                    f"Ensure the prompt template instructs the model to return valid JSON.")

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
