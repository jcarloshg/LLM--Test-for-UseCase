from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.prompts import PromptTemplate

from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


class ExecutableChain(ABC):
    """Abstract base class for executable chains with RAG support.

    Provides a foundation for implementing chain execution with language models
    and retrieval-augmented generation patterns.
    """

    def __init__(self, prompt_emplate: PromptTemplate, llm: Optional[any] = None):
        """Initialize the executable chain.

        Args:
            prompt_emplate: The prompt template to use for chain execution
            llm: Optional language model instance (can be set later with update_llm)
        """
        self.prompt_emplate = prompt_emplate
        self.llm: any = llm
        self.retriever = None

    def update_llm(self, llm: any) -> None:
        """Update the LLM instance.

        Args:
            llm: Language model instance to use for chain execution
        """
        self.llm = llm

    def update_retriever(self, retriever) -> None:
        """Update the retriever instance.

        Args:
            retriever: VectorStoreRetriever instance for RAG operations
        """
        self.retriever = retriever

    @abstractmethod
    def execute(self, prompt: str, max_retries: int = 3) -> ExecutableChainResponse:
        """Execute the chain with the given prompt.

        Args:
            prompt: The input prompt to execute
            max_retries: Maximum number of retries for validation (default: 3)

        Returns:
            ExecutableChainResponse containing the chain execution result

        Raises:
            ValueError: If required components (LLM, retriever) are not configured
            Exception: If chain execution fails
        """
        pass
