from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


class ExecutableChain(ABC):

    def __init__(self, prompt_emplate: PromptTemplate, retriever: VectorStoreRetriever, llm: Optional[any] = None):
        self.prompt_emplate = prompt_emplate
        self.llm: any = llm
        self.retriever: VectorStoreRetriever = retriever

    def update_llm(self, llm: any) -> None:
        """Update the LLM instance"""
        self.llm = llm

    @abstractmethod
    def execute(self, prompt: str) -> ExecutableChainResponse:
        """
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
                "question": RunnableLambda(lambda x: x["question"])
            }
            | RAG_PROMPT
            | llm
            | JsonOutputParser()
        )
        """
        pass
