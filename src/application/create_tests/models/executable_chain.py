from abc import ABC, abstractmethod
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from src.application.create_tests.models.executable_chain_response import ExecutableChainResponse


class ExecutableChain(ABC):

    def __init__(self, prompt_emplate: PromptTemplate, llm: any, retriever: VectorStoreRetriever):
        self.prompt_emplate = prompt_emplate
        self.llm: any = llm
        self.retriever: VectorStoreRetriever = retriever

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
