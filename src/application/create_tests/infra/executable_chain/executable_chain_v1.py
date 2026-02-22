
import time

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

    def __init__(self, prompt_emplate: PromptTemplate, llm: any, retriever: VectorStoreRetriever):
        self.prompt_emplate = prompt_emplate
        self.llm: any = llm
        self.retriever: VectorStoreRetriever = retriever

    def execute(self, prompt: str) -> ExecutableChainResponse:
        """Execute the RAG chain with the given prompt.

        Invokes the chain with the prompt input and measures execution latency.

        Args:
            prompt: The user prompt/question to process

        Returns:
            ExecutableChainResponse: Response containing generated JSON, latency, tokens, model, and provider

        Raises:
            Exception: If chain execution fails
        """
        try:

            # ─────────────────────────────────────
            # create chain
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
            # Return as ExecutableChainResponse
            # ─────────────────────────────────────
            return ExecutableChainResponse(
                result=result,
                latency=latency,
                tokens=0,
                model=getattr(self.llm, 'model_name', 'unknown'),
                provider=getattr(self.llm, '__class__', 'unknown').__name__,
            )

        except Exception as e:
            print(f"="*60)
            print(f"[ExecutableChainV1] - Exception: {str(e)}")
            print(f"="*60)
            raise Exception(f"Failed to execute chain: {str(e)}")
