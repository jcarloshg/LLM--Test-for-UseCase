import logging
from fastapi import Request
from pydantic import ValidationError

from src.application.create_tests.application.create_tests_application import CreateTestsApplication
from src.application.create_tests.infra.executable_chain.executable_chain_prompting import ExecutableChainPrompting
from src.application.create_tests.infra.executable_chain.executable_chain_rag import ExecutableChainRAG
from src.application.create_tests.infra.vectorstores import load_faiss_vectorstore
from src.application.create_tests.models import RAG_PROMPT, IMPROVED_PROMPT_V1
from src.application.create_tests.models.generate_test_cases_request import GenerateRequest
from src.application.create_tests.models.models_config import ModelsConfig
from src.application.shared.models.custom_response import CustomResponse
from src.application.shared.infrastructure.logging_config import get_logger
from src.application.shared.infrastructure.logging_context import get_correlation_id

logger = get_logger(__name__)


async def creaet_test_controller(request: Request) -> CustomResponse:
    try:

        # ─────────────────────────────────────
        # Extract JSON body from request and validate it
        # ─────────────────────────────────────
        body = await request.json()
        generate_request = GenerateRequest(**body)

        # ─────────────────────────────────────
        # init dependencies
        # ─────────────────────────────────────
        #         # llm rag
        #         llm = ModelsConfig().get_llama3_2_1b()
        #
        #         # Load FAISS vectorstore
        #         try:
        #             retriever = load_faiss_vectorstore()
        #         except FileNotFoundError as e:
        #             print(f"="*60)
        #             print(f"[creaet_test_controller] - FileNotFoundError {str(e)} ")
        #             print(f"="*60)
        #             return CustomResponse.error(message=str(e))
        #
        #         # executable_chain_rag
        #         executable_chain_rag = ExecutableChainRAG(
        #             prompt_emplate=RAG_PROMPT,
        #             retriever=retriever,
        #             llm=llm,
        #         )

        # llm prompting
        llm = ModelsConfig().get_llama3_2_1b()
        executable_chain_prompting = ExecutableChainPrompting(
            prompt_emplate=IMPROVED_PROMPT_V1,
            llm=llm
        )

        # ─────────────────────────────────────
        # init use case
        # ─────────────────────────────────────
        create_tests_application = CreateTestsApplication(
            executable_chain=executable_chain_prompting
        )

        create_tests_application_responde = create_tests_application.run(
            generate_request=generate_request
        )

        return create_tests_application_responde

    except ValidationError as e:
        logger.warning(
            "Request validation error",
            extra={
                "correlation_id": get_correlation_id(),
                "error_type": "ValidationError",
                "error_message": str(e.errors()[0]),
            }
        )
        return CustomResponse.error(
            message=f"Validation error: {e.errors()[0]}"
        )
    except Exception as e:
        logger.error(
            "Controller error during test case generation",
            extra={
                "correlation_id": get_correlation_id(),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        return CustomResponse.something_was_wrong()
