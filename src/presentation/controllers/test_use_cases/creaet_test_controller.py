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

logger = logging.getLogger(__name__)


async def creaet_test_controller(request: Request) -> CustomResponse:
    try:

        # ─────────────────────────────────────
        # Extract JSON body from request and validate it
        # ─────────────────────────────────────
        body = await request.json()
        generate_request = GenerateRequest(**body)

        # ─────────────────────────────────────
        # llm rag - init dependencies
        # ─────────────────────────────────────
#         llm = ModelsConfig().get_llama3_2_1b()
#
#         # Load FAISS vectorstore
#         try:
#             retriever = load_faiss_vectorstore()
#         except FileNotFoundError as e:
#             print(f"="*60)
#             print(
#                 f"[creaet_test_controller] - FileNotFoundError {str(e)} ")
#             print(f"="*60)
#             return CustomResponse.error(message=str(e))
#
#         # executable_chain_rag
#         executable_chain_rag = ExecutableChainRAG(
#             prompt_emplate=RAG_PROMPT,
#             retriever=retriever,
#             llm=llm,
#         )
#
#         # init the use case
#         create_tests_application = CreateTestsApplication(
#             executable_chain=executable_chain_rag
#         )
#
#         # run the use case
#         create_tests_application_responde = create_tests_application.run(
#             generate_request=generate_request
#         )
#
#         return create_tests_application_responde

        # ─────────────────────────────────────
        # llm prompting - dependecies
        # ─────────────────────────────────────
        llm = ModelsConfig().get_llama3_2_1b()
        executable_chain_prompting = ExecutableChainPrompting(
            prompt_emplate=IMPROVED_PROMPT_V1,
            llm=llm
        )

        # init use case
        create_tests_application = CreateTestsApplication(
            executable_chain=executable_chain_prompting
        )

        # run the use case
        create_tests_application_responde = create_tests_application.run(
            generate_request=generate_request
        )

        return create_tests_application_responde

    except ValidationError as e:
        return CustomResponse.error(
            message=f"Validation error: {e.errors()[0]}"
        )
    except Exception as e:
        print(f"={'='*60}")
        print(f"Controller error: {e}")
        print(f"{'='*60}")
        return CustomResponse.something_was_wrong()
