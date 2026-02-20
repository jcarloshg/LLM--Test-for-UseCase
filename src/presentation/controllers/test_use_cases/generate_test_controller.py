from fastapi import Request
from pydantic import ValidationError

# from src.application.generate_test.infrastructure.prompt_builder import PromptBuilder
from src.application.generate_test.application.generate_test_cases_use_case import GenerateTestCasesUseCase
from src.application.generate_test.infrastructure.anthropic.llm_client_anthropic import LLMClientAnthropic
from src.application.generate_test.infrastructure.ollama.llm_client_ollama import LLMClientOllama
from src.application.generate_test.infrastructure.prompt_builder_cla import PromptBuilderCla
from src.application.generate_test.models.generate_test_cases_request import GenerateRequest
from src.application.generate_test.models.llm_config import LLMConfig
from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
from src.application.shared.models.custom_response import CustomResponse


async def generate_test_controller(request: Request) -> CustomResponse:
    """Handle test case generation request.

    Args:
        request: FastAPI request object containing user story and configuration

    Returns:
        CustomResponse: Response with generated test cases or error message
    """
    try:
        # Extract JSON body from request and validate it
        body = await request.json()
        generate_request = GenerateRequest(**body)

        # Initialize dependencies
        # prompt_builder = PromptBuilder()
        prompt_builder = PromptBuilderCla()
        llm_config = LLMConfig(
            api_key=ENVIRONMENT_CONFIG.ANTHOPIC_KEY,
            model=ENVIRONMENT_CONFIG.ANTHOPIC_MODEL
        )
        # llm_client_ollama = LLMClientOllama(config=llm_config)
        llm_client_ollama = LLMClientAnthropic(llm_config)

        # Initialize && run use case
        generate_test_cases = GenerateTestCasesUseCase(
            prompt_builder=prompt_builder,
            llm_client=llm_client_ollama
        )
        tests_cases_response = generate_test_cases.run(
            generate_request=generate_request
        )

        # TODO: Implement actual test case generation logic
        return tests_cases_response

    except ValidationError as e:
        return CustomResponse.error(
            message=f"Validation error: {e.errors()[0]}"
        )
    except Exception as e:
        print(f"={'='*60}")
        print(f"Controller error: {e}")
        print(f"{'='*60}")
        return CustomResponse.error(
            message=f"Failed to generate test cases: {str(e)}"
        )
