from src.application.generate_test.models.generate_test_cases_request import GenerateRequest
from src.application.generate_test.models.llm_client import LlmClient
from src.application.generate_test.models.prompt_builder import IPromptBuilder
from src.application.shared.models.custom_response import CustomResponse


class GenerateTestCasesUseCase():
    def __init__(
        self,
        prompt_builder: IPromptBuilder,
        llm_client: LlmClient
    ):
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client

    def run(self, generate_request: GenerateRequest) -> CustomResponse:
        try:
            # create prompt
            prompts = self.prompt_builder.build(generate_request.user_story)

            # generate test cases using LLM
            response = self.llm_client.generate(
                prompt=prompts.get("user", ""),
                system_prompt=prompts.get("system", "")
            )

            return CustomResponse.created(
                message="Test cases generated from use case",
                data={
                    "generate_request": generate_request.model_dump(),
                    "prompts": prompts,
                    "generated_tests": response.model_dump()
                }
            )
        except Exception as e:
            print("="*60)
            print(f"Error: {str(e)}")
            print("="*60)
            return CustomResponse.error(
                message=f"Failed to generate test cases: {str(e)}"
            )
