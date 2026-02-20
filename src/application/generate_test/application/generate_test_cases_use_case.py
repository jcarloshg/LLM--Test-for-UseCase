from src.application.generate_test.models.generate_test_cases_request import GenerateRequest
from src.application.generate_test.models.prompt_builder import IPromptBuilder
from src.application.shared.models.custom_response import CustomResponse


class GenerateTestCasesUseCase():
    def __init__(self, prompt_builder: IPromptBuilder):
        self.prompt_builder = prompt_builder

    def run(self, generate_request: GenerateRequest) -> CustomResponse:
        try:
            # create prompt
            prompts = self.prompt_builder.build(generate_request.user_story)
            print(prompts)

            return CustomResponse.created(
                data={
                    "generate_request": generate_request.model_dump(),
                    "prompts": prompts
                },
                msg="Test cases generated from use case",
            )
        except Exception as e:
            return CustomResponse.error(
                msg=f"Failed to generate test cases: {str(e)}"
            )
