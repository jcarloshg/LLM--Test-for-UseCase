import json

from src.application.generate_test.models.structure import StructureValidator
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

            # valid the json
            output_json = json.loads(response.text)
            output_json_validated = StructureValidator.validate(output_json)
            print(f"="*60)
            print(output_json_validated)
            print(f"="*60)
            if not output_json_validated["valid"]:
                raise Exception("invalid json format")

            test_cases = output_json_validated['test_cases']

            # ─────────────────────────────────────
            # TODO: add this a logging
            # ─────────────────────────────────────

            return CustomResponse.created(
                message="Test cases generated from use case",
                data=test_cases
            )

        except Exception as e:
            print("="*60)
            print(f"Error: {str(e)}")
            print("="*60)
            return CustomResponse.error(
                message=f"Failed to generate test cases: {str(e)}"
            )
