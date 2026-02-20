from src.application.shared.models.custom_response import CustomResponse
from src.application.generate_test.models.generate_test_cases_request import GenerateRequest


class GenerateTestCasesUseCase():
    def __init__(self):
        print(f"="*60)
        print(f"[GenerateTestCasesUseCase] init")
        print(f"="*60)

    def run(self, generate_request: GenerateRequest) -> CustomResponse:
        print(f"="*60)
        print(f"[GenerateTestCasesUseCase] run")
        print(f"="*60)

        return CustomResponse.created(
            data=generate_request.model_dump(),
            msg="Test cases generated from use case"
        )
