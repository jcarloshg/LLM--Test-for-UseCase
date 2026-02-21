
from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.generate_test_cases_request import GenerateRequest
from src.application.shared.models.custom_response import CustomResponse


class CreateTestsApplication:

    def __init__(self, executable_chain: ExecutableChain):
        self.executable_chain = executable_chain

    def run(self, generate_request: GenerateRequest) -> CustomResponse:
        try:

            executable_chain_response = self.executable_chain.execute(
                prompt=generate_request.user_story
            )

            print("\nexecutable_chain_response")
            print(executable_chain_response)

        except Exception as e:
            print("="*60)
            print(f"[CreateTestsApplication] - Exception {str(e)}")
            print("="*60)
            return CustomResponse.something_was_wrong()
