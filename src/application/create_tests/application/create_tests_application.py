from src.application.create_tests.models.executable_chain import ExecutableChain
from src.application.create_tests.models.generate_test_cases_request import GenerateRequest
from src.application.shared.models.custom_response import CustomResponse
from src.application.shared.infrastructure.logging_config import get_logger
from src.application.shared.infrastructure.logging_context import get_correlation_id

logger = get_logger(__name__)


class CreateTestsApplication:

    def __init__(self, executable_chain: ExecutableChain):
        self.executable_chain = executable_chain

    def run(self, generate_request: GenerateRequest) -> CustomResponse:
        try:
            logger.info(
                "Starting test case generation",
                extra={
                    "correlation_id": get_correlation_id(),
                    "user_story_length": len(generate_request.user_story),
                }
            )

            executable_chain_response = self.executable_chain.execute(
                prompt=generate_request.user_story
            )

            logger.info(
                "Test case generation completed successfully",
                extra={
                    "correlation_id": get_correlation_id(),
                    "test_cases_count": len(executable_chain_response.result.get("test_cases", [])),
                    "latency": executable_chain_response.latency,
                    "model": executable_chain_response.model,
                }
            )

            return CustomResponse.created(
                message="Test for use case",
                data=executable_chain_response.result,
            )

        except Exception as e:
            logger.error(
                "Test case generation failed",
                extra={
                    "correlation_id": get_correlation_id(),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True,
            )
            return CustomResponse.something_was_wrong()
