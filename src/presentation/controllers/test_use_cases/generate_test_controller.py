from fastapi import Request
from pydantic import ValidationError
from src.application.generate_test.application.generate_test_cases_use_case import GenerateTestCasesUseCase
from src.application.generate_test.models.generate_test_cases_request import GenerateRequest
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

        # init use case
        generate_test_cases = GenerateTestCasesUseCase()
        tests_cases_response = generate_test_cases.run(
            generate_request=generate_request
        )

        # TODO: Implement actual test case generation logic
        return CustomResponse.created(
            data={
                "user_story": generate_request.user_story,
                "test_cases": tests_cases_response.model_dump()
            },
            msg="Test cases generated successfully"
        )

    except ValidationError as e:
        return CustomResponse.error(
            msg=f"Validation error: {e.errors()[0]}"
        )
    except Exception as e:
        print(f"={'='*60}")
        print(f"Controller error: {e}")
        print(f"{'='*60}")
        return CustomResponse.error(
            msg=f"Failed to generate test cases: {str(e)}"
        )
