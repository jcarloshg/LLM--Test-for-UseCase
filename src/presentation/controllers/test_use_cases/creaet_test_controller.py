from fastapi import Request
from pydantic import ValidationError

from src.application.create_tests.models.generate_test_cases_request import GenerateRequest
from src.application.shared.models.custom_response import CustomResponse


async def creaet_test_controller(request: Request) -> CustomResponse:
    try:

        # ─────────────────────────────────────
        # Extract JSON body from request and validate it
        # ─────────────────────────────────────
        body = await request.json()
        generate_request = GenerateRequest(**body)

        return CustomResponse.created(
            message="jej"
            data="hols"
        )

    except ValidationError as e:
        return CustomResponse.error(
            message=f"Validation error: {e.errors()[0]}"
        )
    except Exception as e:
        print(f"={'='*60}")
        print(f"Controller error: {e}")
        print(f"{'='*60}")
        return CustomResponse.something_was_wrong()
