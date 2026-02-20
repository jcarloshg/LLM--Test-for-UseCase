from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request

from src.presentation.controllers import generate_test_controller
from src.application.shared.models.custom_response import CustomResponse

test_use_cases = APIRouter(prefix="/test-use-cases")


@test_use_cases.post("/")
async def create_test(request: Request):
    try:
        response = await generate_test_controller(request)

        # Debug: print the response
        print(f"="*60)
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        print(f"="*60)

        # Use model_dump() directly for safe serialization
        return JSONResponse(
            content=response.model_dump(),
            status_code=response.status_code
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"="*60)
        print(f"Route error: {e}")
        print(f"Error type: {type(e)}")
        print(f"="*60)

        error_response = CustomResponse.error(message=str(e))
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=error_response.status_code
        )
