from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request

from src.presentation.controllers import generate_test_controller
from src.application.shared.models.custom_response import CustomResponse

test_use_cases = APIRouter(prefix="/test-use-cases")


@test_use_cases.post("/")
async def create_test(request: Request):
    try:
        response = generate_test_controller(request)
        return response.get_JSONResponse()
    except HTTPException:
        raise
    except Exception as e:
        error_response = CustomResponse.error(msg=str(e))
        print(f"="*60)
        print(error_response)
        print(f"="*60)
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=error_response.status_code
        )
