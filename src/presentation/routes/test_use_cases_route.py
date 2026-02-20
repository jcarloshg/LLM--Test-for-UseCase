from fastapi import APIRouter, Request

from src.presentation.controllers import generate_test_controller

test_use_cases = APIRouter(prefix="/test-use-cases")


@test_use_cases.post("/")
async def create_test(request: Request):
    response = await generate_test_controller(request)
    return response.to_json_response()
