from fastapi import Request
from src.application.shared.models.custom_response import CustomResponse


def generate_test_controller(request: Request) -> CustomResponse:
    try:
        # return "asdf"
        # raise Exception("error mano")

        return CustomResponse.created(
            data={"user_story": "this is a user story"},
            msg="User created"
        )

    except Exception as e:
        print(f"="*60)
        print(e)
        print(e.__str__)
        print(f"="*60)

        return CustomResponse.error(
            msg="error amno",
        )
