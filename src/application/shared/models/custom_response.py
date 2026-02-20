from pydantic import BaseModel
from typing import Any, Optional
from fastapi.responses import JSONResponse


class CustomResponse(BaseModel):
    message: str = ""
    is_success: bool = True
    data: Optional[Any] = None
    status_code: int = 200

    @staticmethod
    def success(msg: str = "Success", success: bool = True, data: Optional[Any] = None):
        return CustomResponse(
            message=msg,
            is_success=success,
            data=data
        )

    @staticmethod
    def created(msg: str = "Created", data: Optional[Any] = None):
        return CustomResponse(
            message=msg,
            is_success=True,
            data=data,
            status_code=201
        )

    @staticmethod
    def error(msg: str = "Error"):
        return CustomResponse(
            message=msg,
            is_success=False,
            data=None,
            status_code=500
        )

    def get_response(self):
        response = {
            "message": self.message,
            "is_success": self.is_success,
            "data": self.data
        }
        return response

    def get_JSONResponse(self):
        # Serialize data if it's a Pydantic model
        data = self.data
        if isinstance(self.data, BaseModel):
            data = self.data.model_dump()

        response = {
            "message": self.message,
            "is_success": self.is_success,
            "data": data
        }
        return JSONResponse(
            content=response,
            status_code=self.status_code
        )
