from pydantic import BaseModel, Field
from typing import Any, Optional, Dict
from fastapi.responses import JSONResponse


class CustomResponse(BaseModel):
    """Standard API response model for consistent response formatting.

    This model provides a uniform structure for all API responses, including
    success and error responses. It includes metadata about the response status
    and allows flexible data payloads.

    Attributes:
        message: Human-readable response message
        is_success: Boolean indicating if operation succeeded (True) or failed (False)
        data: Optional response payload containing the actual data or error details
        status_code: HTTP status code for the response (200, 201, 400, 500, etc.)
    """

    message: str = Field(
        default="",
        description="Human-readable response message"
    )
    is_success: bool = Field(
        default=True,
        description="Whether the operation was successful"
    )
    data: Optional[Any] = Field(
        default=None,
        description="Response payload (data or error details)"
    )
    status_code: int = Field(
        default=200,
        ge=100,
        le=599,
        description="HTTP status code"
    )

    class Config:
        """Pydantic configuration for response model."""
        str_strip_whitespace = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "message": "Operation successful",
                "is_success": True,
                "data": {"result": "example"},
                "status_code": 200
            }
        }

    # ─────────────────────────────────────
    # Factory methods organized by status code
    # ─────────────────────────────────────

    # 2xx Success Responses
    @staticmethod
    def success(
        message: str = "Success",
        data: Optional[Any] = None,
        status_code: int = 200
    ) -> "CustomResponse":
        """Create a success response (200 OK).

        Args:
            message: Success message (default: "Success")
            data: Response payload
            status_code: HTTP status code (default: 200)

        Returns:
            CustomResponse: Success response instance
        """
        return CustomResponse(
            message=message,
            is_success=True,
            data=data,
            status_code=status_code
        )

    @staticmethod
    def created(
        message: str = "Created",
        data: Optional[Any] = None
    ) -> "CustomResponse":
        """Create a created response (201 Created).

        Args:
            message: Success message (default: "Created")
            data: Response payload

        Returns:
            CustomResponse: Created response instance with 201 status code
        """
        return CustomResponse(
            message=message,
            is_success=True,
            data=data,
            status_code=201
        )

    # 5xx Server Error Responses
    @staticmethod
    def error(
        message: str = "Error",
        status_code: int = 500
    ) -> "CustomResponse":
        """Create an error response.

        Args:
            message: Error message (default: "Error")
            status_code: HTTP error status code (default: 500)

        Returns:
            CustomResponse: Error response instance
        """
        return CustomResponse(
            message=message,
            is_success=False,
            data=None,
            status_code=status_code
        )

    @staticmethod
    def something_was_wrong(message: str = "Something went wrong. Try later.") -> "CustomResponse":
        """Create an internal server error response (500 Internal Server Error).

        Convenience method for creating 500 Internal Server Error responses.

        Args:
            message: Error message (default: "Something went wrong. Try later.")

        Returns:
            CustomResponse: Error response instance with 500 status code
        """
        return CustomResponse(
            message=message,
            is_success=False,
            data=None,
            status_code=500
        )

    # Serialization methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary.

        Returns:
            Dictionary with message, is_success, and data fields
        """
        return {
            "message": self.message,
            "is_success": self.is_success,
            "data": self.data
        }

    def to_json_response(self) -> JSONResponse:
        """Convert response to FastAPI JSONResponse with proper status code.

        Handles serialization of Pydantic models in the data field.

        Returns:
            JSONResponse: FastAPI response with appropriate status code
        """
        # Serialize data if it's a Pydantic model
        data = self.data
        if isinstance(self.data, BaseModel):
            data = self.data.model_dump()

        response_dict = {
            "message": self.message,
            "is_success": self.is_success,
            "data": data
        }

        return JSONResponse(
            content=response_dict,
            status_code=self.status_code
        )

    # Deprecated methods for backward compatibility
    def get_response(self) -> Dict[str, Any]:
        """Deprecated: Use to_dict() instead.

        Returns:
            Dictionary representation of response
        """
        return self.to_dict()

    def get_JSONResponse(self) -> JSONResponse:
        """Deprecated: Use to_json_response() instead.

        Returns:
            JSONResponse with proper status code
        """
        return self.to_json_response()
