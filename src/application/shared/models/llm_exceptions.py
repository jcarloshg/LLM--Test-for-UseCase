"""Custom exception classes for LLM operations."""


class LLMClientException(Exception):
    """Exception raised when LLM model call fails.

    This exception is raised when an error occurs during LLM model inference,
    including API errors, timeout errors, validation errors, or any other
    issues that occur when calling the language model.

    Attributes:
        message: Human-readable error message
        model: Name of the model that failed (optional)
        provider: LLM service provider (ollama, anthropic, openai, etc.)
        original_error: The original exception that caused this error (optional)
    """

    def __init__(
        self,
        message: str,
        model: str = None,
        provider: str = None,
        original_error: Exception = None
    ):
        """Initialize LLMClientException.

        Args:
            message: Human-readable error message describing what went wrong
            model: Name of the model that failed (optional)
            provider: LLM service provider name (optional)
            original_error: The original exception that triggered this error (optional)
        """
        self.message = message
        self.model = model
        self.provider = provider
        self.original_error = original_error

        # Construct detailed error message
        error_details = [message]
        if provider:
            error_details.append(f"Provider: {provider}")
        if model:
            error_details.append(f"Model: {model}")
        if original_error:
            error_details.append(f"Original error: {str(original_error)}")

        full_message = " | ".join(error_details)
        super().__init__(full_message)

    def __str__(self) -> str:
        """String representation of the exception."""
        return f"LLMClientException: {self.message}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"LLMClientException(message={self.message!r}, "
            f"model={self.model!r}, provider={self.provider!r})"
        )
