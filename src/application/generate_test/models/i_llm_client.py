from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class ILlmClientResponse(BaseModel):
    """Response model for LLM client generation.

    Attributes:
        text: Generated text response from the LLM
        latency: Time taken to generate the response in seconds
        tokens: Number of tokens used in the generation
        model: Model identifier used for generation
        provider: LLM service provider used
    """
    text: str = Field(..., description="Generated text response")
    latency: float = Field(..., description="Response latency in seconds")
    tokens: int = Field(default=0, description="Number of tokens used")
    model: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="LLM service provider")

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True


class ILlmClient(ABC):
    """Abstract base class for LLM clients.

    Defines the interface that all LLM client implementations must follow.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> ILlmClientResponse:
        """Generate text using the LLM.

        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt to guide LLM behavior (optional)

        Returns:
            ILlmClientResponse: Response containing generated text and metadata

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
