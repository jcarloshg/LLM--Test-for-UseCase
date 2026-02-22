from typing import Any
from pydantic import BaseModel, Field


class ExecutableChainResponse(BaseModel):
    """"
    executable_chain_response
    """
    result: Any = Field(..., description="Generated result from the chain")
    latency: float = Field(..., description="Response latency in seconds")
    tokens: int = Field(default=0, description="Number of tokens used")
    model: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="LLM service provider")

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
