from pydantic import BaseModel, Field
from typing import Optional

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG

# ─────────────────────────────────────
# TODO: implement this here ENVIRONMENT_CONFIG
# ─────────────────────────────────────


class LLMConfig(BaseModel):
    """
    Configuration for LLM (Language Model) clients.
    """
    provider: str = Field(
        default="ollama",
        description="LLM service provider"
    )
    model: str = Field(
        default=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
        description="Model identifier"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for response randomness (0.0-1.0)"
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens to generate"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for provider authentication"
    )

    class Config:
        """Pydantic configuration for strict validation."""
        str_strip_whitespace = True
        validate_assignment = True
