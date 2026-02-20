import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env.dev
load_dotenv(".env")


# ─────────────────────────────────────
# Environment Configuration Model
# ─────────────────────────────────────
class EnvironmentConfig(BaseModel):
    """Validated environment configuration."""

    # ─────────────────────────────────────
    # OLLAMA Configuration
    # ─────────────────────────────────────
    OLLAMA_SERVICE_HOST: str = Field(
        default="http://localhost:11435",
        alias="OLLAMA-SERVICE-HOST",
        description="The base URL of the Ollama service"
    )
    OLLAMA_SERVICE_MODEL_QWEN3VL4B: str = Field(
        default="llama3.2:3b",
        alias="OLLAMA_SERVICE_MODEL_QWEN3VL4B",
        description="The Qwen3-VL:4B model name"
    )

    # // ─────────────────────────────────────
    # Anthopic Configuration
    # // ─────────────────────────────────────
    ANTHOPIC_KEY: str = Field(
        default="ANTHOPIC_KEY",
        alias="ANTHOPIC_KEY",
        description="API key for Anthopic service"
    )
    ANTHOPIC_MODEL: str = Field(
        default="ANTHOPIC_MODEL",
        alias="ANTHOPIC_MODEL",
        description="Model name for Anthopic service"
    )

    # // ─────────────────────────────────────
    # ERROR HANDLING & RETRIES
    # // ─────────────────────────────────────
    MAX_RETRIES: int = Field(
        default=3,
        alias="MAX_RETRIES",
        description="Maximum number of retries for API calls"
    )
    MAX_RETRIES_USER_MSG: str = Field(
        default="The AI service is currently unavailable. Please try again in a moment.",
        alias="MAX_RETRIES_USER_MSG",
        description="User-friendly error message for retry failures"
    )
    MAX_RETRIES_DEV_MSG: str = Field(
        default="Failed to call Ollama. Attemp # ",
        alias="MAX_RETRIES_DEV_MSG",
        description="Developer error message template for retry failures"
    )

    def __str__(self) -> str:
        """String representation of environment configuration."""
        return (
            f"EnvironmentConfig(\n"
            f"  OLLAMA_SERVICE_HOST: {self.OLLAMA_SERVICE_HOST}\n"
            f"  OLLAMA_SERVICE_MODEL_QWEN3VL4B: {self.OLLAMA_SERVICE_MODEL_QWEN3VL4B}\n"
            f"  ANTHOPIC_KEY: {self.ANTHOPIC_KEY}\n"
            f"  ANTHOPIC_MODEL: {self.ANTHOPIC_MODEL}\n"
            f"  MAX_RETRIES: {self.MAX_RETRIES}\n"
            f"  MAX_RETRIES_USER_MSG: {self.MAX_RETRIES_USER_MSG}\n"
            f"  MAX_RETRIES_DEV_MSG: {self.MAX_RETRIES_DEV_MSG}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()

    class Config:
        populate_by_name = True


# // ─────────────────────────────────────
# Load and validate environment variables
# ENVIRONMENT_CONFIG
# // ─────────────────────────────────────
ENVIRONMENT_CONFIG = EnvironmentConfig(
    **{
        "OLLAMA-SERVICE-HOST": os.getenv(
            "OLLAMA-SERVICE-HOST",
            "http://localhost:11435"
        ),
        "OLLAMA_SERVICE_MODEL_QWEN3VL4B": os.getenv(
            "OLLAMA_SERVICE_MODEL_QWEN3VL4B",
            "llama3.2:3b"
        ),
        "ANTHOPIC_KEY": os.getenv(
            "ANTHOPIC_KEY",
            "ANTHOPIC_KEY"
        ),
        "ANTHOPIC_MODEL": os.getenv(
            "ANTHOPIC_MODEL",
            "ANTHOPIC_MODEL"
        ),
        "MAX_RETRIES": os.getenv(
            "MAX_RETRIES",
            "3"
        ),
        "MAX_RETRIES_USER_MSG": os.getenv(
            "MAX_RETRIES_USER_MSG",
            "The AI service is currently unavailable. Please try again in a moment."
        ),
        "MAX_RETRIES_DEV_MSG": os.getenv(
            "MAX_RETRIES_DEV_MSG",
            "Failed to call Ollama. Attemp # "
        ),
    }
)

print(f"ENVIRONMENT_CONFIG {ENVIRONMENT_CONFIG.__str__}")
