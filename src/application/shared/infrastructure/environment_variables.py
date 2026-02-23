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
        alias="OLLAMA_SERVICE_HOST",
        description="The base URL of the Ollama service"
    )
    OLLAMA_SERVICE_MODEL_LLAMA3_2_1B: str = Field(
        default="llama3.2:1b",
        alias="OLLAMA_SERVICE_MODEL_LLAMA3-2-1B",
        description="The Llama3.2:1B model name"
    )
    OLLAMA_SERVICE_MODEL_LLAMA3_2_3B: str = Field(
        default="llama3.2:3b",
        alias="OLLAMA_SERVICE_MODEL_LLAMA3-2-3B",
        description="The Llama3.2:3B model name"
    )
    OLLAMA_SERVICE_MODEL_QWEN3VL8B: str = Field(
        default="qwen3-vl:8b",
        alias="OLLAMA_SERVICE_MODEL_QWEN3VL8B",
        description="The Qwen3-VL:8B model name"
    )
    OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B: str = Field(
        default="llama3-chatqa:8b",
        alias="OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B",
        description="The Llama3-ChatQA:8B model name"
    )
    OLLAMA_SERVICE_MODEL_EMBEDDING: str = Field(
        default="nomic-embed-text",
        alias="OLLAMA_SERVICE_MODEL_EMBEDDING",
        description="The embedding model name"
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

    # // ─────────────────────────────────────
    # DATA & VECTORSTORE
    # // ─────────────────────────────────────
    JSON_FILE_PATH: str = Field(
        default="data/examples/user_stories_with_test_cases.json",
        alias="JSON_FILE_PATH",
        description="Path to the JSON file with user stories and test cases"
    )
    VECTORSTORE_PATH: str = Field(
        default="data/vectorstore_faiss",
        alias="VECTORSTORE_PATH",
        description="Path to the FAISS vectorstore directory"
    )

    # // ─────────────────────────────────────
    # LOGGING & OBSERVABILITY
    # // ─────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    LOG_FORMAT: str = Field(
        default="json",
        alias="LOG_FORMAT",
        description="Log format (json or text)"
    )
    LOG_FILE_PATH: str = Field(
        default="logs/app.json.log",
        alias="LOG_FILE_PATH",
        description="Path to log file for structured logging"
    )
    LOG_ROTATION_SIZE: int = Field(
        default=10485760,
        alias="LOG_ROTATION_SIZE",
        description="Log rotation size in bytes (10MB default)"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        alias="LOG_BACKUP_COUNT",
        description="Number of backup log files to keep"
    )
    ENVIRONMENT: str = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Deployment environment (development, staging, production)"
    )
    SERVICE_VERSION: str = Field(
        default="1.0.0",
        alias="SERVICE_VERSION",
        description="Service version"
    )

    def __str__(self) -> str:
        """String representation of environment configuration."""
        return (
            f"EnvironmentConfig(\n"
            f"  OLLAMA_SERVICE_HOST: {self.OLLAMA_SERVICE_HOST}\n"
            f"  OLLAMA_SERVICE_MODEL_LLAMA3_2_1B: {self.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B}\n"
            f"  OLLAMA_SERVICE_MODEL_LLAMA3_2_3B: {self.OLLAMA_SERVICE_MODEL_LLAMA3_2_3B}\n"
            f"  OLLAMA_SERVICE_MODEL_QWEN3VL8B: {self.OLLAMA_SERVICE_MODEL_QWEN3VL8B}\n"
            f"  OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B: {self.OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B}\n"
            f"  OLLAMA_SERVICE_MODEL_EMBEDDING: {self.OLLAMA_SERVICE_MODEL_EMBEDDING}\n"
            f"  ANTHOPIC_KEY: {self.ANTHOPIC_KEY}\n"
            f"  ANTHOPIC_MODEL: {self.ANTHOPIC_MODEL}\n"
            f"  MAX_RETRIES: {self.MAX_RETRIES}\n"
            f"  MAX_RETRIES_USER_MSG: {self.MAX_RETRIES_USER_MSG}\n"
            f"  MAX_RETRIES_DEV_MSG: {self.MAX_RETRIES_DEV_MSG}\n"
            f"  JSON_FILE_PATH: {self.JSON_FILE_PATH}\n"
            f"  VECTORSTORE_PATH: {self.VECTORSTORE_PATH}\n"
            f"  LOG_LEVEL: {self.LOG_LEVEL}\n"
            f"  LOG_FORMAT: {self.LOG_FORMAT}\n"
            f"  LOG_FILE_PATH: {self.LOG_FILE_PATH}\n"
            f"  ENVIRONMENT: {self.ENVIRONMENT}\n"
            f"  SERVICE_VERSION: {self.SERVICE_VERSION}\n"
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
        "OLLAMA_SERVICE_HOST": os.getenv(
            "OLLAMA_HOST",
            os.getenv("OLLAMA_SERVICE_HOST", "http://localhost:11435")
        ),
        "OLLAMA_SERVICE_MODEL_LLAMA3-2-1B": os.getenv(
            "OLLAMA_SERVICE_MODEL_LLAMA3-2-1B",
            "llama3.2:1b"
        ),
        "OLLAMA_SERVICE_MODEL_LLAMA3-2-3B": os.getenv(
            "OLLAMA_SERVICE_MODEL_LLAMA3-2-3B",
            "llama3.2:3b"
        ),
        "OLLAMA_SERVICE_MODEL_QWEN3VL8B": os.getenv(
            "OLLAMA_SERVICE_MODEL_QWEN3VL8B",
            "qwen3-vl:8b"
        ),
        "OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B": os.getenv(
            "OLLAMA_SERVICE_MODEL_LLAMA3_CHATQA_8B",
            "llama3-chatqa:8b"
        ),
        "OLLAMA_SERVICE_MODEL_EMBEDDING": os.getenv(
            "OLLAMA_SERVICE_MODEL_EMBEDDING",
            "nomic-embed-text"
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
        "JSON_FILE_PATH": os.getenv(
            "JSON_FILE_PATH",
            "data/user_stories_with_test_cases.json"
        ),
        "VECTORSTORE_PATH": os.getenv(
            "VECTORSTORE_PATH",
            "data/vectorstore_faiss"
        ),
        "LOG_LEVEL": os.getenv(
            "LOG_LEVEL",
            "INFO"
        ),
        "LOG_FORMAT": os.getenv(
            "LOG_FORMAT",
            "json"
        ),
        "LOG_FILE_PATH": os.getenv(
            "LOG_FILE_PATH",
            "logs/app.json.log"
        ),
        "LOG_ROTATION_SIZE": int(os.getenv(
            "LOG_ROTATION_SIZE",
            "10485760"
        )),
        "LOG_BACKUP_COUNT": int(os.getenv(
            "LOG_BACKUP_COUNT",
            "5"
        )),
        "ENVIRONMENT": os.getenv(
            "ENVIRONMENT",
            "development"
        ),
        "SERVICE_VERSION": os.getenv(
            "SERVICE_VERSION",
            "1.0.0"
        ),
    }
)
