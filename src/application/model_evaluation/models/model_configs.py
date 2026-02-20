# model_configs.py
from dataclasses import dataclass
from typing import Optional
from typing import List

from src.application.generate_test.models.llm_config import LLMConfig
from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


@dataclass
class ModelConfig:
    """
    Best Practice: Standardize model configuration
    """
    name: str
    provider: str  # "openai", "anthropic", "ollama"
    model_id: str
    llm_config: LLMConfig
    temperature: float = 0.0  # Deterministic for evaluation
    max_tokens: int = 500
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For self-hosted


class ModelRegistry:
    """Predefined model configurations"""

    MODELS = {

        "claude-haiku": ModelConfig(
            name="Claude Haiku 4.5",
            provider="anthropic",
            model_id=ENVIRONMENT_CONFIG.ANTHOPIC_MODEL,
            api_key=ENVIRONMENT_CONFIG.ANTHOPIC_KEY,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            llm_config=LLMConfig(
                provider="anthropic",
                api_key=ENVIRONMENT_CONFIG.ANTHOPIC_KEY,
                model=ENVIRONMENT_CONFIG.ANTHOPIC_MODEL
            )
        ),

        "llama-3.2-1b": ModelConfig(
            name="Llama 3.2 1B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            llm_config=LLMConfig(
                provider="ollama",
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B
            )
        ),

        "llama-3.2-3b": ModelConfig(
            name="Llama 3.2 3B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            llm_config=LLMConfig(
                provider="ollama",
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL4B
            )
        ),

        "qwen3-vl-8b": ModelConfig(
            name="Qwen3-VL 8B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL8B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            llm_config=LLMConfig(
                provider="ollama",
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL8B
            )
        )
    }

    @classmethod
    def get_models_to_compare(cls) -> List[ModelConfig]:
        """
        Best Practice: Always compare at least 3 models:
        - 1 high-end (Claude)
        - 2 open-source (Llama, Qwen)

        Only includes Anthropic and Ollama providers.
        """
        return [
            cls.MODELS["llama-3.2-1b"],
            cls.MODELS["llama-3.2-3b"],
            cls.MODELS["qwen3-vl-8b"]
        ]
