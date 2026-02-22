# model_configs.py
from dataclasses import dataclass
from typing import Optional
from typing import List

from langchain_ollama import OllamaLLM
# from langchain_anthropic import AnthropicLLM

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


@dataclass
class ModelConfig:
    """
    Best Practice: Standardize model configuration
    """
    name: str
    provider: str  # "openai", "anthropic", "ollama"
    model_id: str
    llm: any
    base_url: Optional[str] = None  # For self-hosted
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 3500
    temperature: float = 0.0  # Deterministic for evaluation
    api_key: Optional[str] = None


class ModelRegistry:
    """Predefined model configurations"""

    MODELS = {
        "llama-3.2-1b": ModelConfig(
            name="Llama 3.2 1B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_tokens=3500,
            temperature=0.0,
            llm=OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B,
                temperature=0.0,
            )
        ),

        "llama-3.2-3b": ModelConfig(
            name="Llama 3.2 3B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_3B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_tokens=3500,
            temperature=0.0,
            llm=OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_3B,
                temperature=0.0
            )
        ),

        "qwen3-vl-8b": ModelConfig(
            name="Qwen3-VL 8B",
            provider="ollama",
            model_id=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL8B,
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_tokens=3500,
            temperature=0.0,
            llm=OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL8B,
                temperature=0.0
            )
        )
    }

    def get_models_to_compare(self) -> List[ModelConfig]:
        return [
            # self.MODELS["llama-3.2-1b"],
            # self.MODELS["llama-3.2-3b"],
            self.MODELS["qwen3-vl-8b"]
        ]
