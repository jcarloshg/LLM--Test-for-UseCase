# model_configs.py
from dataclasses import dataclass
from typing import Optional
from typing import List


@dataclass
class ModelConfig:
    """
    Best Practice: Standardize model configuration
    """
    name: str
    provider: str  # "openai", "anthropic", "ollama"
    model_id: str
    temperature: float = 0.0  # Deterministic for evaluation
    max_tokens: int = 500
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For self-hosted


class ModelRegistry:
    """Predefined model configurations"""

    MODELS = {
        "gpt-4": ModelConfig(
            name="GPT-4",
            provider="openai",
            model_id="gpt-4",
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06
        ),

        "gpt-3.5-turbo": ModelConfig(
            name="GPT-3.5 Turbo",
            provider="openai",
            model_id="gpt-3.5-turbo",
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015
        ),

        "claude-sonnet": ModelConfig(
            name="Claude Sonnet 3.5",
            provider="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015
        ),

        "llama-3.2-3b": ModelConfig(
            name="Llama 3.2 3B",
            provider="ollama",
            model_id="llama3.2:3b",
            cost_per_1k_input=0.0,  # Self-hosted
            cost_per_1k_output=0.0
        ),

        "mistral-7b": ModelConfig(
            name="Mistral 7B",
            provider="ollama",
            model_id="mistral:7b",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0
        )
    }

    @classmethod
    def get_models_to_compare(cls) -> List[ModelConfig]:
        """
        Best Practice: Always compare at least 3 models:
        - 1 high-end (GPT-4, Claude)
        - 1 mid-tier (GPT-3.5)
        - 1 open-source (Llama, Mistral)
        """
        return [
            cls.MODELS["gpt-4"],
            cls.MODELS["gpt-3.5-turbo"],
            cls.MODELS["llama-3.2-3b"]
        ]
