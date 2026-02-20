import time
import ollama
import os

from src.application.generate_test.models.llm_config import LLMConfig
from src.application.generate_test.models.llm_client import LlmClient, ILlmClientResponse
from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


class LLMClientOllama(LlmClient):
    """Ollama implementation of ILlmClient.

    Provides integration with Ollama for local LLM inference.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Ollama LLM client.

        Args:
            config: LLMConfig instance.
        """
        self.config = config
        # Set Ollama host from environment config
        os.environ["OLLAMA_HOST"] = ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST

    def generate(self, prompt: str, system_prompt: str = "") -> ILlmClientResponse:
        """Generate text using Ollama.

        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt to guide LLM behavior (optional)

        Returns:
            ILlmClientResponse: Response containing generated text and metadata

        Raises:
            Exception: Any errors from Ollama API calls
        """
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ─────────────────────────────────────
        # TODO: add this a logging
        # ─────────────────────────────────────
        response = ollama.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        )

        # ─────────────────────────────────────
        # TODO: add this a logging
        # ─────────────────────────────────────
        print(f"="*60)
        print(response)
        print(f"="*60)

        latency = time.time() - start
        tokens = response.get('eval_count', 0) + \
            response.get('prompt_eval_count', 0)

        return ILlmClientResponse(
            text=response['message']['content'],
            latency=latency,
            tokens=tokens,
            model=self.config.model,
            provider="ollama"
        )
