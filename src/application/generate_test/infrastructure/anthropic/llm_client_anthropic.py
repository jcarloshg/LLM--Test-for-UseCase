import time
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from src.application.generate_test.models.llm_config import LLMConfig
from src.application.generate_test.models.llm_client import LlmClient, ILlmClientResponse


class LLMClientAnthropic(LlmClient):
    """Anthropic implementation of ILlmClient.

    Provides integration with Anthropic for LLM inference using LangChain.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic LLM client.

        Args:
            config: LLMConfig instance.
        """
        self.config = config
        self.client = ChatAnthropic(
            api_key=config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def generate(self, prompt: str, system_prompt: str = "") -> ILlmClientResponse:
        """Generate text using Anthropic.

        Args:
            prompt: User prompt for the LLM
            system_prompt: System prompt to guide LLM behavior (optional)

        Returns:
            ILlmClientResponse: Response containing generated text and metadata

        Raises:
            Exception: Any errors from Anthropic API calls
        """
        start = time.time()

        # ─────────────────────────────────────
        # TODO: add this a logging
        # ─────────────────────────────────────
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.client.invoke(messages)

        # ─────────────────────────────────────
        # TODO: add this a logging
        # ─────────────────────────────────────
        print(f"="*60)
        print(response)
        print(f"="*60)

        latency = time.time() - start

        # Extract token counts from response metadata
        tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = response.usage_metadata.get(
                "input_tokens", 0) + response.usage_metadata.get("output_tokens", 0)

        return ILlmClientResponse(
            text=response.content,
            latency=latency,
            tokens=tokens,
            model=self.config.model,
            provider="anthropic"
        )
