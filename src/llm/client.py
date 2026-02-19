# src/llm/client.py
from typing import Optional, Dict, Any
import ollama
import openai
from pydantic import BaseModel
import time
import os


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    temperature: float = 0.3  # Low for consistency
    max_tokens: int = 2000
    api_key: Optional[str] = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.provider == "openai":
            openai.api_key = config.api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """Generate response from LLM"""
        start = time.time()

        try:
            if self.config.provider == "ollama":
                response = self._call_ollama(prompt, system_prompt)
            elif self.config.provider == "openai":
                response = self._call_openai(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider: {self.config.provider}")

            return {
                "text": response["text"],
                "latency": time.time() - start,
                "tokens": response.get("tokens", 0),
                "model": self.config.model,
                "provider": self.config.provider
            }
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "latency": time.time() - start,
                "tokens": 0
            }

    def _call_ollama(self, prompt: str, system_prompt: str) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        )

        return {
            "text": response['message']['content'],
            "tokens": response.get('eval_count', 0) + response.get('prompt_eval_count', 0)
        }

    def _call_openai(self, prompt: str, system_prompt: str) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }


# Test script
if __name__ == "__main__":
    config = LLMConfig()
    client = LLMClient(config)

    result = client.generate("Say hello")
    print(f"✅ Model test: {result['text']}")
    print(f"Latency: {result['latency']:.2f}s")
