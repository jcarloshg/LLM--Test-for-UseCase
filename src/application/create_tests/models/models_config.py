from langchain_ollama import OllamaLLM

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG


class SingletonMeta(type):
    """Metaclass for implementing singleton pattern."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Ensure only one instance of the class exists."""
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ModelsConfig(metaclass=SingletonMeta):
    """Singleton configuration manager for LLM models.

    Ensures only one instance exists globally and provides factory methods to create
    and cache LLM instances using environment configuration.
    """

    def __init__(self):
        """Initialize the models configuration with cached model instances."""
        self._llama3_2_1b = None
        self._llama3_2_3b = None
        self._qwen3vl_8b = None

    def get_llama3_2_1b(self) -> OllamaLLM:
        """Get or create the Llama3.2:1B model instance.

        Returns:
            OllamaLLM: Cached instance of Llama3.2:1B model
        """
        if self._llama3_2_1b is None:
            self._llama3_2_1b = OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B,
                temperature=0.7
            )
        return self._llama3_2_1b

    def get_llama3_2_3b(self) -> OllamaLLM:
        """Get or create the Llama3.2:3B model instance.

        Returns:
            OllamaLLM: Cached instance of Llama3.2:3B model
        """
        if self._llama3_2_3b is None:
            self._llama3_2_3b = OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_3B,
                temperature=0.7
            )
        return self._llama3_2_3b

    def get_qwen3vl_8b(self) -> OllamaLLM:
        """Get or create the Qwen3-VL:8B model instance.

        Returns:
            OllamaLLM: Cached instance of Qwen3-VL:8B model
        """
        if self._qwen3vl_8b is None:
            self._qwen3vl_8b = OllamaLLM(
                base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
                model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_QWEN3VL8B,
                temperature=0.7
            )
        return self._qwen3vl_8b
