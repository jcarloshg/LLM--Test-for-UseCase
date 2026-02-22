"""FAISS vectorstore management module."""

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG

logger = logging.getLogger(__name__)


def load_faiss_vectorstore() -> VectorStoreRetriever:
    """Load FAISS vectorstore and return as retriever.

    Loads an existing FAISS vectorstore from the configured path.
    If the vectorstore doesn't exist, raises FileNotFoundError with instructions.

    Returns:
        VectorStoreRetriever: The loaded vectorstore as a retriever

    Raises:
        FileNotFoundError: If vectorstore doesn't exist at the configured path
    """
    vectorstore_path = ENVIRONMENT_CONFIG.VECTORSTORE_PATH

    # ─────────────────────────────────────
    # Check if vectorstore exists
    # ─────────────────────────────────────
    if os.path.exists(vectorstore_path):
        logger.info("Loading existing FAISS vectorstore from %s...", vectorstore_path)

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B
        )

        # Load vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Convert to retriever
        retriever = vectorstore.as_retriever()
        logger.info("✅ FAISS vectorstore loaded successfully")

        return retriever

    else:
        logger.error("FAISS vectorstore not found!")
        logger.error("Please create it first by running: python scripts/save_vectorstore.py")
        raise FileNotFoundError(
            "Vectorstore not found. Create it by running: python scripts/save_vectorstore.py"
        )
