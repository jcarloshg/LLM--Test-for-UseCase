"""FAISS vectorstore management module."""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG

logger = logging.getLogger(__name__)


class VectorStoreFAISS:
    """Unified FAISS vectorstore management class.

    Handles both creation/saving and loading of FAISS vectorstores with
    consistent embedding parameter handling.
    """

    def __init__(self, llm_embedding=None):
        """Initialize VectorStoreFAISS with optional embedding instance.

        Args:
            llm_embedding: Embedding model instance to use for vectorization
        """
        self.llm_embedding = llm_embedding
        self.vectorstore_path = ENVIRONMENT_CONFIG.VECTORSTORE_PATH

    def create_and_save(
        self,
        json_path: Optional[str] = None,
        vectorstore_path: Optional[str] = None,
    ) -> None:
        """Create FAISS vectorstore from user stories and save locally.

        Args:
            json_path: Path to JSON file with user stories (default: from config)
            vectorstore_path: Path to save vectorstore (default: from config)

        Raises:
            ValueError: If embedding is not configured
            Exception: If vectorstore creation fails
        """
        if self.llm_embedding is None:
            raise ValueError(
                "Embedding is not configured. Pass llm_embedding to init_embedding() "
                "or VectorStoreFAISS constructor."
            )

        json_path = json_path or ENVIRONMENT_CONFIG.JSON_FILE_PATH
        vectorstore_path = vectorstore_path or self.vectorstore_path

        logger.info("=" * 60)
        logger.info("Creating FAISS Vectorstore")
        logger.info("=" * 60)

        try:
            # Load data
            logger.info("Loading data from %s...", json_path)
            data = self._load_documents(json_path)
            user_stories = data.get('user_stories', []) if isinstance(
                data, dict) else data
            logger.info("Loaded %d user stories", len(user_stories))

            # Process user stories into texts and metadatas
            logger.info("Processing user stories...")
            texts, metadatas = self._process_stories(user_stories)

            # Create vectorstore
            logger.info("Creating FAISS vectorstore...")
            vectorstore = FAISS.from_texts(
                texts,
                self.llm_embedding,
                metadatas=metadatas
            )

            # Save vectorstore
            logger.info("Saving vectorstore to '%s'...", vectorstore_path)
            vectorstore.save_local(vectorstore_path)

            logger.info("=" * 60)
            logger.info("✅ FAISS Vectorstore created successfully!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("Error creating vectorstore: %s", str(e))
            raise

    def load(self) -> VectorStoreRetriever:
        """Load FAISS vectorstore and return as retriever.

        Loads an existing FAISS vectorstore from the configured path.
        If the vectorstore doesn't exist, raises FileNotFoundError with instructions.

        Returns:
            VectorStoreRetriever: The loaded vectorstore as a retriever

        Raises:
            ValueError: If embedding is not configured
            FileNotFoundError: If vectorstore doesn't exist at the configured path
        """
        if self.llm_embedding is None:
            raise ValueError(
                "Embedding is not configured. Pass llm_embedding to init_embedding() "
                "or VectorStoreFAISS constructor."
            )

        # ─────────────────────────────────────
        # Check if vectorstore exists
        # ─────────────────────────────────────
        if os.path.exists(self.vectorstore_path):
            logger.info("Loading existing FAISS vectorstore from %s...",
                        self.vectorstore_path)

            # Load vectorstore
            vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.llm_embedding,
                allow_dangerous_deserialization=True
            )

            # Convert to retriever
            retriever = vectorstore.as_retriever()
            logger.info("✅ FAISS vectorstore loaded successfully")

            return retriever

        else:
            logger.error("FAISS vectorstore not found!")
            logger.error(
                "Please create it first by running: python scripts/save_vectorstore.py")
            raise FileNotFoundError(
                "Vectorstore not found. Create it by running: python scripts/save_vectorstore.py"
            )

    @staticmethod
    def _load_documents(json_path: str) -> Dict[str, Any]:
        """Load user stories from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Loaded JSON data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _process_stories(
        user_stories: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Process user stories into texts and metadata.

        Args:
            user_stories: List of user story dictionaries

        Returns:
            Tuple of (texts, metadatas)
        """
        texts = []
        metadatas = []

        for story in user_stories:
            story_id = story.get('id', '')
            user_story_text = story.get('user_story', '')
            quality_score = story.get('quality_score', 0)
            test_cases = story.get('test_cases', [])

            # Create comprehensive text
            test_cases_summary = "\n".join([
                f"- {tc.get('id', '')}: {tc.get('title', '')}"
                for tc in test_cases
            ])

            content = f"{user_story_text}\n\nTest Cases:\n{test_cases_summary}"
            texts.append(content)

            metadatas.append({
                'id': story_id,
                'title': user_story_text[:100],
                'quality_score': quality_score,
                'test_case_count': len(test_cases)
            })

        return texts, metadatas
