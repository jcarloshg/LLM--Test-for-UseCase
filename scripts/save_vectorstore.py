"""Script to create and save FAISS vectorstore"""

import sys
from pathlib import Path

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.shared.infrastructure.environment_variables import ENVIRONMENT_CONFIG
from utils.helpers import load_documents, setup_logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv


load_dotenv()


def create_and_save_vectorstore():
    """Create FAISS vectorstore from user stories and save locally"""

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Creating FAISS Vectorstore")
    logger.info("=" * 60)

    try:
        # Load data
        json_path = ENVIRONMENT_CONFIG.JSON_FILE_PATH
        logger.info("Loading data from %s...", json_path)
        data = load_documents(json_path)
        user_stories = data.get('user_stories', []) if isinstance(
            data, dict) else data
        logger.info("Loaded %d user stories", len(user_stories))

        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings = OllamaEmbeddings(
            base_url=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_HOST,
            model=ENVIRONMENT_CONFIG.OLLAMA_SERVICE_MODEL_LLAMA3_2_1B
        )

        # Process user stories into texts and metadatas
        logger.info("Processing user stories...")
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

        # Create vectorstore
        logger.info("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_texts(
            texts,
            embeddings,
            metadatas=metadatas
        )

        # Save vectorstore
        logger.info("Saving vectorstore to '%s'...", ENVIRONMENT_CONFIG.VECTORSTORE_PATH)
        vectorstore.save_local(ENVIRONMENT_CONFIG.VECTORSTORE_PATH)

        logger.info("=" * 60)
        logger.info("✅ FAISS Vectorstore created successfully!")
        logger.info("=" * 60)
        print("\n✅ Ready to use! Run: python main.py")

    except Exception as e:
        logger.error("Error creating vectorstore: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    create_and_save_vectorstore()
