"""Utility helper functions"""
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(level: str = None) -> logging.Logger:
    """Setup logging configuration"""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_documents(file_path: str) -> list:
    """Load documents from JSON file"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, dict):
        # Check for user_stories format
        if "user_stories" in data:
            documents = []
            for story in data["user_stories"]:
                doc = {
                    "id": story.get("id", ""),
                    "title": story.get("user_story", ""),
                    "content": _flatten_user_story(story)
                }
                documents.append(doc)
            return documents
        # Check for documents format
        elif "documents" in data:
            return data["documents"]
        else:
            return [data]
    elif isinstance(data, list):
        return data
    else:
        return [data]


def _flatten_user_story(story: dict) -> str:
    """Flatten user story with test cases into readable text"""
    content = f"User Story: {story.get('user_story', '')}\n"
    content += f"Quality Score: {story.get('quality_score', 'N/A')}\n\n"

    test_cases = story.get("test_cases", [])
    if test_cases:
        content += "Test Cases:\n"
        for tc in test_cases:
            content += f"\n  Test Case {tc.get('id', '')}: {tc.get('title', '')}\n"
            content += f"    Type: {tc.get('type', '')}\n"
            content += f"    Priority: {tc.get('priority', '')}\n"

            if tc.get('preconditions'):
                content += f"    Preconditions: {', '.join(tc['preconditions'])}\n"

            if tc.get('steps'):
                content += f"    Steps: {', '.join(tc['steps'])}\n"

            if tc.get('expected_result'):
                content += f"    Expected Result: {tc['expected_result']}\n"

    return content


def format_output(result: dict) -> str:
    """Format chain output for display"""
    if isinstance(result, dict):
        return result.get("output", "").strip()
    return str(result).strip()
