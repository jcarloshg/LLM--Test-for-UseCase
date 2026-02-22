# test_case.py
from pydantic import BaseModel


class TestCase(BaseModel):
    """Single test case structure"""
    id: str
    user_story: str
    difficulty: str  # e.g., "easy", "medium", "hard"
