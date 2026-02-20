from pydantic import BaseModel
from typing import Dict, Any


class TestCase(BaseModel):
    """Single test case structure"""
    id: str
    input: str
    expected_output: str
    category: str  # e.g., "simple", "complex", "edge_case"
    difficulty: str  # e.g., "easy", "medium", "hard"
    metadata: Dict[str, Any] = {}  # here the use cases
