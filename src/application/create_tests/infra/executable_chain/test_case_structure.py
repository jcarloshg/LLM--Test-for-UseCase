"""
Test case structure validation model.
Defines the required structure for generated test cases.
"""
from typing import List, Any
from pydantic import BaseModel, Field


class TestCaseStructure(BaseModel):
    """
    Validates test case structure for generated artifacts.

    All fields are required to ensure test cases are complete and valid.
    """
    id: str = Field(..., description="Unique test case identifier (e.g., TC_001)")
    type: str = Field(..., description="Test case type (e.g., positive, negative, edge_case)")
    title: str = Field(..., description="Brief descriptive title of the test case")
    priority: str = Field(..., description="Priority level (e.g., high, medium, low)")
    preconditions: List[str] = Field(..., description="List of preconditions required before test execution")
    steps: List[str] = Field(..., description="List of step-by-step instructions for test execution")
    expected_result: str = Field(..., description="Expected outcome or result of the test")
    quality_score: int = Field(default=5, description="Quality score for the test case (0-10), defaults to 5 if not provided")

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True


class TestCasesResponse(BaseModel):
    """
    Validates the response structure containing a list of test cases.
    """
    test_cases: List[TestCaseStructure] = Field(..., description="List of test cases")
