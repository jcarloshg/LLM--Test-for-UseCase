# src/validators/structure.py
from pydantic import BaseModel, Field, validator
from typing import List, Literal


class TestCase(BaseModel):
    """Single test case structure"""
    id: str = Field(..., pattern=r"^TC_\d+$")
    title: str = Field(..., min_length=10, max_length=200)
    priority: Literal["critical", "high", "medium", "low"]
    given: str = Field(..., min_length=10)
    when: str = Field(..., min_length=10)
    then: str = Field(..., min_length=10)

    @validator('given', 'when', 'then')
    def not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty")
        return v.strip()


class TestCaseOutput(BaseModel):
    """Complete output structure"""
    test_cases: List[TestCase] = Field(..., min_items=3, max_items=10)

    @validator('test_cases')
    def validate_ids_unique(cls, v):
        ids = [tc.id for tc in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Test case IDs must be unique")
        return v


class StructureValidator:
    """Validate test case structure"""

    @staticmethod
    def validate(output_json: dict) -> dict:
        """
        Returns:
        {
            "valid": bool,
            "errors": list,
            "test_cases": list (if valid)
        }
        """
        try:
            validated = TestCaseOutput(**output_json)
            return {
                "valid": True,
                "errors": [],
                "test_cases": [tc.dict() for tc in validated.test_cases],
                "count": len(validated.test_cases)
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "test_cases": [],
                "count": 0
            }
