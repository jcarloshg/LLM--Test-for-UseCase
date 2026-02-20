# src/validators/structure.py
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class GherkinFormat(BaseModel):
    """Gherkin BDD format structure"""
    given: str = Field(..., min_length=1)
    when: str = Field(..., min_length=1)
    then: str = Field(..., min_length=1)

    @validator('given', 'when', 'then')
    def not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Field cannot be empty")
        return v.strip()


class Step(BaseModel):
    """Single step with Gherkin format"""
    given: str = Field(..., min_length=1)
    when: str = Field(..., min_length=1)
    then: str = Field(..., min_length=1)

    @validator('given', 'when', 'then')
    def validate_step_fields(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Step field cannot be empty")
        return v.strip()


class TestCase(BaseModel):
    """Single test case structure - supports both simple and comprehensive formats"""
    id: str = Field(..., pattern=r"^TC[_-]\d+$")
    title: str = Field(..., min_length=10, max_length=200)
    priority: Literal["critical", "high", "medium", "low"]

    # Simple format fields (backward compatible)
    given: Optional[str] = Field(None, min_length=10)
    when: Optional[str] = Field(None, min_length=10)
    then: Optional[str] = Field(None, min_length=10)

    # Comprehensive format fields
    type: Optional[Literal["positive", "negative", "edge_case", "boundary"]] = None
    preconditions: Optional[List[str]] = None
    steps: Optional[List] = None
    gherkin: Optional[GherkinFormat] = None
    expected_result: Optional[str] = None
    quality_score: Optional[int] = Field(None, ge=1, le=10)
    tags: Optional[List[str]] = None

    @validator('given', 'when', 'then')
    def validate_simple_fields(cls, v):
        if v is not None and (not v or v.strip() == ""):
            raise ValueError("Field cannot be empty")
        return v.strip() if v else None

    @validator('preconditions', 'tags', pre=True, always=True)
    def validate_lists(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Field must be a list")
        return v

    @validator('steps', pre=True, always=True)
    def validate_steps(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Steps must be a list")
        # Steps can be either strings or Step objects (dicts with given/when/then)
        validated_steps = []
        for step in v:
            if isinstance(step, dict):
                # Convert dict to Step object for validation
                validated_steps.append(Step(**step))
            elif isinstance(step, str):
                # Keep string steps as-is
                validated_steps.append(step)
            else:
                raise ValueError("Step must be either a string or an object with given/when/then")
        return validated_steps


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
                "test_cases": [tc.model_dump() for tc in validated.test_cases],
                "count": len(validated.test_cases)
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "test_cases": [],
                "count": 0
            }
