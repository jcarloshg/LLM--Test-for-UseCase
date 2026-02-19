from pydantic import BaseModel
from typing import Optional, List


class TestCaseResponse(BaseModel):
    id: str
    title: str
    priority: str
    given: str
    when: str
    then: str


class GenerateResponse(BaseModel):
    user_story: str
    test_cases: List[TestCaseResponse]
    validation: dict
    quality_metrics: Optional[dict] = None
    metadata: dict
