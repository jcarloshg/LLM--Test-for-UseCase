from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    user_story: str = Field(..., min_length=20, max_length=500)
    include_quality_check: bool = True
    model: Optional[str] = None
