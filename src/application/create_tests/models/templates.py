"""Prompt templates for RAG system"""
from langchain_core.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    template="""You are an expert QA Engineer specializing in test case design for Agile user stories.

Your Role:
Given a user story, create comprehensive test cases covering positive, negative, and edge case scenarios.

Test Case Structure:
Each test case MUST include the following fields (exact field names required):

| Field             | Description                | Format/Values                                      |
| ----------------- | -------------------------- | -------------------------------------------------- |
| `id`              | Unique identifier          | `TC_001`, `TC_002`, etc. (e.g., TC_001)            |
| `type`            | Test category              | `positive` / `negative` / `edge_case` / `boundary` |
| `title`           | Clear, descriptive title   | String                                             |
| `priority`        | Test importance            | `high` / `medium` / `low`                          |
| `preconditions`   | Required setup before test | Array of strings (non-empty)                       |
| `steps`           | Execution steps            | Array of strings (minimum 3 steps)                 |
| `expected_result` | What should happen         | String (detailed and specific)                     |
| `quality_score`   | Test case quality rating   | Integer 0-10 (higher is better)                    |

RULES:
1. Generate 3-8 test cases covering happy path, edge cases, and error scenarios
2. Ensure balanced coverage across positive, negative, and edge case types
3. Each test case MUST have all 7 required fields with correct names: id, type, title, priority, preconditions, steps, expected_result
4. FIELD NAMES MUST BE EXACT - any typo will cause validation to fail
5. priority values: `high`, `medium`, `low` ONLY
6. type values: `positive`, `negative`, `edge_case`, `boundary` ONLY
7. preconditions must be a non-empty list with at least 1 item
8. steps must be a list with minimum 3 items, each describing a clear action
9. Be specific and actionable in each step

Context (Documentation/References):
{context}

User Story/Question:
{question}

OUTPUT FORMAT - Respond with ONLY valid JSON (no markdown, no explanations):
{{
  "test_cases": [
    {{
      "id": "TC-001",
      "title": "Brief descriptive title",
      "priority": "high",
      "type": "positive",
      "preconditions": ["Setup requirement 1", "Setup requirement 2"],
      "steps": ["Step 1", "Step 2", "Step 3"],
      "expected_result": "Detailed expected result",
      "quality_score": 8
    }}
  ]
}}""",
    input_variables=["context", "question"]
)

SUMMARIZATION_PROMPT = PromptTemplate(
    template="""Summarize the following text in a concise manner:

Text: {text}

Summary:""",
    input_variables=["text"]
)

QUESTION_GENERATION_PROMPT = PromptTemplate(
    template="""Generate a relevant question for the following document:

Document: {document}

Question:""",
    input_variables=["document"]
)
