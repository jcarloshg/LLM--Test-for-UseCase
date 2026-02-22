"""Prompt templates for RAG system"""
from langchain_core.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    template="""
You are an expert QA Engineer specializing in test case design for Agile user stories.

Your Role:
Given a user story, create comprehensive test cases covering positive, negative, and edge case scenarios.

Test Case Structure:
Each test case must include the following fields:

| Field             | Description                | Format/Values                                      |
| ----------------- | -------------------------- | -------------------------------------------------- |
| `id`              | Unique identifier          | `TC-XXX` (e.g., TC-001, TC-002)                    |
| `type`            | Test category              | `positive` / `negative` / `edge_case` / `boundary` |
| `title`           | Clear, descriptive title   | String                                             |
| `priority`        | Test importance            | `critical` / `high` / `medium` / `low`             |
| `preconditions`   | Required setup before test | Array of strings                                   |
| `steps`           | Execution steps            | Array of strings ONLY (e.g., ["Step 1: Do X", "Step 2: Do Y"])  |
| `expected_result` | What should happen         | String                                             |
| `quality_score`   | Test coverage quality      | `1-10` (10 = comprehensive)                        |

RULES:
1. Generate 3-8 test cases covering happy path, edge cases, and error scenarios
2. Ensure balanced coverage across positive, negative, and edge case types
3. Each test case MUST have all required fields
4. Priority distribution: 1-2 critical, 2-3 high, 1-2 medium, 0-1 low
5. Quality score should reflect test comprehensiveness (6-9 range typical)
6. Be specific and actionable in each step
7. IMPORTANT: "steps" MUST be an array of strings ONLY - NOT objects with "number" or "step" keys
8. IMPORTANT: "preconditions" MUST be an array of strings ONLY - NOT objects

Context:
{context}

User Story:
{question}

Respond with ONLY valid JSON (no markdown, no explanations):
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

PROMPT = PromptTemplate(
    template="""Create 3-8 test cases for this user story. Include positive, negative, and edge cases.

Required JSON fields (use exact names): id, type, title, priority, preconditions, steps, expected_result, quality_score
- id: TC_001, TC_002, etc.
- type: positive | negative | edge_case | boundary
- priority: high | medium | low
- preconditions: list of strings (≥1 item)
- steps: list of strings (≥3 items, each is a clear action)
- expected_result: specific outcome
- quality_score: 0-10

User Story:
{question}

Respond with ONLY valid JSON (no markdown, no explanations):
{{
  "test_cases": [
    {{
      "id": "TC-001",
      "type": "positive",
      "title": "...",
      "priority": "high", 
      "preconditions": ["..."],
      "steps": ["...", "...", "..."],
      "expected_result": "...",
      "quality_score": 8
    }}
  ]
}}""",
    input_variables=["question"]
)

IMPROVED_PROMPT_V1 = PromptTemplate(
    template="""
You are an expert QA Engineer specializing in test case design for Agile user stories.

Your Role:
Given a user story, create comprehensive test cases covering positive, negative, and edge case scenarios.

Test Case Structure:
Each test case must include the following fields:

| Field             | Description                | Format/Values                                      |
| ----------------- | -------------------------- | -------------------------------------------------- |
| `id`              | Unique identifier          | `TC-XXX` (e.g., TC-001, TC-002)                    |
| `type`            | Test category              | `positive` / `negative` / `edge_case` / `boundary` |
| `title`           | Clear, descriptive title   | String                                             |
| `priority`        | Test importance            | `critical` / `high` / `medium` / `low`             |
| `preconditions`   | Required setup before test | Array of strings                                   |
| `steps`           | Execution steps            | Array of strings ONLY (e.g., ["Step 1: Do X", "Step 2: Do Y"])  |
| `expected_result` | What should happen         | String                                             |
| `quality_score`   | Test coverage quality      | `1-10` (10 = comprehensive)                        |

RULES:
1. Generate 3-8 test cases covering happy path, edge cases, and error scenarios
2. Ensure balanced coverage across positive, negative, and edge case types
3. Each test case MUST have all required fields
4. Priority distribution: 1-2 critical, 2-3 high, 1-2 medium, 0-1 low
5. Quality score should reflect test comprehensiveness (6-9 range typical)
6. Be specific and actionable in each step
7. IMPORTANT: "steps" MUST be an array of strings ONLY - NOT objects with "number" or "step" keys
8. IMPORTANT: "preconditions" MUST be an array of strings ONLY - NOT objects

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
    input_variables=["question"]
)


# SUMMARIZATION_PROMPT = PromptTemplate(
#     template="""Summarize the following text in a concise manner:
#
# Text: {text}
#
# Summary:""",
#     input_variables=["text"]
# )
#
# QUESTION_GENERATION_PROMPT = PromptTemplate(
#     template="""Generate a relevant question for the following document:
#
# Document: {document}
#
# Question:""",
#     input_variables=["document"]
# )
