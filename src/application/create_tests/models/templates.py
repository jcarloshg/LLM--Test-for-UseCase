"""Prompt templates for RAG system"""
from langchain_core.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    template="""Create 3-8 test cases for this user story. Include positive, negative, and edge cases.

Required JSON fields (use exact names): id, type, title, priority, preconditions, steps, expected_result, quality_score
- id: TC_001, TC_002, etc.
- type: positive | negative | edge_case | boundary
- priority: high | medium | low
- preconditions: list of strings (≥1 item)
- steps: list of strings (≥3 items, each is a clear action)
- expected_result: specific outcome
- quality_score: 0-10

Context:
{context}

User Story:
{question}

Respond with ONLY valid JSON (no markdown, no explanations):
{{
  "test_cases": [
    {{"id": "TC-001", "type": "positive", "title": "...", "priority": "high", "preconditions": ["..."], "steps": ["...", "...", "..."], "expected_result": "...", "quality_score": 8}}
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
