# Architecture Decisions

## Model Selection
- **Primary:** Ollama Llama 3.2 3B (free, local)
- **Fallback:** OpenAI GPT-3.5-turbo (higher quality)
- **Reasoning:** Start free, upgrade if quality insufficient

## Output Format
- **Choice:** JSON with strict schema
- **Reasoning:** Easy to validate, parse, and integrate

## Quality Validation
- **Structural:** Pydantic models (all fields present)
- **Semantic:** LLM-as-judge (relevance to user story)
- **Coverage:** Count of test cases (min 3)
- **Reasoning:** Multi-layered validation catches different issues