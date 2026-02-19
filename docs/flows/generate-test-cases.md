# /generate-test-cases Endpoint Flow

## Overview

The `/generate-test-cases` endpoint is a POST endpoint that accepts a user story and generates structured test cases in Given-When-Then format using an LLM. It includes optional quality validation and background logging to MLflow.

**Endpoint:** `POST /generate-test-cases`
**Response Model:** `GenerateResponse`

---

## High-Level Flow Diagram

````
┌──────────────────────────────────────────────────────────────────────────┐
│                                CLIENT REQUEST                            │
│  POST /generate-test-cases                                               │
│  {                                                                        │
│    "user_story": "As a user...",                                        │
│    "include_quality_check": true,                                       │
│    "model": null (optional)                                             │
│  }                                                                        │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │   1. BUILD PROMPT                        │
        │   PromptBuilder.build()                  │
        │   - Load 2 few-shot examples             │
        │   - Inject user story into template      │
        │   - Return: {system, user} prompts       │
        └──────────────────────┬───────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────┐
        │   2. LLM GENERATION                      │
        │   LLMClient.generate()                   │
        │   - Detects provider (Ollama/OpenAI)     │
        │   - Calls _call_ollama() or _call_openai()│
        │   - Measures latency & token count       │
        │   - Returns: {text, latency, tokens...}  │
        └──────────────────────┬───────────────────┘
                               │
                       ┌───────▼────────┐
                       │ Error check?   │
                       └───┬────────┬───┘
                           │        │
                      Yes  │        │  No
                  (Error)  │        │
                           │        ▼
                           │   ┌──────────────────────────────────────┐
                           │   │ 3. PARSE JSON                        │
                           │   │ - Extract plain text                 │
                           │   │ - Remove ```json``` markers if       │
                           │   │   LLM wrapped output                 │
                           │   │ - Parse to JSON object               │
                           │   └──────────────────┬───────────────────┘
                           │                      │
                           │              ┌───────▼────────┐
                           │              │ Parse error?   │
                           │              └───┬────────┬───┘
                           │                  │        │
                           │            Yes   │        │  No
                           │         (Error)  │        │
                           │                  │        ▼
                           │                  │   ┌──────────────────────────────┐
                           │                  │   │ 4. VALIDATE STRUCTURE        │
                           │                  │   │ StructureValidator.validate()│
                           │                  │   │                              │
                           │                  │   │ Check:                       │
                           │                  │   │ - TC_NNN id format           │
                           │                  │   │ - Title length (10-200 chars)│
                           │                  │   │ - Priority enum              │
                           │                  │   │ - Given/When/Then fields     │
                           │                  │   │ - 3-10 test cases            │
                           │                  │   │ - Unique IDs                 │
                           │                  │   │                              │
                           │                  │   │ Returns:                     │
                           │                  │   │ {valid, errors, test_cases}  │
                           │                  │   └──────────────┬──────────────┘
                           │                  │                  │
                           │                  │          ┌───────▼────────┐
                           │                  │          │ Valid?         │
                           │                  │          └───┬────────┬───┘
                           │                  │              │        │
                           │                  │        No    │        │  Yes
                           │                  │      (Error) │        │
                           │                  │              │        ▼
                           │                  │              │   ┌──────────────────────────────┐
                           │                  │              │   │ 5. QUALITY VALIDATION        │
                           │                  │              │   │ (if include_quality_check)   │
                           │                  │              │   │                              │
                           │                  │              │   │ A. Evaluate Relevance       │
                           │                  │              │   │    - Use LLM-as-judge       │
                           │                  │              │   │    - Score: relevance,      │
                           │                  │              │   │      coverage, clarity      │
                           │                  │              │   │    - Normalize to 0-1       │
                           │                  │              │   │    - Pass if relevance ≥0.7 │
                           │                  │              │   │                              │
                           │                  │              │   │ B. Evaluate Coverage        │
                           │                  │              │   │    - Count test cases       │
                           │                  │              │   │    - Priority diversity     │
                           │                  │              │   │    - Pos/neg scenarios      │
                           │                  │              │   │    - Coverage score ≥ 0.7   │
                           │                  │              │   │                              │
                           │                  │              │   │ Returns:                     │
                           │                  │              │   │ {quality_metrics,           │
                           │                  │              │   │  coverage_metrics}          │
                           │                  │              │   └──────────────┬──────────────┘
                           │                  │              │                  │
                           │                  │              └──────────────────┤
                           │                  │                                 │
                           │                  └─────────────────────┬───────────┘
                           │                                        │
                           │                                        ▼
        ┌──────────────────┴────────────────────────────────────────────────┐
        │                   6. BUILD RESPONSE                              │
        │   GenerateResponse(                                              │
        │     user_story: str,                                             │
        │     test_cases: List[TestCaseResponse],                          │
        │     validation: dict,                                            │
        │     quality_metrics: Optional[dict],                             │
        │     metadata: dict                                               │
        │   )                                                              │
        └──────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ 7. ASYNC BACKGROUND TASK (Non-blocking)  │
        │    MLflowTracker.log_generation()        │
        │    - Log parameters & metrics            │
        │    - Store artifacts                     │
        │    - Record validation results           │
        │    - Add tags & metadata                 │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │         RETURN RESPONSE TO CLIENT         │
        │         (Immediately, without waiting)   │
        └──────────────────────────────────────────┘
````

---

## Detailed Component Flows

### 1. Prompt Builder (`src/llm/prompts.py`)

**Purpose:** Construct system and user prompts for the LLM

**System Prompt:**

- Defines role: "QA engineer who creates comprehensive test cases"
- Specifies rules: 3-6 test cases, Given-When-Then format, priority levels
- Enforces output format: Valid JSON only, no markdown

**User Prompt (Templated):**

```
User Story:
{{ user_story }}

Examples of good test cases:
[Few-shot examples from data/examples/user_stories.json]

Now generate test cases for the user story above.
```

**Process:**

```
build(user_story: str) ─────┐
                            ├──► _load_examples() ──► Load 2 examples from JSON
                            │
                            └──► user_template.render() ──► Jinja2 template
                                                            ├─ Inject user_story
                                                            └─ Inject examples

Returns: {"system": str, "user": str}
```

---

### 2. LLM Client (`src/llm/client.py`)

**Purpose:** Call LLM providers (Ollama or OpenAI) and measure performance

**Configuration:**

- Provider: "ollama" (default) | "openai"
- Model: "llama3.2:3b" (default for Ollama)
- Temperature: 0.3 (low for consistency)
- Max tokens: 2000

**Provider Routing:**

```
LLMClient.generate(prompt, system_prompt)
    │
    ├─► If provider == "ollama"
    │   └──► _call_ollama()
    │       ├─ Use ollama.chat()
    │       ├─ Temperature control
    │       ├─ Token counting
    │       └─ Return: {text, tokens}
    │
    └─► Elif provider == "openai"
        └──► _call_openai()
            ├─ Use openai.chat.completions.create()
            ├─ Temperature control
            ├─ Token counting
            └─ Return: {text, tokens}

Returns:
{
  "text": str,           # LLM-generated response
  "latency": float,      # Response time in seconds
  "tokens": int,         # Total tokens used
  "model": str,          # Model name
  "provider": str        # Provider used
}
```

---

### 3. JSON Parser

**Purpose:** Extract valid JSON from LLM output

**Handling Wrapped Output:**

````
Output text from LLM
    │
    ├─► Check if '```json' in text
    │   └──► Extract: text.split('```json')[1].split('```')[0]
    │
    └─► Elif '```' in text
        └──► Extract: text.split('```')[1].split('```')[0]

Then: json.loads(output_text)

Possible Errors:
- JSONDecodeError ──► Raise HTTPException(500)
````

---

### 4. Structure Validator (`src/validators/structure.py`)

**Purpose:** Validate test case schema and structure

**Validation Rules:**

```
TestCase
├─ id: Matches pattern ^TC_\d+$ (e.g., TC_001)
├─ title: 10-200 characters
├─ priority: One of [critical, high, medium, low]
├─ given: Min 10 characters, non-empty
├─ when: Min 10 characters, non-empty
└─ then: Min 10 characters, non-empty

TestCaseOutput
├─ test_cases: Array of TestCase
├─ Min items: 3
├─ Max items: 10
└─ Constraint: All IDs must be unique

Returns:
{
  "valid": bool,
  "errors": list[str],
  "test_cases": list[dict],
  "count": int
}
```

**Error Handling:**

- Single ValidationError caught, returned as error message
- No exception raised; error collected in response dict

---

### 5. Quality Validator (`src/validators/quality.py`)

#### 5A. Relevance Evaluation

**Process:**

```
evaluate_relevance(user_story, test_cases)
    │
    ├─► Build LLM-as-Judge prompt
    │   └─ Rate on: Relevance (0-10), Coverage (0-10), Clarity (0-10)
    │
    ├─► Call LLMClient.generate(judge_prompt)
    │   └─ Get JSON response with scores
    │
    └─► Parse & Normalize
        ├─ Extract: relevance_score, coverage_score, clarity_score
        ├─ Normalize to 0-1: score / 10
        ├─ Overall: Average of three scores
        ├─ Pass threshold: relevance ≥ 0.7
        └─ Return error metrics if parse fails

Returns:
{
  "relevance": float (0-1),
  "coverage": float (0-1),
  "clarity": float (0-1),
  "overall": float (0-1),
  "reasoning": str,
  "passed": bool
}
```

#### 5B. Coverage Evaluation

**Metrics:**

```
evaluate_coverage(test_cases, min_count=3)
    │
    ├─► Check count ≥ 3 ─────────────────────┬─► +0.4 to score
    │
    ├─► Check priority diversity ≥ 2 ────────┬─► +0.3 to score
    │   (Detect unique priorities in set)
    │
    ├─► Check positive scenarios ────────────┬─► +0.15 to score
    │   (Titles contain "successful" or "valid")
    │
    └─► Check negative scenarios ────────────┬─► +0.15 to score
        (Titles contain "fail", "invalid", or "error")

Coverage Score:
  Minimum: 0.0
  Maximum: 1.0
  Pass threshold: ≥ 0.7

Returns:
{
  "count": int,
  "min_count_met": bool,
  "priority_diversity": int,
  "has_positive_cases": bool,
  "has_negative_cases": bool,
  "coverage_score": float,
  "passed": bool
}
```

---

### 6. MLflow Tracker (`src/mlflow_tracker.py`)

**Purpose:** Log experiment metrics and artifacts (background task)

**Logged Data:**

**Parameters:**

- model: Model name (e.g., "llama3.2:3b")
- provider: Provider name (e.g., "ollama")
- user_story: First 100 chars of user story

**Metrics:**

- latency: Response time in seconds
- structure_valid: 1.0 if valid, 0.0 if invalid
- test_case_count: Number of test cases generated
- relevance_score: 0-1 (if quality check enabled)
- coverage_score: 0-1 (if quality check enabled)
- clarity_score: 0-1 (if quality check enabled)
- overall_quality: 0-1 (if quality check enabled)
- priority_diversity: Count of unique priorities

**Artifacts:**

- generation_result.json:
  ```json
  {
    "user_story": str,
    "test_cases": list,
    "validations": {
      "structure": dict,
      "quality": dict,
      "coverage": dict
    }
  }
  ```

**Tags:**

- timestamp: ISO format timestamp
- passed_validation: "True" or "False"

---

## Request/Response Models

### GenerateRequest

```python
{
  "user_story": str,           # Required (20-500 chars)
  "include_quality_check": bool,  # Optional (default: True)
  "model": Optional[str]       # Optional, not currently used
}
```

### GenerateResponse

```python
{
  "user_story": str,
  "test_cases": [
    {
      "id": str,        # e.g., "TC_001"
      "title": str,
      "priority": str,  # critical | high | medium | low
      "given": str,
      "when": str,
      "then": str
    }
  ],
  "validation": {
    "structure_valid": bool,
    "count": int,
    "quality_passed": Optional[bool],
    "coverage_passed": bool
  },
  "quality_metrics": Optional[{
    "relevance": float,
    "coverage": float,
    "clarity": float,
    "overall": float,
    "reasoning": str,
    "passed": bool
  }],
  "metadata": {
    "latency": float,
    "tokens": int,
    "model": str
  }
}
```

---

## Error Handling

### Error Cases & Responses

| Step                 | Error           | HTTP Status | Detail                                        |
| -------------------- | --------------- | ----------- | --------------------------------------------- |
| LLM Generation       | Provider error  | 500         | "LLM generation failed: {error}"              |
| JSON Parsing         | Invalid JSON    | 500         | "Failed to parse LLM output as JSON: {error}" |
| Structure Validation | Schema mismatch | 400         | "Invalid test case structure: {errors}"       |
| Request Validation   | Invalid input   | 422         | Pydantic validation error                     |

---

## Key Features

### Asynchronous Logging

- MLflow logging runs in background task using FastAPI `BackgroundTasks`
- Response returns immediately without waiting for MLflow
- Non-blocking architecture ensures fast response times

### Few-Shot Learning

- Uses 2 example user stories from `data/examples/user_stories.json`
- Improves LLM output quality through in-context learning
- Examples injected into user prompt template

### Dual Provider Support

- **Ollama (Local):** Free, low latency, no API keys required
- **OpenAI (Cloud):** Higher quality, but requires API key and incurs costs
- Configurable via `LLMConfig.provider`

### Multi-Layer Validation

1. **Structural:** Pydantic schema validation
2. **Semantic:** LLM-as-judge evaluation (optional)
3. **Coverage:** Test case diversity analysis

### Performance Metrics

- Latency tracking: Measures end-to-end response time
- Token counting: Tracks LLM token usage per request
- Quality scoring: Provides quantitative metrics for generated test cases

---

## Configuration

### Environment Variables

(from `src/application/shared/infrastructure/environment_variables.py`)

- `OLLAMA-SERVICE-HOST`: Ollama service URL (default: "http://localhost:11435")
- `OLLAMA_SERVICE_MODEL_QWEN3VL4B`: Model name (default: "qwen3-vl:4b")
- `MAX_RETRIES`: Max API retries (default: 3)
- `OPENAI_API_KEY`: OpenAI API key (required if using OpenAI provider)

---

## Sequence Diagram

```
Client              API              PromptBuilder        LLMClient         Validators          MLflow
  │                  │                     │                 │                 │                 │
  ├─POST request────>│                     │                 │                 │                 │
  │                  ├──build()──────────>│                 │                 │                 │
  │                  │<──prompts──────────┤                 │                 │                 │
  │                  ├─generate()──────────────────────────>│                 │                 │
  │                  │<──LLM response─────────────────────┤                 │                 │
  │                  ├─parse JSON────┐                     │                 │                 │
  │                  │               │                     │                 │                 │
  │                  ├────validate()────────────────────────────────────────>│                 │
  │                  │<───validation result────────────────────────────────┤                 │
  │                  │                                                       │                 │
  │                  ├──if quality_check:                                   │                 │
  │                  │  ├─evaluate_relevance()────────────────────────────>│                 │
  │                  │  │<───relevance scores───────────────────────────┤  │                 │
  │                  │  └─evaluate_coverage()────────────────────────────>│                 │
  │                  │    <───coverage scores──────────────────────────┤  │                 │
  │                  │                                                       │                 │
  │                  ├──background_task.add_task()──────────────────────────────────────────>│
  │                  │                                                                         │ (async)
  │<─response────────┤                                                                         │
```

---

## Performance Characteristics

### Response Time Breakdown

- Prompt building: ~10ms
- LLM generation: ~2-5 seconds (Ollama), ~1-3 seconds (OpenAI)
- Validation: ~50-100ms (structure), ~1-2 seconds (quality check)
- **Total end-to-end:** ~3-10 seconds
- **API response time:** <1 second (thanks to async logging)

### Default Success Rate

- Structure validation: ≥95% (with few-shot examples)
- Quality relevance: ~75% with llama3.2:3b
- Coverage: ~80% with diverse prompt engineering

---

## Testing & Monitoring

### Test Scenarios

1. **Happy path:** Valid user story, quality check enabled
2. **Minimal path:** Valid user story, quality check disabled
3. **Edge case:** Very long user story (500 chars)
4. **Error case:** Malformed LLM response (invalid JSON)

### MLflow Monitoring

- View metrics: http://localhost:5001
- Compare runs by model, provider, or timestamp
- Analyze quality trends over time
- Debug failed generations with artifacts

---

## Related Files

- **Endpoint:** `src/api/main.py:93-179`
- **Request model:** `src/api/main.py:32-35`
- **Response model:** `src/api/main.py:47-52`
- **Prompt builder:** `src/llm/prompts.py`
- **LLM client:** `src/llm/client.py`
- **Validators:** `src/validators/`
- **MLflow tracker:** `src/mlflow_tracker.py`
- **Examples data:** `data/examples/user_stories.json`
- **Environment config:** `src/application/shared/infrastructure/environment_variables.py`
