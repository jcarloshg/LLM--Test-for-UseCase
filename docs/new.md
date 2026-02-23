# LLM Test Case Generation - Use Case Framework

This project implements a microservice that leverages Large Language Models (LLMs) to automatically generate structured test cases from user stories and project requirements. Using prompt engineering and validation techniques, the service produces JSON-formatted test artifacts with clear acceptance criteria, preconditions, and test steps.

## Table of Contents

### LLMOps Framework (9 Phases)

1. [Phase 1: Problem Definition & Use Case Design](#phase-1-problem-definition--use-case-design)
2. [Architecture Decisions](#architecture-decisions)
3. [Phase 2: Data Collection & Preparation](#phase-2-data-collection--preparation)
4. [Phase 3: Model Selection & Evaluation](#phase-3-model-selection--evaluation)
5. [Phase 5: RAG & Prompting](#phase-5-rag--prompting)
6. [Phase 6: Evaluation & Testing](#phase-6-evaluation--testing)
7. [Phase 7: Deployment & Serving](#phase-7-deployment--serving)
8. [Phase 8: Monitoring & Observability](#phase-8-monitoring--observability)
9. [Phase 9: Feedback & Iteration](#phase-9-feedback--iteration)

### Project Organization

- [Project Structure](#project-structure)
- [File System Organization](#file-system-organization)
  - [Root Level Files](#root-level-files)
  - [Data (`data/`)](#data-data)
  - [Docker (`docker/`)](#docker-docker)
  - [Logs (`logs/`)](#logs-logs)

### Setup & Deployment

- [Getting Started](#getting-started)
- [Next Steps](#next-steps)

## Phase 1: Problem Definition & Use Case Design

### Objective

Clearly define what problem you're solving and whether an LLM is the right solution. Establish success criteria, constraints (budget, latency, privacy), and scope to prevent costly mistakes later. Determine if you need simple prompting, RAG (Retrieval-Augmented Generation), or fine-tuning.

### Key Activities

- **Define specific use case and expected outcomes** - Generate valid, structured test cases from user stories using prompt engineering
- **Identify success metrics (accuracy, latency, cost per request)**
  - Structural compliance (JSON parsing success rate >95%)
  - Quality score tracking
  - Response latency <5 seconds (P90)
  - Infrastructure cost per 1,000 requests
- **Assess if LLM is appropriate vs. simpler solutions** - LLM chosen for natural language understanding and creative test case generation
- **Decide build vs. buy (API vs. self-hosted models)** - Build approach: Deploy and run all models locally using Ollama for complete control and privacy

### Success Metrics & Constraints

#### 1. Precisión y Calidad de la Salida (Accuracy & Quality)

Since the objective is to generate valid and structured test artifacts, accuracy is measured through rule compliance and output utility rather than traditional classification metrics.

| Metric                                                | Details                                                                                                                                  | Target   |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **Structural Compliance (JSON Parsing Success Rate)** | Percentage of LLM responses that parse successfully as valid JSON on first attempt, without triggering retry mechanisms                  | >95%     |
| **Average Quality Score**                             | API returns a quality score evaluating heuristics: empty preconditions, minimum logical steps in test cases, field completeness          | ≥4.0/5.0 |
| **Retry Rate**                                        | Frequency of retry invocations due to malformed responses or schema-breaking hallucinations. Indicates if prompt design needs adjustment | <5%      |

#### 2. Latencia y Rendimiento (Latency & Performance)

Local models (Llama 3, Mistral via Ollama) introduce specific latency challenges, especially without GPU acceleration in production environments.

| Metric | Details | Target |
| -- | | |
| **Total Endpoint Latency (P90 / P99)** | Time from receiving `POST /generate-tests` to returning JSON payload, including inference, parsing, and validation | P90 <5s, P99 <10s |
| **Validation Overhead** | Additional time for structural, heuristic, and LLM-as-judge validation vs. pure text generation | <20% of total latency |

#### 3. Coste por Solicitud y Eficiencia (Cost per Request)

With Ollama, inference cost per API token is $0, but cost shifts to compute infrastructure.

| Metric                                      | Details                                                                                                                                   | Approach                               |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **Infrastructure Cost per 1,000 Requests**  | Monthly server cost ÷ max request throughput without latency degradation                                                                  | Monitor via container resource usage   |
| **Resource Utilization (CPU / RAM / VRAM)** | Track Docker container consumption during peak load. Justify trade-offs (e.g., 4-bit quantized models reduce RAM at slight accuracy cost) | Log CPU/RAM usage, establish baselines |

## Phase 2: Data Collection & Preparation

### Objective

Gather, clean, and organize data needed for your LLM application—including examples for prompts, evaluation datasets, and training data if fine-tuning. Quality over quantity is critical; 100 high-quality examples often beat 1,000 noisy ones.

### Key Activities

- **Collect representative input-output examples** - Gather diverse user stories covering different features, difficulty levels, and user personas
- **Create evaluation datasets with ground truth labels** - Pair user stories with manually validated test cases as reference outputs
- **Clean and anonymize data (remove PII)** - Ensure no sensitive information is exposed in stories or test data
- **Structure data in appropriate formats (JSONL, CSV, JSON)** - Organize data for easy loading and validation
- **Version datasets for reproducibility** - Track dataset changes and maintain audit trail for experiment tracking

### Dataset Overview

Currently, the project includes **49 diverse user stories** across an e-commerce platform:

| Dataset                   | Location                                          | Contents                                                              | Size        |
| ------------------------- | ------------------------------------------------- | --------------------------------------------------------------------- | ----------- |
| **Test Stories**          | `data/test/user_stories.json`                     | 49 user stories with difficulty levels (easy/medium/hard)             | 1,097 bytes |
| **Ground Truth Examples** | `data/examples/user_stories_with_test_cases.json` | 2 user stories with complete test case generations and quality scores | ~5+ KB      |

### Data Distribution

**User Story Difficulty Levels:**

- **Easy:** Product recommendations, account data export, invoice printing, voice search, wishlists, etc.
- **Medium:** Shopping cart, search, inventory management, notifications, checkout, product variants, etc.
- **Hard:** Login/authentication, password reset, profile updates, two-factor authentication, secure payment, etc.

**Difficulty Breakdown (49 total stories):**

- Easy: ~15 stories
- Medium: ~27 stories
- Hard: ~7 stories

### Ground Truth Test Cases

The examples include complete test case sets with:

- **Test Case Types:** Positive, negative, and edge-case scenarios
- **Preconditions:** Prerequisites for test execution
- **Steps:** Sequential test actions
- **Expected Results:** Detailed assertions
- **Quality Scores:** Assessed outputs (0.88-0.92 range)
- **Priority Levels:** High, medium, low test prioritization

**Example**: US_001 (Login functionality) has 3 test cases:

- TC_001: Successful login with valid credentials (positive)
- TC_002: Login fails with invalid password (negative)
- TC_003: Login fails with non-existent email (negative)

### Tools & Technologies

- **Pydantic** - Data validation with schema enforcement
- **JSON Schema Validator** - Validate test case structure
- **DVC (Data Version Control)** - Version datasets for reproducibility (ToDo)
- **Label Studio** - Manual annotation and quality review (ToDo)

## Phase 3: Model Selection & Evaluation

### Objective

Evaluate different open-source LLM options to find the best fit for your use case by testing on your specific data. Consider model size, inference latency, memory requirements, and quality of outputs for local deployment. Focus on privacy-first, cost-effective models running on Ollama.

### Key Activities

- **Compare models on evaluation dataset** - Run the evaluation set through multiple local models and compare outputs
- **Test different model sizes** - Evaluate lightweight (1B), balanced (3B), and higher-quality (8B) models from the Meta Llama family
- **Measure performance metrics relevant to use case** - Evaluate based on structural compliance, quality scores, and semantic accuracy
- **Evaluate latency and resource usage** - Calculate inference time and memory requirements for each model
- **Document model comparison in decision matrix** - Create records for transparent decision-making and future reference

### Model Candidates

#### Open-Source Models (Local / Self-hosted)

| Model             | Provider | Size          | Image              | Latency | Cost     | Notes                                                                       |
| ----------------- | -------- | ------------- | ------------------ | ------- | -------- | --------------------------------------------------------------------------- |
| **Llama 3.2 1B**  | Meta     | 1B parameters | `llama3.2:1b`      | ~1-2s   | $0/token | ✅ **Selected Choice** - Ultra-lightweight, fastest inference, minimal VRAM |
| **Llama 3.2 3B**  | Meta     | 3B parameters | `llama3.2:3b`      | ~2-5s   | $0/token | ✅ **Primary Choice** - Good balance of speed, quality, privacy             |
| Llama 3 ChatQA 8B | Meta     | 8B parameters | `llama3-chatqa:8b` | ~4-8s   | $0/token | Higher quality, slower, better reasoning, requires more VRAM                |

### Evaluation Methodology

#### 1. Performance Metrics

| Metric                        | How to Measure                               | Success Criteria    | Tool                       |
| ----------------------------- | -------------------------------------------- | ------------------- | -------------------------- |
| **Structural Compliance**     | % of valid JSON outputs on first attempt     | >95%                | Pydantic validator         |
| **Quality Score**             | Average heuristic-based quality assessment   | ≥4.0/5.0            | Custom LLM-as-judge prompt |
| **Test Case Count**           | Average number of test cases per story       | ≥3 per story        | Count validation           |
| **Semantic Relevance**        | LLM evaluation of story-to-tests alignment   | ≥4.0/5.0            | Secondary LLM evaluation   |
| **Precondition Completeness** | % of test cases with non-empty preconditions | ≥90%                | Schema validation          |
| **Step Clarity**              | Average steps per test case                  | 2-5 steps (optimal) | Count analysis             |

#### 2. Latency Benchmarks

Test on the evaluation dataset with multiple story lengths:

```
Benchmark Setup:
- Short stories (1 sentence)
- Medium stories (2-3 sentences)
- Long stories (4+ sentences)
- Peak load (10 concurrent requests)

Measure:
- Time to first token (TTF)
- End-to-end latency (P50, P90, P99)
- Throughput (requests/second)
```

#### 3. Cost Analysis

**Cost Calculation Formula:**

```
Cost per Request = (Input Tokens + Output Tokens) × Model Rate
Infrastructure Cost = Monthly Server Cost ÷ Requests/Month

Total Cost of Ownership = API Cost + Infrastructure Cost
```

**Example Comparison (1,000 requests/month):**

| Model                 | Avg Input Tokens | Avg Output Tokens | Cost per 1K | Monthly API Cost | Infrastructure | Total   |
| --------------------- | ---------------- | ----------------- | ----------- | ---------------- | -------------- | ------- |
| **Llama 3.2 1B**      | 800              | 400               | $0          | $0               | $30-50         | $30-50  |
| **Llama 3.2 3B**      | 800              | 400               | $0          | $0               | $50-100        | $50-100 |
| **Llama 3 ChatQA 8B** | 800              | 400               | $0          | $0               | $80-150        | $80-150 |

### Evidence RAG vs Prompt

![alt text](docs/resource/img/mlflow.png)
![alt text](docs/resource/img/mlflow_01.png)

## Phase 5: RAG & Prompting

### Objective

Design and implement prompt engineering strategies and retrieval-augmented generation (RAG) pipelines that provide the LLM with necessary context and guidance to generate accurate, comprehensive test cases. Choose the right approach (direct prompting vs. RAG) based on your data and use case requirements.

### Key Activities

- **Design effective prompts** - Craft prompt templates that clearly guide the LLM to generate well-structured test cases with preconditions, steps, and expected results
- **Implement RAG pipelines** - Create retrieval systems that provide relevant examples and context when generating test cases for complex user stories
- **Optimize prompts through iteration** - Test and refine prompts based on output quality, adjusting templates to improve test case comprehensiveness and accuracy
- **Handle edge cases** - Engineer prompts to handle ambiguous user stories, incomplete requirements, and edge case scenarios
- **Balance latency and quality** - Trade off between simple prompting for speed and RAG-enhanced generation for better context awareness

### Implementation Approaches

#### 1. Direct Prompting Strategy

**Use Case:** Simple to moderately complex user stories that don't require external context or examples

**Implementation:** `src/application/create_tests/infra/executable_chain/executable_chain_prompting.py`

```python
class ExecutableChainPrompting(ExecutableChain):
    """Direct prompting without retrieval-augmented generation.

    Suitable for tasks that don't require context retrieval.
    Simple, fast execution without external knowledge retrieval.
    """

    def execute(self, prompt: str, max_retries: int = 3) -> ExecutableChainResponse:
        # Chain: PromptTemplate → LLM → RobustJsonOutputParser
        # Creates simple pipeline: prompt → inference → validated JSON
```

**Advantages:**

- Lower latency (no retrieval overhead)
- Simpler architecture (no vector store dependencies)
- Faster response times for straightforward user stories
- Easier to debug and understand

**Disadvantages:**

- Limited context awareness
- May struggle with complex or ambiguous stories
- No access to examples or domain knowledge
- Higher retry rate for edge cases

**Configuration:**

- Prompt template guides model behavior
- Retry logic handles JSON parsing failures
- Max retries: 3 attempts for structure validation

#### 2. RAG-Enhanced Strategy

**Use Case:** Complex user stories, domain-specific requirements, or situations requiring example-based generation

**Implementation:** `src/application/create_tests/infra/executable_chain/executable_chain_rag.py`

```python
class ExecutableChainRAG(ExecutableChain):
    """RAG pattern with retrieval-augmented generation.

    Provides context retrieval from vector stores and example-based generation.
    Includes caching for performance optimization.
    """

    def execute(self, prompt: str, max_retries: int = 3) -> ExecutableChainResponse:
        # Chain: Retriever (cached) → PromptTemplate + Context → LLM → RobustJsonOutputParser
        # Augments prompt with retrieved examples and context before inference
```

**Advantages:**

- Better context awareness with retrieved examples
- Handles complex and ambiguous stories better
- Can ground generation in existing test cases
- Higher quality output for domain-specific needs

**Disadvantages:**

- Higher latency due to retrieval step
- Requires vector store setup and maintenance
- Dependency on retrieval quality
- More complex debugging

**Core Features:**

- **Cached Retrieval:** `_cached_retrieve()` method with LRU cache (max 100 items) reduces repeated retrievals
- **Context Formatting:** Retrieved documents formatted as context for the LLM
- **Validation & Retry:** Same structure validation as direct prompting with automatic retries
- **Async Support:** `execute_async()` for batch processing with concurrent rate limiting (max 3 concurrent by default)
- **Cache Management:** `get_cache_stats()` and `clear_cache()` methods for monitoring and maintenance

**Configuration:**

- Vector store retriever for semantic search
- RAG cache with configurable size
- Concurrent execution limits for batch operations
- Retry logic identical to direct prompting (max 3 attempts)

### Choosing Your Approach

| Factor                 | Direct Prompting                | RAG-Enhanced                             |
| ---------------------- | ------------------------------- | ---------------------------------------- |
| **Latency**            | <1s                             | 2-5s                                     |
| **Best For**           | Simple stories, quick responses | Complex stories, domain knowledge needed |
| **Setup Complexity**   | Low                             | Medium-High                              |
| **Accuracy**           | Good (80-90%)                   | Better (90-95%)                          |
| **Cost**               | Lower (no retrieval)            | Slightly higher (retrieval overhead)     |
| **Example Dependency** | Prompt-based only               | Retriever-based context                  |
| **Cache Benefits**     | N/A                             | Significant for repeated queries         |

### Common Patterns

**Pattern 1: Hybrid Approach**

```
1. Try direct prompting first
2. If quality score < threshold, fall back to RAG
3. Cache RAG results for similar future queries
```

**Pattern 2: Progressive Enhancement**

```
1. Start with simple prompting templates
2. Monitor failure rates and retry counts
3. Gradually add RAG for frequently failing story types
4. Use cached results to improve latency
```

**Pattern 3: Context-Aware Routing**

```
1. Analyze incoming user story complexity
2. Simple stories → Direct prompting
3. Complex stories → RAG with cached retriever
4. Monitor and adapt thresholds based on quality metrics
```

### Validation & Error Handling

Both implementations include robust validation:

- **Structure Validation:** Ensures response contains `test_cases` array with valid test case objects
- **JSON Parsing:** Automatic retry on invalid JSON responses
- **Type Checking:** Validates response is dictionary type (not string or list)
- **Attempt Tracking:** Logs retry attempts and reasons for debugging

### Outputs

- **Prompt templates:** Optimized prompts guiding test case generation
- **RAG pipeline:** Vector store setup with retriever and caching infrastructure
- **Validation schemas:** Test case structure definitions for output validation
- **Performance metrics:** Latency, retry rates, and cache effectiveness statistics

## Phase 6: Evaluation & Testing

### Objective

Systematically measure if your LLM application meets quality standards before production deployment. Use both automated metrics and human evaluation to assess accuracy, relevance, safety, and consistency. Testing catches issues early and provides benchmarks for measuring improvements.

### Key Activities

- **Run automated evaluations on test dataset** - Execute evaluation scripts against held-out test stories to measure structural and semantic quality
- **Implement LLM-as-judge for qualitative assessment** - Use a separate LLM prompt to evaluate test case relevance, completeness, and alignment with user stories
- **Conduct human evaluation samples** - Have QA specialists review and score random samples of generated test cases
- **Test edge cases and adversarial inputs** - Evaluate model behavior on edge cases (vague stories, incomplete requirements, conflicting criteria)
- **Measure cost, latency, and throughput** - Profile inference time, resource consumption, and sustained throughput under load

### Evaluation Framework

#### 1. Automated Metrics

| Metric                      | Definition                                         | How to Measure                                        | Target       | Tool                                                       |
| --------------------------- | -------------------------------------------------- | ----------------------------------------------------- | ------------ | ---------------------------------------------------------- |
| **Structural Compliance**   | % of valid JSON outputs parseable on first attempt | Count successful parses ÷ total requests              | >95%         | Pydantic validator                                         |
| **Field Completeness**      | % of test cases with all required fields non-empty | Check preconditions, steps, expected_result populated | >90%         | Schema validator                                           |
| **Test Case Count**         | Average number of test cases generated per story   | Sum all test cases ÷ number of stories                | ≥3 per story | Count analysis                                             |
| **Precondition Relevance**  | Quality of preconditions (not generic/empty)       | Automated script evaluation                           | ≥4.0/5.0     | `src/application/evaluate_models/model/quality_tracker.py` |
| **Step Specificity**        | Steps are concrete and measurable (not vague)      | Automated script scoring                              | ≥4.0/5.0     | `src/application/evaluate_models/model/quality_tracker.py` |
| **Expected Result Clarity** | Assertions are clear and verifiable                | Automated script evaluation                           | ≥4.0/5.0     | `src/application/evaluate_models/model/quality_tracker.py` |

#### 2. Human Evaluation Protocol

**Sample Selection:**

- Random sampling: 10-20% of test dataset (5-10 stories)
- Stratified sampling: 2-3 stories per difficulty level (easy/medium/hard)
- Edge case samples: Stories with unusual requirements or complex scenarios

**Evaluation Rubric (1-5 scale):**

| Criterion        | 1 (Poor)                     | 3 (Acceptable)                         | 5 (Excellent)                           |
| ---------------- | ---------------------------- | -------------------------------------- | --------------------------------------- |
| **Relevance**    | Test case unrelated to story | Covers main feature but misses details | Fully addresses all story aspects       |
| **Completeness** | Missing test scenarios       | Has positive + negative cases          | Includes positive, negative, edge cases |
| **Clarity**      | Steps are vague/confusing    | Steps are mostly clear                 | Steps are precise, specific, actionable |
| **Usability**    | Tester cannot execute        | Tester needs clarification             | Ready to execute, fully self-contained  |

**Success Criteria:**

- Average human score ≥4.0/5.0 for all criteria
- ≥90% inter-rater agreement on pass/fail decisions

#### 3. Edge Case Testing

Test model behavior on challenging scenarios:

```
Edge Case Categories:
1. Vague Requirements - "As a user, I want better performance"
2. Conflicting Criteria - Requirements that contradict each other
3. Multi-Role Stories - Stories with multiple actors/roles
4. Non-Functional Requirements - Stories focused on performance, security, or compliance
5. Complex User Flows - Stories with many sequential steps
6. Boundary Conditions - Stories with specific limits or constraints
```

**Test Examples:**

- Very short stories (1 sentence)
- Very long stories (10+ sentences)
- Stories with technical jargon
- Stories without clear acceptance criteria
- Stories mentioning external systems/APIs

#### 4. Performance Benchmarks

| Metric                     | Target      | Measurement                     |
| -------------------------- | ----------- | ------------------------------- |
| **P50 Latency**            | <3s         | Median response time            |
| **P90 Latency**            | <5s         | 90th percentile                 |
| **P99 Latency**            | <10s        | 99th percentile                 |
| **Throughput**             | >10 req/min | Single instance sustained load  |
| **Memory Usage**           | <4GB        | Peak RAM during inference       |
| **Cost per 1,000 Stories** | <$100       | Infrastructure cost calculation |

### Tools & Technologies

- **DeepEval** - LLM evaluation framework with pre-built metrics (ToDo)
- **Pydantic** - Schema validation and structural compliance checking
- **Load Testing Tools** (locust, wrk) - Latency and throughput benchmarking (ToDo)

## Phase 7: Deployment & Serving

### Objective

Transition your LLM application from development to production by integrating into your architecture, setting up APIs, implementing security, and ensuring scalability. Includes both technical infrastructure and user-facing interfaces.

### Key Activities

- **Create API endpoints with authentication** - Design RESTful API with token-based auth (JWT/API keys)
- **Implement rate limiting and caching** - Prevent abuse, reduce latency with response caching
- **Set up load balancing and auto-scaling** - Distribute load across multiple instances, scale based on demand
- **Configure security measures (input validation, filtering)** - Validate inputs, implement prompt injection defenses
- **Stage rollout (beta → full production)** - Test with limited users first, monitor metrics, gradually roll out

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client Applications                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    HTTP/REST API
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                    API Gateway / Load Balancer                  │
│  (Rate Limiting, Authentication, Request Routing)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼───┐          ┌───▼───┐          ┌───▼───┐
    │ API   │          │ API   │          │ API   │
    │Pod 1  │          │Pod 2  │          │Pod 3  │
    ││          ││          ││
    │Cache │          │Cache │          │Cache │
    └───┬───┘          └───┬───┘          └───┬───┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼────────┐    ┌───▼────────┐    ┌───▼────────┐
    │ Ollama     │    │ Ollama     │    │ Ollama     │
    │Instance 1  │    │Instance 2  │    │Instance 3  │
    │(1B Model)  │    │(1B Model)  │    │(1B Model)  │
    └────────────┘    └────────────┘    └────────────┘
```

### API Design

#### Endpoint Specification

**Endpoint:** `POST /generate-tests`

```python
# Request
{
  "user_story": "As a customer, I want to search for products by category so that I can find items quickly",
  "difficulty": "medium",  # optional: easy, medium, hard
  "num_test_cases": 4,     # optional: default 3
  "model": "llama3.2:1b"   # optional: which model to use
}

# Response (Success - 200)
{
  "request_id": "req_12345",
  "story_id": "US_001",
  "model_used": "llama3.2:1b",
  "processing_time_ms": 2850,
  "test_cases": [
    {
      "id": "TC_001",
      "type": "positive",
      "title": "Search by valid category",
      "priority": "high",
      "preconditions": ["User is logged in", "Product database is populated"],
      "steps": [
        "Click search bar",
        "Select category filter",
        "Click search"
      ],
      "expected_result": "Products matching category are displayed"
    },
    # ... more test cases
  ],
  "quality_score": 0.92,
  "success": true
}

# Response (Error - 400)
{
  "error": "Invalid request",
  "message": "User story cannot be empty",
  "request_id": "req_12345",
  "success": false
}
```

#### Other Endpoints

| Endpoint          | Method | Purpose                                        | Auth Required      |
| ----------------- | ------ | ---------------------------------------------- | ------------------ |
| `/generate-tests` | POST   | Generate test cases from user story            | Yes (API key)      |
| `/health`         | GET    | Service health check                           | No                 |
| `/metrics`        | GET    | Prometheus metrics (requests, latency, errors) | No (internal only) |
| `/models`         | GET    | List available models and their status         | Yes                |
| `/version`        | GET    | API version information                        | No                 |
| `/estimate-cost`  | POST   | Estimate cost/latency before generation        | Yes                |

### Security Implementation

#### Input Validation

```python
# Validate and sanitize user story input
class UserStoryInput(BaseModel):
    user_story: str = Field(..., min_length=10, max_length=2000)
    difficulty: Optional[str] = Field(default="medium", pattern="^(easy|medium|hard)$")
    num_test_cases: Optional[int] = Field(default=3, ge=1, le=10)

    @validator('user_story')
    def validate_story(cls, v):
        # Remove potentially harmful characters
        # Check for prompt injection patterns
        if any(prompt_injection_pattern in v for pattern in DANGEROUS_PATTERNS):
            raise ValueError("Input contains suspicious patterns")
        return v
```

#### Authentication & Authorization

- **API Key Authentication:** Each client gets unique API key
- **JWT Tokens:** For web clients with session management
- **Rate Limiting:**
  - Default: 10 requests/minute per key
  - Premium: 100 requests/minute per key
  - Burst: Allow up to 5 concurrent requests

#### Prompt Injection Defense

```python
# Detect and block prompt injection attempts
DANGEROUS_PATTERNS = [
    "ignore previous instructions",
    "system:",
    "prompt:",
    "forget everything",
    "do the opposite"
]

# Input filtering
def sanitize_input(user_story: str) -> str:
    # Remove control characters
    story = ''.join(c for c in user_story if c.isprintable())
    # Limit consecutive special characters
    story = re.sub(r'([^\w\s]){3,}', r'\1\1', story)
    return story
```

### Performance Optimization

#### Caching Strategy

| Cache Level        | Implementation     | TTL        | Key                                 |
| ------------------ | ------------------ | ---------- | ----------------------------------- |
| **Response Cache** | Redis              | 1 hour     | `{story_hash}_{model}_{difficulty}` |
| **Model Cache**    | In-memory          | Persistent | Model weights in VRAM               |
| **Template Cache** | Application memory | Persistent | Prompt templates                    |

**Cache Hit Logic:**

```python
cache_key = hash(story + model + difficulty)
if cache.exists(cache_key):
    return cache.get(cache_key)  # Return cached result
else:
    result = generate_tests(story)
    cache.set(cache_key, result, ttl=3600)  # Cache for 1 hour
    return result
```

#### Inference Optimization

- **Model Quantization:** Use 4-bit quantization to reduce VRAM
- **Batch Processing:** Queue requests and process in batches
- **Token Streaming:** Return results as they're generated (SSE)
- **Hardware Acceleration:** Use GPU when available

### Deployment Workflow

#### 1. Containerization

```dockerfile
FROM ollama/ollama:latest

# Copy application code
COPY src/ /app/src/
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r /app/requirements.txt

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI application
CMD ["python", "/app/src/main.py"]
```

### Monitoring & Observability

Complete production monitoring and observability setup for the LLM test case generation microservice. Tracks system health, performance, quality metrics, and business outcomes.

#### 1. Real-Time Metrics

**Key Performance Indicators (KPIs):**

| Metric                   | Type                      | Target             | Alert Threshold |
| ------------------------ | ------------------------- | ------------------ | --------------- |
| **Request Latency**      | Histogram (P50, P90, P99) | P90 <5s            | P90 >5s         |
| **Throughput**           | Counter (req/sec)         | 10+ req/sec        | <5 req/sec      |
| **Success Rate**         | Gauge                     | >95%               | <95%            |
| **Error Rate**           | Counter                   | <1%                | >2%             |
| **Queue Depth**          | Gauge                     | <10 items          | >20 items       |
| **Cache Hit Rate**       | Gauge                     | >70%               | <60%            |
| **Model Inference Time** | Histogram                 | <3s (1B), <5s (3B) | Exceeds target  |
| **Quality Score**        | Gauge (avg)               | ≥4.0/5.0           | <3.8/5.0        |

#### 2. Dashboards

**Main Dashboard (Real-Time Overview):**

- Requests/min, P90 latency, error rate
- Quality score trend (24h, 7d, 30d)
- Test case generation success rate
- Model distribution and fallback usage
- Infrastructure resource usage

**Quality Dashboard:**

- Quality metrics by model and difficulty
- Test case count distribution
- Precondition/step/result quality scores
- User satisfaction ratings

**Cost Dashboard:**

- Daily cost trend
- Cost per request
- Cost by model and feature
- Monthly forecast vs. actual

#### 3. Distributed Tracing

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Use in endpoints
@app.post("/generate-tests")
async def generate_tests(request: UserStoryInput):
    with trace.get_tracer(__name__).start_as_current_span("generate_tests") as span:
        span.set_attribute("story_id", request.story_id)
        span.set_attribute("difficulty", request.difficulty)
        # ... implementation
```

#### 4. Performance Profiling

**Baseline Metrics (Collected Continuously):**

```python
# Profile each request
@app.middleware("http")
async def profile_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    # Record timing breakdown
    timings = {
        "input_validation": ...,
        "model_inference": ...,
        "output_validation": ...,
        "response_serialization": ...
    }

    metrics.record_request_profile(request.url.path, timings)
    return response
```

#### 5. Cost Tracking

**Real-Time Cost Monitoring:**

```python
# Track costs in real-time
async def track_cost(request_data: dict, response_data: dict):
    # Infrastructure cost
    infra_cost = calculate_infrastructure_cost(
        duration_ms=response_data["duration_ms"],
        model=response_data["model"],
        resources_used=response_data["resources"]
    )

    # Log cost
    cost_logger.info({
        "request_id": request_data["id"],
        "cost_dollars": infra_cost,
        "model": response_data["model"],
        "tokens_generated": response_data["tokens"]
    })

    # Update cost counter
    api_cost_total.inc(infra_cost)
```

**Monthly Cost Report:**

- Total infrastructure cost
- Cost per request average
- Cost breakdown by model
- Cost forecast for next month
- Cost optimization recommendations

### Tools & Technologies

- **FastAPI** - High-performance async API framework with automatic documentation
- **Pydantic** - Request/response validation
- **Docker** - Containerization for consistency across environments
- **Ollama** - Local inference engine
- **Uvicorn** - ASGI server for FastAPI

### Evidence

## Phase 8: Monitoring & Observability

### Objective

Continuously track your LLM application's reliability, cost-effectiveness, and quality in production. Monitor both technical metrics (latency, errors) and quality metrics (user satisfaction, accuracy) to quickly identify and respond to issues.

### Key Activities

- **Track usage and cost metrics** - Monitor requests per minute, average latency, resource utilization, and associated infrastructure costs
- **Monitor quality metrics (user feedback, ratings)** - Collect and analyze user satisfaction ratings, error reports, and qualitative feedback on test case quality
- **Set up alerts for anomalies** - Trigger notifications when latency spikes, error rates exceed threshold, or costs deviate from baseline
- **Log all inputs/outputs for debugging** - Store request/response pairs for traceability, debugging, and quality analysis
- **Analyze failure patterns** - Identify recurring issues, common error types, and failure modes for continuous improvement

### Observability Framework

#### 1. Technical Metrics

**Request-Level Metrics:**

| Metric | Description | Tool | Alert Threshold |
| - | | - | -- |
| **Request Latency** | Time from request to response (P50, P90, P99) | Prometheus | P90 >5s |
| **Requests Per Second** | Throughput/RPS | Prometheus | >50 RPS |
| **Error Rate** | % of failed requests | Prometheus | >2% |
| **Status Code Distribution** | 2xx, 4xx, 5xx breakdown | Prometheus | >1% 5xx errors |
| **Cache Hit Rate** | % of cached responses | Prometheus | <70% indicates issue |
| **Queue Depth** | Pending requests in queue | Prometheus | >10 items |

**System-Level Metrics:**

| Metric | Description | Tool | Alert Threshold |
| | - | | |
| **CPU Usage** | Pod CPU utilization | Kubernetes metrics | >80% |
| **Memory Usage** | Pod memory utilization | Kubernetes metrics | >85% |
| **Disk Usage** | Persistent volume usage | Kubernetes metrics | >90% |
| **GPU Memory** | VRAM usage (if available) | nvidia-smi | >90% |
| **Network I/O** | Bytes in/out per second | Kubernetes metrics | Baseline dependent |
| **Pod Restart Count** | Unexpected restarts | Kubernetes metrics | >0 in 1 hour |

**Model-Specific Metrics:**

| Metric                                | Description                                 | Calculation                        | Target   |
| ------------------------------------- | ------------------------------------------- | ---------------------------------- | -------- |
| **Test Case Generation Success Rate** | % of stories that generate valid test cases | successes ÷ total requests         | >95%     |
| **Average Test Cases Per Story**      | Mean number of test cases produced          | sum of test cases ÷ stories        | ≥3.0     |
| **JSON Parsing Success Rate**         | % of outputs that parse as valid JSON       | valid outputs ÷ total responses    | >98%     |
| **Average Quality Score**             | Mean quality assessment score               | average of quality_scores          | ≥4.0/5.0 |
| **Retry Rate**                        | % of requests requiring retries             | retry requests ÷ total requests    | <3%      |
| **Model Switch Rate**                 | % of requests falling back to 3B model      | fallback requests ÷ total requests | <5%      |

#### 2. Cost Tracking

**Cost Alerts:**

| Alert | Condition | Action |
| -- | - | |
| **Budget overage** | Actual > forecast × 1.2 | Notify ops team |
| **Cost spike** | Daily cost > daily avg × 1.5 | Investigate usage surge |
| **Efficiency degradation** | Cost per request ↑ 20% | Check for retries or errors |

#### 3. Logging Strategy

**Log Levels:**

| Level | Use Case | Examples |
| | | - |
| **DEBUG** | Development troubleshooting | Token details, intermediate values |
| **INFO** | Normal operation events | Request start/completion, model switches |
| **WARN** | Potential issues | Slow requests, retries, cache misses |
| **ERROR** | Failures | Generation failures, validation errors |
| **CRITICAL** | System-level issues | Service unavailable, out of memory |

#### 5. Dashboard Design

**Main Dashboard (Real-time Overview):**

```
┌─────────────────────────────────────────────────────────────┐
│ Test Case Generation API - Production Dashboard              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Health Status: ✅ Healthy (99.9% uptime last 24h)           │
│                                                              │
│ ┌───────────────────┬───────────────────┬─────────────────┐ │
│ │ Requests/min      │ Avg Latency       │ Error Rate      │ │
│ │ 125 req/min       │ 2.3s (P90: 4.2s)  │ 0.8%            │ │
│ └───────────────────┴───────────────────┴─────────────────┘ │
│                                                              │
│ ┌───────────────────┬───────────────────┬─────────────────┐ │
│ │ Quality Score     │ Success Rate      │ Daily Cost      │ │
│ │ 4.1/5.0           │ 95.2%             │ $87.50          │ │
│ └───────────────────┴───────────────────┴─────────────────┘ │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Latency Distribution (last 1 hour)                       │ │
│ │ P50: 1.8s | P90: 4.2s | P99: 8.5s                        │ │
│ │ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ User Satisfaction (last 7 days)                          │ │
│ │ ⭐⭐⭐⭐⭐ 5 stars: 42% │ ⭐⭐⭐⭐ 4 stars: 38%             │ │
│ │ ⭐⭐⭐ 3 stars: 15% │ ⭐⭐ 2 stars: 4% │ ⭐ 1 star: 1%     │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ Recent Alerts: None | Last Updated: 2 minutes ago            │
└─────────────────────────────────────────────────────────────┘
```

**Quality Dashboard:**

```
┌─────────────────────────────────────────────────────────────┐
│ Quality Metrics Dashboard                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Test Case Metrics:                                           │
│  • Structural Compliance: 96.2% (target: >95%)              │
│  • Field Completeness: 94.1% (target: >90%)                 │
│  • Avg Test Cases: 3.2 per story (target: ≥3.0)            │
│  • JSON Valid: 98.5% (target: >98%)                         │
│                                                              │
│ Quality Scores by Model:                                     │
│  • llama3.2:1b: 3.8/5.0 (used 80% of time)                  │
│  • llama3.2:3b: 4.2/5.0 (used 20% of time)                  │
│                                                              │
│ Quality by Difficulty:                                       │
│  • Easy: 4.3/5.0 (50 samples)                               │
│  • Medium: 4.0/5.0 (120 samples)                            │
│  • Hard: 3.7/5.0 (30 samples)                               │
│                                                              │
│ Top Issues (Last 24h):                                       │
│  1. Vague expected results (8% of cases) - prompt update     │
│  2. Missing edge cases (5% of cases) - model evaluation      │
│  3. Too many test cases (3% of cases) - threshold tuning     │
└─────────────────────────────────────────────────────────────┘
```

### Alert Configuration

**Critical Alerts (Immediate Notification):**

```yaml
alerts:
  - name: ServiceDown
    condition: up{job="test-generator"} == 0
    duration: 2m
    severity: critical
    action: page_oncall

  - name: HighErrorRate
    condition: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    duration: 5m
    severity: critical
    action: page_oncall

  - name: DiskAlmostFull
    condition: disk_usage_percent > 90
    duration: 0m
    severity: critical
    action: page_oncall
```

**Warning Alerts (Email/Slack):**

```yaml
- name: HighLatency
  condition: histogram_quantile(0.9, http_duration_seconds) > 5
  duration: 10m
  severity: warning
  action: slack_notify

- name: QualityDegradation
  condition: avg(quality_score) < 3.8
  duration: 1h
  severity: warning
  action: slack_notify

- name: UnexpectedCostIncrease
  condition: daily_cost > avg_daily_cost * 1.3
  duration: 30m
  severity: warning
  action: slack_notify
```

### Debugging Tools

**Tracing Request:**

```python
# Enable distributed tracing to follow request through system
import opentelemetry

# Example: Trace a problematic request
trace_id = "req_12345"

# Retrieve logs
logs = elasticsearch.search(
    query={"match": {"trace_id": trace_id}},
    sort=[{"timestamp": "asc"}]
)

# View journey
for log in logs:
    print(f"{log['timestamp']}: {log['event']} - {log['duration_ms']}ms")
    # Output:
    # 10:30:45.000: request_received
    # 10:30:45.100: input_validated
    # 10:30:47.900: model_invoked
    # 10:30:47.950: output_generated
    # 10:30:48.050: validation_complete
    # 10:30:48.080: response_sent
```

### Tools & Technologies

- **Prometheus** - Metrics collection and alerting
- **Grafana** - Metrics visualization and dashboards
- **Datadog/New Relic** - All-in-one monitoring platform (alternative)
- **CloudWatch** - AWS native monitoring (if on AWS)
- **OpenTelemetry** - Distributed tracing
- **Alertmanager** - Alert routing and grouping

#### Evidence

![alt text](docs/resource/img/image.png)
![alt text](docs/resource/img/log.png)

## Phase 9: Feedback & Iteration

### Objective

LLMOps is a continuous cycle—analyze production data, identify improvements, update prompts or models, and respond to changing requirements. Regular maintenance ensures your application stays relevant, accurate, and aligned with user needs as technology and business evolve.

### Key Activities

- **Review monitoring data for improvement areas** - Analyze dashboards, logs, and metrics to identify patterns where quality drops or errors occur
- **Collect and analyze user feedback** - Aggregate ratings, comments, and bug reports to understand real-world satisfaction and pain points
- **Update prompts based on failure patterns** - Refine prompts when analysis shows systemic issues (e.g., "missing edge cases in 20% of hard stories")
- **Retrain or switch models when needed** - Evaluate new model versions, test on your dataset, and upgrade when quality improves justify the change
- **Expand evaluation datasets with production examples** - Add real-world failures to eval dataset to prevent regression

### Feedback Loop Cycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLMOps Continuous Cycle                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│          ┌──────────────────────────────────────────┐            │
│          │    1. Monitor & Analyze (1-2 weeks)      │            │
│          │  • Review error patterns and metrics    │            │
│          │  • Aggregate user feedback              │            │
│          │  • Identify top 3 improvement areas     │            │
│          └────────────────┬─────────────────────────┘            │
│                           │                                      │
│          ┌────────────────▼─────────────────────────┐            │
│          │    2. Plan Improvements (1 week)         │            │
│          │  • Root cause analysis                  │            │
│          │  • Design prompt/model changes          │            │
│          │  • Create test cases for fixes          │            │
│          └────────────────┬─────────────────────────┘            │
│                           │                                      │
│          ┌────────────────▼─────────────────────────┐            │
│          │    3. Implement & Test (1-2 weeks)       │            │
│          │  • Update prompts/models                │            │
│          │  • A/B test on validation set           │            │
│          │  • Document changes                     │            │
│          └────────────────┬─────────────────────────┘            │
│                           │                                      │
│          ┌────────────────▼─────────────────────────┐            │
│          │    4. Evaluate & Validate (1 week)       │            │
│          │  • Run full evaluation suite             │            │
│          │  • Compare baseline vs improved          │            │
│          │  • Get stakeholder approval              │            │
│          └────────────────┬─────────────────────────┘            │
│                           │                                      │
│          ┌────────────────▼─────────────────────────┐            │
│          │   5. Deploy & Monitor (ongoing)          │            │
│          │  • Canary rollout (5% → 100%)           │            │
│          │  • Monitor key metrics                   │            │
│          │  • Measure impact of changes             │            │
│          │  • Document learnings                    │            │
│          └────────────────┬─────────────────────────┘            │
│                           │                                      │
│                           └──► Back to Step 1                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Collection for Feedback

#### User Feedback Forms

**In-App Rating Collection:**

```python
# After each generation, collect feedback
feedback_schema = {
    "request_id": "req_12345",
    "timestamp": "2024-02-23T10:30:45Z",
    "user_id": "user_abc",
    "story_id": "US_001",
    "rating": 4,  # 1-5 stars
    "satisfaction": {
        "quality": 4,        # Relevance and accuracy of test cases
        "completeness": 5,   # Coverage of scenarios
        "usability": 3,      # Easy to execute/understand
        "speed": 5           # Response time acceptable
    },
    "feedback_type": "suggestion",  # ["praise", "bug", "suggestion"]
    "comment": "Would like more edge cases for negative scenarios",
    "would_recommend": true,         # NPS question
    "improvement_priority": "medium"
}
```

#### Production Data Analysis

**Monthly Data Review Meeting:**

```markdown
# February 2024 - Production Data Review

## Key Metrics

- Total Requests: 45,000
- Success Rate: 95.2% (target: >95%)
- Avg Quality Score: 4.1/5.0 (baseline: 3.9)
- User Satisfaction: 82% (4+ stars)
- P90 Latency: 4.2s (target: <5s)
- Monthly Cost: $2,847 (on budget)

## Top Issues (Last Month)

1. **"Hard" stories generate fewer test cases** (18% < 3 cases)
   - Impact: 540 stories affected
   - Root cause: Model struggles with complex requirements
   - Fix: Update prompt with better complex story examples

2. **Missing edge cases in 20% of cart operations**
   - Impact: Quality scores 3.2/5.0 vs 4.1 average
   - Root cause: Prompt doesn't explicitly ask for edge cases
   - Fix: Add edge case template to prompt

3. **Latency spike at 2-3 PM daily**
   - Impact: P99 reaches 15s (target <10s)
   - Root cause: Peak load, all pods at 85%+ CPU
   - Fix: Auto-scale to 5 pods during peak hours

## Positive Findings

- 92% of users mark results as "useful"
- No major security issues reported
- Llama 3.2 1B performs well (no fallbacks needed)

## Action Items for March

- [ ] Update prompt with edge case template
- [ ] Increase auto-scale ceiling from 3 to 5 pods
- [ ] A/B test new prompt on 10% of traffic
- [ ] Evaluate Llama 3.2 3B for hard stories only
- [ ] Add explicit edge case examples to training data

## Next Review: March 23, 2024
```

### Iteration Process

#### 1. Analyze Failure Patterns

**Example: Edge Case Missing Issue**

```python
# Query production logs for failures
failures = elasticsearch.search({
    "query": {
        "bool": {
            "must": [
                {"term": {"quality_score": {"gte": 2.0, "lte": 3.5}}},
                {"term": {"event": "test_generation_complete"}}
            ]
        }
    },
    "size": 100
})

# Analyze patterns
print(f"Total low-quality generations: {len(failures)}")
print(f"Common story topics: {Counter([f['topic'] for f in failures])}")
print(f"Avg test case count: {mean([f['test_count'] for f in failures])}")
# Output:
# Total low-quality generations: 847
# Common story topics: [('cart_operations', 156), ('payment', 134), ('search', 98)]
# Avg test case count: 2.1  # Below 3.0 target

# Identify the issue
for failure in failures[:5]:
    print(f"Story: {failure['story']}")
    print(f"  - Test cases: {failure['test_count']}")
    print(f"  - Quality: {failure['quality_score']}")
    print(f"  - User feedback: {failure['user_comment']}")
```

#### 2. Update Prompts

**Example: Add Edge Cases Template**

```python
# Original prompt
original_prompt = """
Generate test cases for this user story.
For each test case, include:
- Preconditions
- Steps
- Expected result

User Story: {story}
"""

# Improved prompt (based on failure analysis)
improved_prompt = """
Generate comprehensive test cases for this user story.
Include test cases for:
1. Happy path (normal, successful scenario)
2. Error cases (invalid inputs, failures)
3. Edge cases (boundary conditions, unusual inputs)
4. Security cases (where applicable)

For each test case, include:
- Preconditions (what must be true before test)
- Steps (numbered, specific actions)
- Expected result (what should happen)

User Story: {story}

Examples of edge cases to consider:
- Empty/null inputs
- Maximum/minimum values
- Concurrent operations
- Network failures
- Permission restrictions
"""

# A/B test on sample
results_original = test_prompt(original_prompt, test_set=10)  # 2.1 test cases avg
results_improved = test_prompt(improved_prompt, test_set=10)  # 3.4 test cases avg

improvement = (results_improved - results_original) / results_original * 100
print(f"Improvement: {improvement:.1f}%")  # Output: 61.9%
```

#### 3. A/B Test Changes

**Testing Framework:**

```python
# Split traffic for A/B testing
class PromptVersion(Enum):
    CONTROL = "original"  # 95% traffic
    TREATMENT = "improved"  # 5% traffic

# Route requests
def get_prompt_version(user_id: str) -> PromptVersion:
    hash_val = hash(user_id) % 100
    if hash_val < 5:
        return PromptVersion.TREATMENT
    return PromptVersion.CONTROL

# Measure impact over 1 week
results = {
    "control": {
        "quality_score": 3.9,
        "test_case_count": 2.1,
        "user_satisfaction": 0.79,
        "sample_size": 8550
    },
    "treatment": {
        "quality_score": 4.2,  # +7.7% improvement
        "test_case_count": 3.4,  # +61.9% improvement
        "user_satisfaction": 0.86,  # +8.9% improvement
        "sample_size": 450
    }
}

# Statistical significance test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(treatment_scores, control_scores)
if p_value < 0.05:
    print("✅ Improvement is statistically significant (p < 0.05)")
    # Roll out to 100% traffic
else:
    print("⚠️ Result could be due to chance, continue testing")
```

#### 4. Expand Evaluation Dataset

**Adding Production Examples:**

```python
# Monthly: Add real failures to evaluation dataset
failed_stories = elasticsearch.search({
    "query": {"term": {"quality_score": {"lt": 3.0}}},
    "size": 50
})

# Manually review and improve
for story in failed_stories:
    print(f"Story: {story['story']}")

    # Manually generate better test cases
    corrected_test_cases = [
        {
            "type": "positive",
            "title": "Valid cart operation",
            "preconditions": [...],
            "steps": [...],
            "expected_result": "..."
        },
        # ... more cases
    ]

    # Add to evaluation dataset with human-generated ground truth
    eval_dataset.add({
        "story_id": story['id'],
        "user_story": story['story'],
        "ground_truth_test_cases": corrected_test_cases,
        "source": "production_improvement",
        "date_added": "2024-02-23"
    })

print(f"Added {len(failed_stories)} new examples to evaluation dataset")
```

## Project Structure

```
.
├── README.md                          # Complete project documentation and LLMOps framework
├── .env                               # Environment variables (models, API keys, configs)
├── .gitignore                         # Git ignore rules
├── docker-compose.yml                 # Docker Compose configuration
├── requirements.txt                   # Python dependencies
│
├── docs/                              # Documentation
│   ├── decisions.md                   # Architecture and technology decisions
│   └── LLMOps/                        # LLMOps framework detailed phases
│       ├── 01.definition.md           # Phase 1: Problem Definition
│       ├── 02.data_preparation.md     # Phase 2: Data Collection & Preparation
│       └── 03.model-selection.md      # Phase 3: Model Selection & Evaluation
│
├── src/                               # Source code
│   ├── main.py                        # FastAPI application entry point
│   ├── config.py                      # Configuration management
│   ├── models.py                      # Pydantic models for validation
│   ├── api/
│   │   ├── endpoints.py               # API route handlers
│   │   └── auth.py                    # Authentication & authorization
│   ├── services/
│   │   ├── generation.py              # Test case generation logic
│   │   ├── validation.py              # Output validation
│   │   ├── ollama_client.py           # Ollama model interface
│   │   └── cache.py                   # Redis caching layer
│   ├── prompts/
│   │   ├── generate_tests.yaml        # Main generation prompt
│   │   └── evaluate_quality.yaml      # Quality evaluation prompt
│   └── utils/
│       ├── logging.py                 # Structured logging setup
│       ├── metrics.py                 # Prometheus metrics
│       └── security.py                # Input validation & sanitization
│
├── data/                              # Data and datasets
│   ├── examples/
│   │   └── user_stories_with_test_cases.json   # Ground truth examples
│   ├── test/
│   │   └── user_stories.json          # Test dataset (49 stories)
│   └── results.json                   # Generated test case results
│
├── docker/                            # Docker configuration
│   ├── Dockerfile                     # FastAPI application image
│   ├── docker-compose.yml             # Multi-container orchestration
│   ├── ollama/
│   │   ├── Dockerfile                 # Ollama service image
│   │   └── entrypoint.sh              # Model loading script
│   └── nginx/
│       └── nginx.conf                 # Reverse proxy configuration
│
├── monitoring/                        # Monitoring & observability
│   ├── prometheus/
│   │   ├── prometheus.yml             # Prometheus configuration
│   │   └── alerts.yml                 # Alert rules
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── main_dashboard.json    # Real-time metrics dashboard
│   │   │   └── quality_dashboard.json # Quality metrics dashboard
│   │   └── provisioning/              # Grafana provisioning configs
│   ├── elasticsearch/
│   │   └── elasticsearch.yml          # Elasticsearch config
│   └── logstash/
│       └── logstash.conf              # Log processing pipeline
│
├── scripts/                           # Utility scripts
│   ├── setup.sh                       # Initial setup script
│   ├── run_evaluation.py              # Automated evaluation runner
│   ├── analyze_logs.py                # Log analysis script
│   ├── generate_report.py             # Report generation
│   └── migrate_models.sh              # Model migration script
│
├── .github/
│   └── workflows/
│       ├── tests.yml                  # Automated testing on push
│       ├── evaluation.yml             # Weekly evaluation runs
│       ├── deploy.yml                 # Deployment automation
│       └── monitoring.yml             # Monitoring checks
│
└── logs/                              # Application logs (gitignored)
    ├── application.log                # Application logs
    ├── access.log                     # API access logs
    └── error.log                      # Error logs
```

## File System Organization

### Root Level Files

| File | Purpose |
| -- | |
| `README.md` | Complete project documentation covering all 9 LLMOps phases |
| `.env` | Environment variables (model names, API keys, configuration) |
| `requirements.txt` | Python package dependencies |
| `docker-compose.yml` | Local development environment setup |
| `.gitignore` | Git ignore rules (logs, cache, .env) |

### Data (`data/`)

- **examples/** - Ground truth examples for evaluation
  - `user_stories_with_test_cases.json` - 2 complete stories with perfect test cases
- **test/** - Test dataset
  - `user_stories.json` - 49 user stories with difficulty levels (easy/medium/hard)
- **results.json** - Generated test case results and quality scores

### Docker (`docker/`)

- **Dockerfile** - FastAPI application image
  - Base: Python 3.9
  - Dependencies: FastAPI, Pydantic, requests, redis-py
  - Health checks enabled

- **docker-compose.yml** - Local environment orchestration
  - FastAPI service (port 8000)
  - Ollama service (port 11434)
  - Redis cache (port 6379)
  - Prometheus (port 9090)

- **ollama/Dockerfile** - Ollama service image with model preloading

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Ollama (for local model inference)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLM--Test-for-UseCase

# Build and start services
docker-compose up -d

# Install the models
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull llama3-chatqa:8b
docker compose exec ollama ollama pull qwen3-vl:8b
docker compose exec ollama ollama pull llama3.2:3b

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the microservice
python src/main.py

# Test the API
curl -X POST http://localhost:8000/generate-tests \
  -H "Content-Type: application/json" \
  -d '{"story": "As a user, I want to log in with my email..."}'
```

## Next Steps

### Upcoming Enhancements & Implementation Tasks

This section outlines all pending features and improvements marked as "ToDo" throughout the LLMOps framework. Items are organized by phase and priority.

---

### Phase 2: Data Collection & Preparation

#### 1. Implement DVC (Data Version Control)

**Purpose:** Version datasets for reproducibility and experiment tracking

**Status:** ToDo

**Description:**

- Set up DVC to track all datasets in `data/` directory
- Enable reproducible data pipelines for train/test splits
- Maintain audit trail of dataset changes
- Link DVC with git for version control integration

**Implementation Details:**

```bash
# Initialize DVC in project
dvc init

# Track datasets
dvc add data/test/user_stories.json
dvc add data/examples/user_stories_with_test_cases.json

# Create data pipeline (dvc.yaml)
# Define stages for data collection, cleaning, splitting
```

**Expected Outcome:**

- All datasets versioned and tracked
- Data lineage documented
- Reproducible data pipelines for future experiments
- Ability to roll back to previous dataset versions

---

#### 2. Integrate Label Studio for Manual Annotation

**Purpose:** Enable manual annotation and quality review of test cases

**Status:** ToDo

**Description:**

- Set up Label Studio instance for human-in-the-loop annotation
- Create annotation interface for test case quality review
- Define labeling guidelines and scoring rubrics
- Build feedback loop from QA to model improvements

**Implementation Details:**

```python
# Label Studio configuration
# Annotation tasks:
# 1. Rate test case quality (1-5 scale)
# 2. Flag issues (missing steps, vague preconditions, etc.)
# 3. Suggest improvements
# 4. Mark edge cases and special scenarios
```

**Expected Outcome:**

- Quality annotations for generated test cases
- Human feedback integrated into evaluation metrics
- Training data for fine-tuning models
- Continuous quality improvement cycle

---

### Phase 6: Evaluation & Testing

#### 1. Implement DeepEval Framework

**Purpose:** Use LLM evaluation framework with pre-built metrics for automated quality assessment

**Status:** ToDo

**Description:**

- Integrate DeepEval library for standardized LLM evaluation
- Implement pre-built metrics (relevance, faithfulness, coherence)
- Create custom metrics for test case evaluation
- Automate evaluation pipeline in CI/CD

**Installation:**

```bash
pip install deepeval
```

**Configuration:**

```python
from deepeval import evaluate
from deepeval.metrics import Relevance, Faithfulness

# Define evaluation metrics
# - Relevance: Test case relevant to user story
# - Faithfulness: Test case grounded in story requirements
# - Completeness: Coverage of story scenarios
# - Clarity: Test steps are clear and specific
```

**Integration Points:**

- Phase 6 evaluation pipeline
- CI/CD workflow for automated testing
- Quality tracking dashboard
- Report generation

**Expected Outcome:**

- Standardized, reproducible evaluation metrics
- Automated quality scoring for all generated test cases
- Benchmarking across model versions
- Evidence-based model selection

---

#### 2. Set Up Load Testing Tools (locust, wrk)

**Purpose:** Latency and throughput benchmarking under production load

**Status:** ToDo

**Description:**

- Install and configure load testing tools (locust and/or wrk)
- Create load test scenarios simulating production traffic
- Profile API latency at various request volumes
- Identify performance bottlenecks and scaling limits

**Installation:**

```bash
pip install locust
# or
brew install wrk  # macOS
```

**Locust Configuration Example:**

```python
from locust import HttpUser, task

class TestCaseGenerationUser(HttpUser):
    @task
    def generate_tests(self):
        payload = {
            "story": "As a customer, I want to search for products...",
            "difficulty": "medium"
        }
        self.client.post("/generate-tests", json=payload)
```

**Load Test Scenarios:**

- Ramp up: Gradually increase users from 1 to 100
- Sustained: Hold at peak load for 5-10 minutes
- Spike: Sudden burst to test handling capacity
- Stress: Push beyond expected capacity to find limits

**Metrics to Track:**

- Response time (P50, P90, P99)
- Requests per second (RPS)
- Error rate under load
- Memory and CPU usage during load
- Cache hit rates

**Expected Outcome:**

- Baseline performance metrics at different load levels
- Identification of performance degradation points
- Scaling recommendations (vertical vs. horizontal)
- SLA targets for production deployment
- Documentation of performance characteristics

---

### Summary of All ToDo Items

| Phase       | Feature                   | Priority | Complexity | Est. Effort |
| ----------- | ------------------------- | -------- | ---------- | ----------- |
| **Phase 2** | DVC Data Versioning       | Medium   | Medium     | 4-6 hours   |
| **Phase 2** | Label Studio Integration  | High     | High       | 8-12 hours  |
| **Phase 6** | DeepEval Metrics          | High     | Medium     | 6-8 hours   |
| **Phase 6** | Load Testing (locust/wrk) | Medium   | Medium     | 4-6 hours   |

---

### Recommended Implementation Order

1. **Start with Load Testing** (Phase 6)
   - Quick to set up and provides immediate insights
   - Helps validate current infrastructure
   - Foundation for scaling decisions

2. **Implement DeepEval** (Phase 6)
   - Enhances evaluation pipeline
   - Provides standardized metrics
   - Feeds into quality tracking

3. **Set Up DVC** (Phase 2)
   - Enables reproducible workflows
   - Prepares for iterative improvements
   - Supports future experiments

4. **Integrate Label Studio** (Phase 2)
   - Most complex, highest impact long-term
   - Establishes human-in-the-loop feedback
   - Powers continuous improvement cycle

---

### Integration with Existing Systems

**DeepEval Integration:**

- Feeds evaluation metrics into Phase 6: Evaluation & Testing
- Metrics displayed on Grafana dashboards (Phase 8)
- Results logged for feedback analysis (Phase 9)

**Load Testing Integration:**

- Results inform Phase 7: Deployment & Serving scaling decisions
- Latency metrics feed into monitoring alerts (Phase 8)
- Data informs infrastructure cost calculations

**DVC Integration:**

- Enables Phase 2: Data Collection & Preparation reproducibility
- Supports Phase 5: RAG & Prompting with versioned context
- Facilitates Phase 9: Feedback & Iteration experiments

**Label Studio Integration:**

- Provides ground truth for Phase 6: Evaluation & Testing
- Generates training data for potential fine-tuning
- Feeds human feedback into Phase 9: Feedback & Iteration

---
