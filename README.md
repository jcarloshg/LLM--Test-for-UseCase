# LLM Test Case Generation - Use Case Framework

This project implements a microservice that leverages Large Language Models (LLMs) to automatically generate structured test cases from user stories and project requirements. Using prompt engineering and validation techniques, the service produces JSON-formatted test artifacts with clear acceptance criteria, preconditions, and test steps.

## Table of Contents

### Setup & Deployment

- [Getting Started](#getting-started)
- [Next Steps](#next-steps)

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

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional but recommended for better performance)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLM--Test-for-UseCase

# Create .env file (configure as needed)
cp .env.example .env  # if available, or create manually

# Build and start all services
docker-compose up -d

# Wait for Ollama to be ready (check health with)
docker-compose exec ollama ollama list

# Pull the required models (if not auto-loaded)
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull llama3-chatqa:8b
docker compose exec ollama ollama pull qwen3-vl:8b
docker compose exec ollama ollama pull llama3.2:3b
```

### Running the Service

Once `docker-compose up -d` completes, the services are automatically running:

- **Test Case Generation API**: http://localhost:8001
- **Grafana Dashboard**: http://localhost:3000 (monitoring and visualization)
- **MLflow UI**: http://localhost:5001 (experiment tracking)
- **Ollama**: http://localhost:11435 (LLM backend)
- **Loki**: http://localhost:3100 (log aggregation)

Test the API:

```bash
curl -X POST http://localhost:8001/generate-tests \
  -H "Content-Type: application/json" \
  -d '{"story": "As a user, I want to log in with my email..."}'
```

View logs:

```bash
# Check Docker Compose logs
docker-compose logs -f api

# View logs in Grafana
# Navigate to http://localhost:3000 and access Loki logs
```

## Phase 1: Problem Definition & Use Case Design

### Objective

Establish the foundation by clearly defining:

- **Problem Statement**: Generate valid, structured test cases from user stories using prompt engineering and LLM capabilities
- **Solution Fit**: LLM chosen for natural language understanding and creative test case generation
- **Scope**: Simple prompting approach with local model deployment via Ollama
- **Constraints**: Budget (compute-based), latency (<5s P90), privacy (self-hosted)

### Key Activities

**1. Define Use Case & Outcomes**

- Receive a user story as input
- Generate structured test cases using an LLM
- Perform basic quality validation on the generated output
- Expose functionality through a REST API

**2. Success Metrics & Constraints**

These metrics define success across three dimensions:

**a) Accuracy & Quality (Precisión y Calidad de la Salida)**

Measured through rule compliance and output utility rather than traditional classification metrics.

| Metric                    | Target   | Details                                                                |
| ------------------------- | -------- | ---------------------------------------------------------------------- |
| **Structural Compliance** | >95%     | JSON parsing success on first attempt, no retry mechanisms             |
| **Quality Score**         | ≥4.0/5.0 | Heuristic evaluation: preconditions, logical steps, field completeness |
| **Retry Rate**            | <5%      | Malformed responses or schema violations                               |

**b) Performance & Latency (Latencia y Rendimiento)**

Managing inference latency with local models (Llama 3, Qwen, Mistral via Ollama).

| Metric                  | Target            | Details                                                                         |
| ----------------------- | ----------------- | ------------------------------------------------------------------------------- |
| **End-to-End Latency**  | P90 <5s, P99 <10s | Time from `POST /generate-tests` to response (inference + parsing + validation) |
| **Validation Overhead** | <20%              | Additional validation time vs. pure text generation                             |

**c) Cost & Resource Efficiency (Coste por Solicitud y Eficiencia)**

With Ollama, token cost = $0. Cost shifts to compute infrastructure.

| Metric                   | Approach                  | Details                                   |
| ------------------------ | ------------------------- | ----------------------------------------- |
| **Infrastructure Cost**  | Monitor resource usage    | Monthly server cost ÷ max throughput      |
| **Resource Utilization** | Log CPU/RAM/VRAM baseline | Docker container consumption at peak load |

## Phase 2: Data Collection & Preparation

### Objective

Prepare high-quality training and evaluation data for your LLM application:

- Gather diverse examples for prompts, evaluation, and potential fine-tuning
- Ensure data quality, diversity, and proper formatting
- Maintain reproducibility and traceability throughout the data lifecycle

> **Key Principle:** Quality over quantity—100 high-quality examples often outperform 1,000 noisy ones.

### Key Activities

**1. Collect Representative Examples**

- Gather diverse user stories covering different features, complexity levels, and user personas
- Aim for balanced representation across domains (e-commerce, authentication, analytics, etc.)
- Capture edge cases and boundary conditions

**2. Clean & Anonymize Data**

- Remove personally identifiable information (PII) from stories and test data
- Standardize formatting and remove inconsistencies
- Validate data integrity before storage

**3. Structure Data for Use**

- Format as JSON, JSONL, or CSV for easy loading
- Include metadata (difficulty, domain, quality score)
- Create clear separation between input and output

### Dataset Overview

Currently, the project includes **49 diverse user stories** across an e-commerce platform:

| Dataset                   | Location                                                                                           | Contents                                                      | Size        |
| ------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------- |
| **Test Stories**          | [data/test/user_stories.json](data/test/user_stories.json)                                         | 49 user stories with difficulty levels (easy/medium/hard)     | 1,097 bytes |
| **Ground Truth Examples** | [data/examples/user_stories_with_test_cases.json](data/examples/user_stories_with_test_cases.json) | 49 user stories with 100 test cases, quality scores 0.71-0.92 | ~90+ KB     |

### Data Distribution

**49 User Stories Across Difficulty Levels:**

| Difficulty | Count | Example User Stories                                                               |
| ---------- | ----- | ---------------------------------------------------------------------------------- |
| **Easy**   | ~15   | Product recommendations, account export, invoice printing, voice search, wishlists |
| **Medium** | ~27   | Shopping cart, search, inventory, notifications, checkout, product variants        |
| **Hard**   | ~7    | Login/auth, password reset, profile updates, 2FA, secure payment processing        |

### Tools & Technologies

| Tool                           | Purpose                                          | Status         | Implementation                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------ | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pydantic**                   | Input/output validation with schema enforcement  | ✅ Implemented | [input](src/application/create_tests/models/generate_test_cases_request.py) [output](src/application/create_tests/infra/executable_chain/robust_json_output_parser.py) |
| **JSON Schema Validator**      | Validate test case structure compliance          | ✅ Implemented | [robust_json_output_parser.py](src/application/create_tests/infra/executable_chain/robust_json_output_parser.py)                                                       |
| **DVC (Data Version Control)** | Version control for datasets and reproducibility | 📋 ToDo        | 📋 ToDo                                                                                                                                                                |
| **Label Studio**               | Human-in-the-loop annotation and QA review       | 📋 ToDo        | 📋 ToDo                                                                                                                                                                |

**Data Pipeline (ToDo):**

```
Raw Data → Collection → Cleaning → Validation → Versioning → Ready for Use
           (49 stories)   (PII)      (Schema)     (DVC)      (training/eval)
```

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

Comprehensive evaluation across three dimensions:

#### 1. Accuracy & Quality (Precisión y Calidad de la Salida)

Measured through rule compliance and output utility rather than traditional classification metrics.

- Implementation: [src/application/evaluate_models/model/quality_tracker.py](src/application/evaluate_models/model/quality_tracker.py)

| Metric                        | Target       | Details                                                                |
| ----------------------------- | ------------ | ---------------------------------------------------------------------- |
| **Structural Compliance**     | >95%         | JSON parsing success on first attempt, no retry mechanisms             |
| **Quality Score**             | ≥4.0/5.0     | Heuristic evaluation: preconditions, logical steps, field completeness |
| **Retry Rate**                | <5%          | Malformed responses or schema violations                               |
| **Test Case Count**           | ≥3 per story | Average number of test cases generated per user story                  |
| **Precondition Completeness** | ≥90%         | % of test cases with non-empty, meaningful preconditions               |
| **Step Clarity**              | 2-5 steps    | Average steps per test case (optimal range)                            |
| **Semantic Relevance**        | ≥4.0/5.0     | LLM evaluation of story-to-tests alignment                             |

#### 2. Performance & Latency (Latencia y Rendimiento)

Managing inference latency with local models (Llama 3, Qwen, Mistral via Ollama). Tracking percentile-based metrics per best practices.

- Implementation: [src/application/evaluate_models/model/latency_tracker.py](src/application/evaluate_models/model/latency_tracker.py)

| Metric                  | Target      | Details                                               |
| ----------------------- | ----------- | ----------------------------------------------------- |
| **Mean Latency**        | <3s         | Average response time across all requests             |
| **Median (P50)**        | <2.5s       | 50th percentile - typical response time               |
| **P95 Latency**         | <5s         | 95% of requests complete within this time (SLA basis) |
| **P99 Latency**         | <10s        | 99th percentile latency (worst case scenarios)        |
| **Min Latency**         | Baseline    | Fastest recorded response time                        |
| **Max Latency**         | <15s        | Slowest recorded response time                        |
| **Std Dev**             | <2s         | Standard deviation indicates consistency              |
| **Throughput**          | >10 req/min | Sustained requests per minute on single instance      |
| **Validation Overhead** | <20%        | Additional validation time vs. pure text generation   |

**SLA Definition (Best Practice):**

> "95% of all requests must complete within the P95 latency target"
>
> - Implemented via `LatencyTracker.meets_sla(latencies, sla_p95_ms)`
> - Validates P95 percentile against configurable threshold

#### 3. Cost & Resource Efficiency (Coste por Solicitud y Eficiencia)

With Ollama, token cost = $0. Cost shifts to compute infrastructure.

- Implementation: [src/application/evaluate_models/model/cost_tracker.py](src/application/evaluate_models/model/cost_tracker.py)
- Evaluation Integration: [src/application/evaluate_models/application/evaluate_models_application.py](src/application/evaluate_models/application/evaluate_models_application.py)

| Metric                      | Target      | Details                                                           |
| --------------------------- | ----------- | ----------------------------------------------------------------- |
| **Cost per 1,000 Requests** | <$20        | Monthly server cost ÷ (max_requests/day × 30) × 1,000             |
| **Cost Efficiency Score**   | >0.7        | Composite score: (latency_efficiency + throughput_efficiency) / 2 |
| **Throughput**              | >30 req/min | Sustained requests per minute (60 ÷ avg_latency)                  |
| **CPU Usage**               | <80%        | Average CPU usage during inference (2-core baseline)              |
| **Memory per Request**      | <100 MB     | (Base model memory ÷ num_requests) + 50 MB overhead               |
| **Concurrent Capacity**     | >2 requests | Estimated simultaneous requests based on available memory         |
| **Total Execution Time**    | <500s       | Sum of all request latencies for evaluation set                   |

**Default Configuration (Baseline):**

```
Monthly Server Cost:    $100 USD
Max Requests/Day:       5,000
Container Memory:       8 GB
Container CPU Cores:    2
Base Model Memory:      4.0 GB (quantized 4-bit)
Memory Overhead:        50 MB per request
```

**Cost Calculation Formula:**

```
Cost per 1,000 = (Monthly Cost ÷ (Max Requests/Day × 30)) × 1,000
Example: ($100 ÷ (5,000 × 30)) × 1,000 = $0.67 per 1,000 requests
```

**Optimization Thresholds & Recommendations:**

| Condition                  | Recommendation                                | Impact                    |
| -------------------------- | --------------------------------------------- | ------------------------- |
| Latency > 3s               | Use 4-bit quantized model or upgrade hardware | Reduce response time      |
| Throughput < 20 req/min    | Increase CPU cores or optimize prompts        | Improve capacity          |
| Cost > $10 per 1K requests | Review server capacity utilization            | Lower infrastructure cost |
| CPU Usage > 80%            | Enable load balancing or add replicas         | Prevent throttling        |
| Efficiency Score < 0.6     | Comprehensive review of configuration         | Improve all metrics       |

#### Model Comparison & MLflow Evaluation Results

| Model                 | Quality Score | Avg Latency | P95 Latency | Throughput | Cost/1K | Efficiency |
| --------------------- | ------------- | ----------- | ----------- | ---------- | ------- | ---------- |
| **Llama 3.2 1B**      | 0.78-0.82     | 1.2s        | 1.8s        | 50 req/min | $0.67   | 0.85       |
| **Llama 3.2 3B**      | 0.85-0.90     | 2.5s        | 4.2s        | 24 req/min | $0.67   | 0.72       |
| **Llama 3 ChatQA 8B** | 0.88-0.95     | 4.5s        | 7.1s        | 13 req/min | $0.67   | 0.58       |

_Model Performance Summary (5,000 requests/day baseline)_

### Evidence RAG vs Prompt

![MLflow Model Comparison - Part 1](docs/resource/img/mlflow.png)
_MLflow Dashboard showing quality, latency, and throughput metrics across evaluated models_

![MLflow Model Comparison - Part 2](docs/resource/img/mlflow_01.png)
_MLflow Dashboard showing cost efficiency, resource utilization, and additional performance metrics_

## Phase 4: Prompt Engineering & Optimization

### Objective

Craft effective instructions that guide LLMs to produce desired outputs through iterative refinement. This phase often yields **80% of performance improvements** without model changes. Design system prompts, add examples (few-shot learning), structure outputs, and establish reusable templates.

> **Key Insight:** A well-engineered prompt can make a 1B model outperform a poorly-prompted 8B model.

### Key Activities

**1. Create Prompt Templates**

- Build reusable templates with variables
- Support different input variations
- Enable A/B testing of prompt versions

**2. Implement Few-Shot Examples**

- Provide representative input-output examples
- Show desired format and quality levels
- Improve accuracy through demonstration learning

### Prompt Engineering Techniques

- Implementation: [src/application/create_tests/models/templates.py](src/application/create_tests/models/templates.py)

| Technique             | Use Case               | Implementation                                                                                                       |
| --------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Role-Based**        | Domain expertise       | "You are an expert QA Engineer specializing in test case design..."                                                  |
| **Structured Output** | JSON schema            | Strict JSON with 8 required fields (id, type, title, priority, preconditions, steps, expected_result, quality_score) |
| **Few-Shot Learning** | Format consistency     | Example test case structure embedded in prompt template                                                              |
| **Explicit Rules**    | Error prevention       | 8 numbered rules (coverage balance, priority distribution, array format requirements)                                |
| **Edge Case Focus**   | Comprehensive coverage | Test type categories: positive / negative / edge_case / boundary                                                     |

**Active Prompt Variants from templates.py:**

1. **RAG_PROMPT** (Advanced)
   - Incorporates external context via RAG retrieval
   - Best for: Complex or domain-specific user stories
   - Structure: Role-based → Task clarity → Test field table → Rules → Few-shot example

2. **PROMPT** (Basic)
   - Minimal instructions, fastest inference
   - Best for: Simple stories, speed-critical deployments
   - Structure: Compact field definitions → JSON example

3. **IMPROVED_PROMPT_V1** (Production Default)
   - Enhanced with detailed validation rules and table structure
   - Best for: Quality-critical, production environments
   - Structure: Role-based → Test field table → 8 explicit rules → Few-shot example
   - Key Rules: Coverage balance (3-8 test cases), Priority distribution (1-2 critical, 2-3 high), Quality score (6-9 range)

**Validation Framework (IMPROVED_PROMPT_V1):**

```
Rule 1: Generate 3-8 test cases covering happy path, edge cases, errors
Rule 2: Ensure balanced coverage across positive, negative, edge case types
Rule 3: Each test case MUST have all required fields
Rule 4: Priority distribution (1-2 critical, 2-3 high, 1-2 medium, 0-1 low)
Rule 5: Quality score should reflect comprehensiveness (6-9 typical)
Rule 6: Be specific and actionable in each step
Rule 7: "steps" MUST be array of strings ONLY (NOT objects)
Rule 8: "preconditions" MUST be array of strings ONLY (NOT objects)
```

### Tools & Technologies

| Tool          | Purpose                      | Status         |
| ------------- | ---------------------------- | -------------- |
| **Pydantic**  | Structured output validation | ✅ Implemented |
| **LangChain** | Prompt templating & chaining | ✅ Implemented |
