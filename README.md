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
