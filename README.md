# Test Case Generator - LLMOps Project

> **Automated test case generator** that converts user stories into structured test cases using LLMs, validates quality with MLflow, and serves via FastAPI.

## 🎯 Project Goal

Build a production-ready system that:

- ✅ Accepts user stories via REST API
- ✅ Generates structured test cases in Given-When-Then format
- ✅ Validates quality with multi-layer checks
- ✅ Tracks experiments and metrics with MLflow
- ✅ Works locally with Ollama or cloud with OpenAI

## 🚀 Quick Start

### 1. Prerequisites

- Docker & Docker Compose (for containerized setup)
- OR Python 3.11+ with Ollama running locally

### 2. Start Services

```bash
# Clone/enter project directory
cd test-case-generator

# Start all services (API, Ollama, MLflow)
docker-compose up -d

# Pull Ollama model (first time only)
docker exec ollama-service ollama pull llama3.2:3b

# Check services are running
docker-compose ps
```

Expected output:

```
NAME                COMMAND                  STATUS
ollama-service      "ollama serve"           Up (healthy)
test-case-api       "uvicorn src.api..."     Up (healthy)
mlflow-ui           "mlflow ui ..."          Up (healthy)
```

### 3. Generate Test Cases

```bash
# Using curl
curl -X POST http://localhost:8001/generate-test-cases \
  -H "Content-Type: application/json" \
  -d '{
    "user_story": "As a user, I want to reset my password so I can regain access to my account",
    "include_quality_check": true
  }'

# OR using Python
python examples/usage_example.py
```

### 4. View Results

- **API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **MLflow Dashboard**: http://localhost:5001
- **Ollama**: http://localhost:11435 (internal Docker use only)

## 📋 API Endpoints

### `POST /generate-test-cases`

Generate test cases from a user story.

**Request:**

```json
{
  "user_story": "As a user, I want to reset my password so I can regain access",
  "include_quality_check": true
}
```

**Response:**

```json
{
  "user_story": "As a user, I want to reset my password...",
  "test_cases": [
    {
      "id": "TC_001",
      "title": "Successful password reset with valid email",
      "priority": "critical",
      "given": "User is on forgot password page",
      "when": "User enters valid registered email",
      "then": "Reset link is sent to email within 2 minutes"
    }
  ],
  "validation": {
    "structure_valid": true,
    "count": 4,
    "quality_passed": true,
    "coverage_passed": true
  },
  "quality_metrics": {
    "relevance": 0.85,
    "coverage": 0.8,
    "clarity": 0.88,
    "overall": 0.84,
    "passed": true
  },
  "metadata": {
    "latency": 2.3,
    "tokens": 450,
    "model": "llama3.2:3b",
    "provider": "ollama"
  }
}
```

### `GET /health`

Check API and LLM health status.

**Response:**

```json
{
  "status": "healthy",
  "llm": "connected",
  "model": "llama3.2:3b",
  "provider": "ollama"
}
```

### `GET /metrics`

Get aggregated metrics from MLflow.

**Response:**

```json
{
  "message": "View detailed metrics in MLflow UI",
  "mlflow_ui": "http://localhost:5000",
  "summary": {
    "latency_seconds": {...},
    "structure_valid": {...},
    "test_case_count": {...}
  }
}
```

## 📁 Project Structure

```
test-case-generator/
├── data/
│   ├── examples/              # Few-shot examples for prompting
│   │   └── user_stories.json  # 3 example user stories with test cases
│   └── validation/            # Evaluation dataset
│       └── test_dataset.json  # 5 test user stories
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI application
│   ├── llm/
│   │   ├── client.py         # LLM client (Ollama/OpenAI)
│   │   └── prompts.py        # Prompt templates
│   ├── validators/
│   │   ├── structure.py      # Pydantic models for validation
│   │   └── quality.py        # Quality metrics (LLM-judge + coverage)
│   └── mlflow_tracker.py     # MLflow experiment tracking
├── examples/
│   └── usage_example.py      # API usage examples
├── scripts/
│   └── run_evaluation.py     # Evaluation on test dataset
├── tests/                    # Unit tests
├── mlruns/                   # MLflow experiment tracking
├── reports/                  # Generated evaluation reports
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# Model provider and settings
export OLLAMA_HOST="http://localhost:11434"  # Default: http://localhost:11434
export OPENAI_API_KEY="sk-..."              # Only if using OpenAI

# Server settings
export PORT=8000                             # Default: 8000
```

### Model Configuration

Edit `src/llm/client.py` to change the model:

```python
# Use different model
config = LLMConfig(
    provider="ollama",           # "ollama" or "openai"
    model="llama2:7b",          # Model name
    temperature=0.3,            # Lower = more deterministic
    max_tokens=2000
)
```

**Available Models:**

- **Ollama (Free, Local)**
  - `llama3.2:3b` (3B parameters, ~2GB) - Good balance
  - `llama2:7b` (7B parameters, ~4GB) - Higher quality
  - `mistral:7b` (7B parameters) - Fast

- **OpenAI (Paid, Cloud)**
  - `gpt-3.5-turbo` (~$0.002 per 1K tokens)
  - `gpt-4` (~$0.03 per 1K tokens)

## 📊 Evaluation & Metrics

### Run Evaluation

```bash
# Test on evaluation dataset (5 user stories)
python scripts/run_evaluation.py

# Output:
# EVALUATION SUMMARY
# Pass Rate: 4/5 (80.0%)
# Avg Quality Score: 0.82
# Avg Coverage Score: 0.78
# Avg Latency: 2.4s
```

### View in MLflow

```bash
# Open MLflow UI in browser
mlflow ui --host 0.0.0.0 --port 5000

# Navigate to:
# http://localhost:5000
```

**Key Metrics:**

- `structure_valid`: Pydantic validation passed
- `test_case_count`: Number of generated test cases
- `quality_overall_score`: LLM-judge semantic score (0-1)
- `coverage_score`: Test case diversity score (0-1)
- `latency_seconds`: Response time

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/ -v

# Or specific test
pytest tests/test_prompts.py::test_prompt_quality -v
```

### Test Validators

```bash
# Test structure validation
python -m src.validators.structure

# Test quality validation
python -m src.validators.quality
```

### Test LLM Client

```bash
# Test Ollama connection
python -m src.llm.client

# Test with OpenAI
export OPENAI_API_KEY="sk-..."
python -m src.llm.client
```

## 📈 Success Metrics

| Metric                   | Target | Actual |
| ------------------------ | ------ | ------ |
| Structural Validity      | ≥95%   | -      |
| Coverage (3+ test cases) | ≥90%   | -      |
| Quality Score            | ≥0.75  | -      |
| API Latency              | <5s    | -      |
| Pass Rate (Evaluation)   | ≥70%   | -      |

## 🛠️ Local Development

### Without Docker

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve &
ollama pull llama3.2:3b

# 3. Start MLflow (in separate terminal)
mlflow ui --port 5000 &

# 4. Start API
uvicorn src.api.main:app --reload --port 8000
```

### Adding New Features

1. **New LLM Provider**: Add method to `LLMClient` class
2. **New Validation Rule**: Add validator to `src/validators/quality.py`
3. **New Endpoint**: Add route to `src/api/main.py`
4. **New Metric**: Track in `MLflowTracker`

## 🚨 Troubleshooting

### API Can't Connect to Ollama

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Or in Docker
docker-compose restart ollama-service
```

### Model Not Found

```bash
# Pull the model
ollama pull llama3.2:3b

# Or in Docker
docker exec ollama-service ollama pull llama3.2:3b
```

### Memory Issues

If running out of memory with Ollama:

- Use smaller model: `llama2:3b` or `mistral:7b`
- Reduce `max_tokens` in config
- Increase Docker memory allocation

### JSON Parse Errors

- Check LLM response: `curl -X POST http://localhost:8000/generate-test-cases ...`
- Review prompt in `src/llm/prompts.py`
- Try a different user story
- Use higher temperature for more variation

## 📚 Example Workflows

### Workflow 1: Single Test Case Generation

```bash
curl -X POST http://localhost:8000/generate-test-cases \
  -H "Content-Type: application/json" \
  -d '{
    "user_story": "As a customer, I want to add items to my cart",
    "include_quality_check": true
  }'
```

### Workflow 2: Batch Processing

```python
import requests

stories = [
    "As a user, I want to login",
    "As an admin, I want to delete users",
    "As a seller, I want to upload images"
]

for story in stories:
    response = requests.post(
        "http://localhost:8000/generate-test-cases",
        json={"user_story": story}
    )
    print(response.json()["test_cases"])
```

### Workflow 3: Evaluation & Benchmarking

```bash
# Run evaluation
python scripts/run_evaluation.py

# View results
cat reports/evaluation_results.json

# Compare in MLflow
mlflow ui
```

## 🔐 Security Considerations

- ✅ Input validation: User stories validated for length
- ✅ Output validation: Pydantic strict schema
- ✅ Error handling: Graceful error messages without internals
- ✅ No secrets in code: Use environment variables
- ✅ Rate limiting: Can be added via FastAPI middleware

## 📈 Performance Optimization

### Tips for Faster Generation

1. **Use smaller model**: `llama2:3b` instead of `7b`
2. **Lower temperature**: Reduces variation, faster convergence
3. **Fewer examples**: Use 1 instead of 2 in prompt
4. **Async processing**: API already uses background tasks for MLflow

### Expected Performance

| Model       | Latency | Quality   | Memory |
| ----------- | ------- | --------- | ------ |
| llama2:3b   | 1-2s    | Medium    | 2GB    |
| llama3.2:3b | 2-3s    | Good      | 2GB    |
| llama2:7b   | 3-5s    | High      | 4GB    |
| GPT-3.5     | 1-2s    | Excellent | Cloud  |

## 🎓 Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
- **Ollama**: https://ollama.ai/
- **MLflow**: https://mlflow.org/
- **LLM Prompting**: https://platform.openai.com/docs/guides/prompt-engineering

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Batch processing endpoint
- [ ] Multiple output formats (Cucumber, pytest)
- [ ] Fine-tuning on custom data
- [ ] Human-in-the-loop review interface
- [ ] CI/CD integration
- [ ] Cloud deployment templates
- [ ] Performance benchmarking suite

## 📞 Support

- **Issues**: Check `docker logs` for error details
- **Questions**: Review examples in `examples/` directory
- **Metrics**: Check MLflow UI for detailed tracking
- **Documentation**: See `docs/decisions.md` for architecture
