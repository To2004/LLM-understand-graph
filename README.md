# LLM Graph Reasoning Framework

An agent-based reasoning framework for solving classical graph problems using Large Language Models (LLMs) with explicit parsing, tool execution, and verification.

## Project Overview

This project implements an agentic pipeline that decomposes graph reasoning into explicit, verifiable stages:
- **Agent Parser**: Extracts graph structure from natural language
- **Agent Chooser**: Selects appropriate graph algorithms
- **Agent Verifier**: Validates parsing and solutions
- **Agent Orchestrator**: Coordinates the entire pipeline

## Architecture

```
src/
├── agents/          # Core agent implementations
├── parsers/         # Graph parsing logic
├── algorithms/      # Classical graph algorithm implementations
├── verifiers/       # Validation and verification logic
├── orchestrator/    # Pipeline coordination
├── models/          # LLM integration and adapters
├── benchmarks/      # NLGraph and GraphInstruct evaluation
└── utils/           # Helper utilities

tests/               # Unit and integration tests
experiments/         # Experiment scripts and configurations
configs/             # Configuration files
data/                # Benchmark datasets
results/             # Experimental results and logs
```

## Getting Started

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) - an extremely fast Python package manager (10-100x faster than pip).

**Install UV:**
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 1. Install Dependencies

**With UV (Recommended - Fast!):**
```bash
# Install all dependencies (creates virtual environment automatically)
uv sync

# Or production only
uv sync --no-dev
```

**Traditional method (if you prefer):**
```bash
pip install -r requirements.txt
```

### 2. Setup OpenRouter API (Free Models)
```bash
# Copy the environment template
cp .env.example .env

# Get a free API key at https://openrouter.ai/keys
# Add your key to .env:
# OPENROUTER_API_KEY=your_key_here
```

### 3. Test the LLM Client

**With UV:**
```bash
uv run python examples/check_api_key.py
uv run python examples/test_openrouter.py
```

**Traditional:**
```bash
python examples/check_api_key.py
python examples/test_openrouter.py
```

This will test various free models including:
- **Llama 3.3 70B** (best for reasoning)
- **Gemini Flash 8B** (fastest)
- **Llama 3.1 8B** (balanced)

### 4. Run Tests

**With UV:**
```bash
uv run pytest tests/
# Or use make
make test
```

**Traditional:**
```bash
pytest tests/
```

### 5. Execute Experiments

**With UV:**
```bash
uv run python experiments/run_benchmark.py
```

**Traditional:**
```bash
python experiments/run_benchmark.py
```

## Quick Commands (Makefile)

```bash
make help          # Show all available commands
make install-dev   # Setup development environment
make test          # Run tests
make format        # Format code
make lint          # Check code quality
make check-api     # Verify OpenRouter API key
```

## Available Free Models

| Model | ID | Best For |
|-------|-----|----------|
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` | Graph parsing, algorithm selection |
| Gemini Flash 8B | `google/gemini-flash-1.5-8b-exp-0827:free` | Verification, batch processing |
| Llama 3.1 8B | `meta-llama/llama-3.1-8b-instruct:free` | General purpose tasks |

## Team Members

See individual module TODOs for assigned tasks.

## Related Paper

See `project-latex/AAMAS_2025_sample.tex` for the academic paper describing this framework.
