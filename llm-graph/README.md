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

1. Install dependencies: `pip install -r requirements.txt`
2. Configure models in `configs/models.yaml`
3. Run tests: `pytest tests/`
4. Execute experiments: `python experiments/run_benchmark.py`

## Team Members

See individual module TODOs for assigned tasks.

## Related Paper

See `project-latex/AAMAS_2025_sample.tex` for the academic paper describing this framework.
