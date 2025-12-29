# LLM Graph Reasoning Framework - Development Guide

## Project Overview

This is an agent-based framework for solving classical graph problems using LLMs. The system decomposes graph reasoning into explicit stages: parsing, algorithm selection, execution, and verification.

## Team Organization

### Core Agent Teams (Priority: HIGH)

#### 1. Parser Team
- **Files**: `src/agents/parser.py`
- **Tasks**: 
  - Extract graph structure from natural language
  - Handle various graph formats (directed, weighted, etc.)
  - Implement retry logic for parsing failures
- **Estimated Time**: 2-3 weeks
- **Dependencies**: Models team

#### 2. Chooser Team
- **Files**: `src/agents/chooser.py`
- **Tasks**:
  - Classify task types (connectivity, shortest path, etc.)
  - Select appropriate graph algorithms
  - Extract parameters from queries
- **Estimated Time**: 2-3 weeks
- **Dependencies**: Models team

#### 3. Verifier Team
- **Files**: `src/agents/verifier.py`
- **Tasks**:
  - Validate parsed graph structures
  - Verify algorithm outputs
  - Generate repair feedback
- **Estimated Time**: 2-3 weeks
- **Dependencies**: Models team, Algorithms team

#### 4. Orchestrator Team
- **Files**: `src/orchestrator/pipeline.py`
- **Tasks**:
  - Coordinate agent execution pipeline
  - Implement reject-and-repair loop
  - Use LangGraph for state management
- **Estimated Time**: 2-3 weeks
- **Dependencies**: All agent teams

#### 5. Models Team
- **Files**: `src/models/base.py`, `src/models/openai_client.py`, `src/models/ollama_client.py`
- **Tasks**:
  - Implement LLM client interfaces
  - Support OpenAI API and Ollama
  - Handle structured output generation
- **Estimated Time**: 1-2 weeks
- **Dependencies**: None (start first!)

### Algorithm Implementation Team (Priority: MEDIUM)

#### 6. Algorithms Team
- **Files**: `src/algorithms/*.py`
- **Sub-teams**:
  - Connectivity (BFS, DFS)
  - Shortest Path (Dijkstra, Bellman-Ford)
  - Flow (Max-flow, Min-cut)
  - Cycles (Cycle detection, Topological sort)
  - Matching (Bipartite matching)
- **Tasks**: Implement graph algorithms using NetworkX
- **Estimated Time**: 2 weeks total, 1 week per sub-team
- **Dependencies**: None (can work in parallel)

### Benchmark & Evaluation Team (Priority: MEDIUM)

#### 7. Benchmark Team
- **Files**: `src/benchmarks/*.py`
- **Tasks**:
  - Load NLGraph dataset
  - Load GraphInstruct dataset
  - Implement evaluation metrics
  - Generate reports
- **Estimated Time**: 1-2 weeks
- **Dependencies**: None initially

### Support Teams (Priority: LOW-MEDIUM)

#### 8. Utils Team
- **Files**: `src/utils/*.py`
- **Tasks**:
  - Graph utility functions
  - Logging setup
  - Configuration loading
- **Estimated Time**: 3-4 days
- **Dependencies**: None

#### 9. Experiments Team
- **Files**: `experiments/*.py`
- **Tasks**:
  - Implement experiment runner
  - Run ablation studies
  - Compare baselines
- **Estimated Time**: 1 week
- **Dependencies**: All other teams

#### 10. Testing Team
- **Files**: `tests/*.py`
- **Tasks**:
  - Unit tests for all modules
  - Integration tests
  - Test coverage reporting
- **Estimated Time**: Ongoing
- **Dependencies**: Respective module implementations

## Development Workflow

### Phase 1: Core Infrastructure (Weeks 1-2)
1. **Models Team**: Implement LLM clients first (critical dependency)
2. **Utils Team**: Set up logging and config
3. **Algorithms Team**: Start implementing graph algorithms in parallel

### Phase 2: Agent Development (Weeks 2-4)
1. **Parser Team**: Implement graph parsing with LLM assistance
2. **Chooser Team**: Implement task classification and algorithm selection
3. **Verifier Team**: Implement validation logic
4. Work can proceed in parallel with regular integration meetings

### Phase 3: Pipeline Integration (Weeks 4-6)
1. **Orchestrator Team**: Integrate agents into pipeline
2. **Testing Team**: Start integration testing
3. Debug and fix issues

### Phase 4: Benchmarking (Weeks 6-8)
1. **Benchmark Team**: Complete dataset loaders and evaluators
2. **Experiments Team**: Run full benchmark suite
3. **All Teams**: Analyze results and iterate

## Getting Started

### For New Team Members

1. **Read the Paper**: Check `project-latex/AAMAS_2025_sample.tex` for context
2. **Set Up Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Find Your TODOs**: Search for your team's TODO tags (e.g., `[PARSER-001]`)
4. **Start Small**: Begin with initialization and simple cases
5. **Test Often**: Write tests as you implement

### TODO Tag System

Each TODO is tagged with:
- **Team identifier**: `[PARSER]`, `[CHOOSER]`, `[VERIFIER]`, etc.
- **Task number**: `[PARSER-001]`, `[PARSER-002]`, etc.
- **Priority**: HIGH, MEDIUM, LOW
- **Estimated time**: Days or weeks

### Communication

- Hold weekly integration meetings
- Use shared documentation for design decisions
- Create issues/tickets for bugs and features
- Regular code reviews between team members

## Testing Strategy

1. **Unit Tests**: Each module has its own test file
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test complete pipeline
4. **Benchmark Tests**: Evaluate on real datasets

Run tests with:
```bash
pytest tests/ -v
```

## Key Design Decisions

### Why LangGraph?
- Manages stateful multi-agent coordination
- Provides checkpointing and state recovery
- Enables complex control flow

### Why NetworkX?
- Mature, well-tested graph algorithms
- Provides ground truth for verification
- Excellent documentation

### Why Pydantic?
- Type validation for data structures
- Easy serialization/deserialization
- Clear API contracts

## Common Pitfalls to Avoid

1. **Don't Mock Too Early**: Implement real functionality before testing
2. **Don't Optimize Prematurely**: Get it working first, then optimize
3. **Don't Skip Validation**: Always validate inputs and outputs
4. **Don't Ignore Edge Cases**: Test disconnected graphs, self-loops, etc.
5. **Don't Forget Logging**: Log everything for debugging

## Resources

- **NetworkX Documentation**: https://networkx.org/documentation/stable/
- **LangGraph Documentation**: https://python.langchain.com/docs/langgraph
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **NLGraph Paper**: See references in LaTeX paper

## Questions?

Check the README.md or contact the project lead.

---

**Remember**: This is a research project for an AAMAS 2025 submission. Quality and reproducibility are critical!
