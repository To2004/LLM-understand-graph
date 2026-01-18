# NLGraph Adapter for Agent System

## Overview

The NLGraph adapter enables the agent system to process questions from the NLGraph benchmark dataset. It handles the specific question format used by NLGraph and validates agent responses against expected answers.

## Features

- **Question Parsing**: Extracts graph description and task from NLGraph question format
- **Answer Validation**: Compares agent responses with expected answers
- **Batch Processing**: Process multiple NLGraph samples efficiently
- **Task Type Support**: Works with all 8 NLGraph task types (shortest_path, connectivity, cycle, flow, matching, hamilton, topology, GNN)

## Usage

### Basic Usage

```python
from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter

# Initialize pipeline and adapter
pipeline = GraphReasoningPipeline()
adapter = NLGraphAdapter(pipeline.orchestrator)

# Process a single NLGraph question
question = """In an undirected graph, the nodes are numbered from 0 to 6, and the edges are:
an edge between node 0 and node 1 with weight 1,
...
Q: Give the shortest path from node 4 to node 0.
A:"""

expected_answer = "The shortest path from node 4 to node 0 is 4,5,0 with a total weight of 5"

result = adapter.process_nlgraph_question(question, expected_answer)

print(f"Success: {result.success}")
print(f"Algorithm: {result.algorithm_used}")
print(f"Response: {result.natural_language_response}")
print(f"Matches Expected: {result.matches_expected}")
```

### Batch Processing

```python
from src.benchmarks.nlgraph import NLGraphBenchmark

# Load NLGraph benchmark
benchmark = NLGraphBenchmark('data/NLGraph/NLGraph')
benchmark.load_dataset(split='test', tasks=['shortest_path'])

# Get samples
samples = benchmark.filter_by_task('shortest_path')[:10]

# Process batch
results = adapter.run_batch(samples, max_samples=10)

print(f"Accuracy: {results['accuracy']:.1f}%")
print(f"Correct: {results['correct']}/{results['total']}")
```

## NLGraph Question Format

NLGraph questions follow two main formats:

### Format 1: Explicit Graph Description
```
In an undirected graph, the nodes are numbered from 0 to N, and the edges are:
an edge between node X and node Y with weight W,
...
Q: [Question about the graph]
A:
```

### Format 2: Compact Edge List
```
Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.
Graph: (0,1) (1,2) (2,3) ...
Q: [Question about the graph]
A:
```

## Answer Validation

The adapter validates answers using multiple strategies:

1. **Connectivity**: Matches yes/no responses
2. **Paths**: Extracts and compares node sequences
3. **Numeric**: Compares weights, flows, matching sizes
4. **Fallback**: Substring matching

## Files Created

- `src/agents/nlgraph_adapter.py` - Main adapter implementation
- `tests/test_nlgraph_integration.py` - Integration tests with full pipeline
- `tests/test_nlgraph_parsing.py` - Standalone parsing tests
- `examples/nlgraph_adapter_example.py` - Usage examples

## Testing

Run the standalone parsing test:
```bash
cd code
python tests/test_nlgraph_parsing.py
```

Run the full integration test (requires pydantic and other dependencies):
```bash
cd code
python tests/test_nlgraph_integration.py --mode single
```

Run the example script:
```bash
cd code
python examples/nlgraph_adapter_example.py
```

## Implementation Notes

- The adapter wraps the existing agent orchestrator without modifying core agents
- Maintains backward compatibility with existing query formats
- Question parsing uses regex patterns and heuristics to handle format variations
- Answer validation is approximate and may need refinement for specific task types
