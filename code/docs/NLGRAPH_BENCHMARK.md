# NLGraph Benchmark - Complete Guide

## What is NLGraph?

**NLGraph** is a comprehensive benchmark dataset for evaluating Large Language Models (LLMs) on graph reasoning tasks. It tests whether LLMs can understand graph structures described in natural language and perform various graph algorithms correctly.

## Dataset Structure

### 8 Task Types

NLGraph covers 8 fundamental graph algorithm categories:

1. **Shortest Path** - Find the shortest path between two nodes
2. **Connectivity** - Determine if nodes are connected
3. **Cycle Detection** - Detect cycles in graphs
4. **Topological Sort** - Order nodes in directed acyclic graphs
5. **Maximum Flow** - Calculate maximum flow in flow networks
6. **Bipartite Matching** - Find maximum matching in bipartite graphs
7. **Hamiltonian Path** - Find paths visiting all nodes exactly once
8. **GNN (Graph Neural Networks)** - Simulate message passing

### 3 Difficulty Levels

Each task has three difficulty levels:

- **Easy**: Small graphs (typically 5-10 nodes)
- **Medium**: Medium graphs (typically 10-20 nodes)
- **Hard**: Large graphs (typically 20+ nodes)

### 3 Dataset Splits

- **Train**: Training data
- **Test**: Test data for evaluation
- **Main**: Combined dataset

## Question Format

NLGraph questions are formatted in natural language with two main styles:

### Format 1: Explicit Description
```
In an undirected graph, the nodes are numbered from 0 to 6, and the edges are:
an edge between node 0 and node 1 with weight 1,
an edge between node 0 and node 6 with weight 1,
...
Q: Give the shortest path from node 4 to node 0.
A:
```

### Format 2: Compact Edge List
```
Determine if there is a path between two nodes in the graph. 
Note that (i,j) means that node i and node j are connected.
Graph: (0,1) (1,2) (2,3) (3,4)
Q: Is there a path between node 0 and node 4?
A:
```

## Dataset Statistics

Typical dataset composition:
- **Total samples**: ~3000-5000 per split
- **Per task**: ~400-600 samples
- **Per difficulty**: ~130-200 samples per task

## Running the Benchmark

### Basic Usage

Test all tasks with 10 samples each:
```bash
python run_nlgraph_benchmark.py
```

### Filter by Difficulty

Test only easy questions:
```bash
python run_nlgraph_benchmark.py --difficulty easy
```

Test only hard questions:
```bash
python run_nlgraph_benchmark.py --difficulty hard
```

### Filter by Task Type

Test specific algorithms:
```bash
python run_nlgraph_benchmark.py --tasks shortest_path connectivity
```

### Combine Filters

Test easy shortest path questions:
```bash
python run_nlgraph_benchmark.py --tasks shortest_path --difficulty easy --max-samples 20
```

### Test Different Splits

Use training data:
```bash
python run_nlgraph_benchmark.py --split train
```

## Example Questions by Task

### Shortest Path (Easy)
```
In an undirected graph, nodes are numbered from 0 to 4, edges:
edge between 0 and 1 with weight 2,
edge between 1 and 2 with weight 3,
edge between 2 and 3 with weight 1.
Q: Give the shortest path from node 0 to node 3.
A: The shortest path from node 0 to node 3 is 0,1,2,3 with total weight of 6
```

### Connectivity (Medium)
```
Graph: (0,5) (2,3) (2,5) (3,5) (4,6)
Q: Is there a path between node 2 and node 3?
A: The answer is yes.
```

### Cycle Detection (Hard)
```
In a directed graph with 15 nodes and complex edge structure...
Q: Does this graph contain a cycle?
A: The answer is yes.
```

## Evaluation Metrics

The benchmark runner calculates:

1. **Per-Task Accuracy**: Correct answers / Total samples for each task
2. **Overall Accuracy**: Correct answers / Total samples across all tasks
3. **Per-Difficulty Accuracy**: When filtering by difficulty level

## Output

Results are saved to `logs/nlgraph_results/nlgraph_results_[timestamp].json`:

```json
{
  "timestamp": "2026-01-18T11:30:00",
  "split": "test",
  "max_samples_per_task": 10,
  "summary": {
    "total_processed": 80,
    "total_correct": 65,
    "overall_accuracy": 81.25
  },
  "tasks": {
    "shortest_path": {
      "total": 10,
      "correct": 8,
      "accuracy": 80.0,
      "results": [...]
    },
    ...
  }
}
```

## Command Reference

```bash
# All options
python run_nlgraph_benchmark.py \
  --tasks shortest_path connectivity cycle \
  --max-samples 20 \
  --split test \
  --difficulty medium \
  --output-dir logs/my_results \
  --verbose

# Quick tests
python run_nlgraph_benchmark.py --difficulty easy --max-samples 5
python run_nlgraph_benchmark.py --tasks shortest_path --max-samples 50
python run_nlgraph_benchmark.py --split train --difficulty hard
```

## Tips for Evaluation

1. **Start Small**: Begin with `--max-samples 5 --difficulty easy` to test your setup
2. **Rate Limiting**: The script includes 2-second delays between requests to avoid API rate limits
3. **Task Selection**: Focus on specific tasks relevant to your use case
4. **Difficulty Progression**: Test easy → medium → hard to understand model capabilities
5. **Multiple Runs**: Run multiple times and average results for statistical significance

## Why NLGraph Matters

NLGraph is important because:
- **Real-world Relevance**: Graph reasoning is fundamental to many AI applications
- **Structured Reasoning**: Tests logical reasoning, not just pattern matching
- **Comprehensive**: Covers diverse algorithm types and difficulty levels
- **Standardized**: Enables fair comparison across different LLMs
- **Natural Language**: Tests ability to parse and understand graph descriptions

## Related Files

- `run_nlgraph_benchmark.py` - Main benchmark runner
- `src/benchmarks/nlgraph.py` - Dataset loader
- `src/agents/nlgraph_adapter.py` - Adapter for agent pipeline
- `docs/NLGRAPH_ADAPTER.md` - Adapter documentation
