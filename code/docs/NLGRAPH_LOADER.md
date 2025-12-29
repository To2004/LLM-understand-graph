# NLGraph Dataset Loader Implementation

## Overview

The `nlgraph.py` module provides a complete implementation for loading and working with the NLGraph benchmark dataset. It supports all 8 task types and provides filtering capabilities by task, difficulty, and graph complexity.

## Features

### Supported Task Types
- `shortest_path` - Find shortest paths in weighted graphs
- `connectivity` - Determine if nodes are connected
- `cycle` - Detect cycles in graphs
- `flow` - Maximum flow problems
- `matching` - Maximum matching problems
- `hamilton` - Hamilton path/cycle detection
- `topology` - Topology property identification
- `GNN` - Graph neural network tasks

### Core Functionality

1. **Dataset Loading**
   - Load test, train, or main splits
   - Filter by specific task types
   - Parse JSON files with questions and answers

2. **Filtering Capabilities**
   - Filter by task type
   - Filter by difficulty (easy, medium, hard)
   - Filter by graph complexity (node count)

3. **Graph Parsing**
   - Parse NetworkX graphs from natural language questions
   - Load graphs from structured text files
   - Extract graph metadata (nodes, edges, weights)

4. **Statistics & Analysis**
   - Get dataset statistics
   - Task distribution
   - Difficulty distribution

## Usage

### Basic Usage

```python
from pathlib import Path
from benchmarks.nlgraph import NLGraphBenchmark

# Initialize loader
data_path = Path('data/NLGraph/NLGraph')
benchmark = NLGraphBenchmark(data_path)

# Load test split
samples = benchmark.load_dataset(split='test')
print(f"Loaded {len(samples)} samples")

# Get statistics
stats = benchmark.get_task_statistics()
print(stats)
```

### Loading Specific Tasks

```python
# Load only shortest path and connectivity tasks
samples = benchmark.load_dataset(
    split='test',
    tasks=['shortest_path', 'connectivity']
)
```

### Filtering Samples

```python
# Filter by task
shortest_path_samples = benchmark.filter_by_task('shortest_path')

# Filter by difficulty
easy_samples = benchmark.filter_by_difficulty('easy')

# Filter by complexity (node count)
small_graphs = benchmark.filter_by_complexity(min_nodes=0, max_nodes=10)
```

### Accessing Individual Samples

```python
# Get a specific sample
sample = benchmark.get_sample(0)

print(sample['id'])         # Sample ID
print(sample['task'])       # Task type
print(sample['question'])   # Question text
print(sample['answer'])     # Ground truth answer
print(sample['difficulty']) # Difficulty level
print(sample['graph'])      # NetworkX graph (if parsed)
```

### Loading Graph Files

```python
# Load graph from file
graph = benchmark.load_graph_file(
    task='shortest_path',
    difficulty='easy',
    graph_id=0
)
```

## Data Structure

### Sample Dictionary
Each sample contains:
```python
{
    'id': str,              # Sample identifier
    'task': str,            # Task type
    'question': str,        # Question text in natural language
    'answer': str,          # Ground truth answer
    'difficulty': str,      # 'easy', 'medium', 'hard', or 'unknown'
    'graph': nx.Graph       # NetworkX graph (if parsed)
}
```

### Dataset Statistics
```python
{
    'total_samples': int,
    'tasks': {
        'shortest_path': count,
        'connectivity': count,
        ...
    },
    'difficulties': {
        'easy': count,
        'medium': count,
        'hard': count
    }
}
```

## Integration with Evaluator

The NLGraph loader is designed to work seamlessly with the BenchmarkEvaluator:

```python
from benchmarks.nlgraph import NLGraphBenchmark
from benchmarks.evaluator import BenchmarkEvaluator

# Load dataset
benchmark = NLGraphBenchmark('data/NLGraph/NLGraph')
samples = benchmark.load_dataset(split='test')

# Extract data for evaluation
predictions = [...]  # Your model predictions
ground_truths = [s['answer'] for s in samples]
metadata = [{
    'task': s['task'],
    'difficulty': s['difficulty'],
    'graph': s.get('graph')
} for s in samples]

# Evaluate
evaluator = BenchmarkEvaluator()
metrics = evaluator.evaluate(predictions, ground_truths, metadata)
```

## File Structure

The NLGraph dataset has the following structure:
```
data/NLGraph/NLGraph/
├── shortest_path/
│   ├── test.json        # Test questions and answers
│   ├── train.json       # Training questions and answers
│   ├── main.json        # Main dataset
│   ├── graph/           # Graph files
│   │   ├── easy/
│   │   │   └── standard/
│   │   │       ├── graph0.txt
│   │   │       ├── graph1.txt
│   │   │       └── ...
│   │   └── hard/
│   └── prompt/          # Prompting templates
├── connectivity/
├── cycle/
├── flow/
├── matching/
├── hamilton/
├── topology/
└── GNN/
```

## Graph File Format

Graph files (`.txt`) use the following format:
```
n m
u1 v1 [weight1]
u2 v2 [weight2]
...
source target
```

Where:
- `n` = number of nodes
- `m` = number of edges
- Each edge line: `u v [weight]`
- Last line: query nodes (for shortest path)

## Error Handling

The implementation includes robust error handling:
- Validates dataset path on initialization
- Gracefully handles missing files
- Prints warnings for missing tasks
- Returns empty lists for invalid queries
- Catches parsing errors and continues

## Dependencies

Required packages:
- `networkx` - For graph data structures
- `json` - For loading JSON files (built-in)
- `pathlib` - For path handling (built-in)

Install networkx:
```bash
pip install networkx
```

## Testing

Run the example script to test the implementation:
```bash
cd code
python examples/test_nlgraph_loader.py
```

## Notes

1. **Case Sensitivity**: The task type 'GNN' uses uppercase to match the directory name
2. **Graph Parsing**: Not all questions have parseable graphs; the loader attempts parsing but returns None on failure
3. **Memory**: Loading all tasks at once may consume significant memory for large datasets
4. **Splits**: The loader supports 'test', 'train', and 'main' splits based on available JSON files
