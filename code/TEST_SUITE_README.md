# Algorithm Test Suite

This directory contains a comprehensive test suite for all supported graph algorithms in the LLM Graph Reasoning Framework.

## Quick Start

### Run All Tests

Test all 8 algorithm types with 24 total test cases:

```bash
uv run .\code\test_all_algorithms.py
```

### Run Specific Algorithm Test

Test a specific algorithm type with a single query:

```bash
# Format: --single ALGORITHM_TYPE:INDEX
uv run .\code\test_all_algorithms.py --single SHORTEST_PATH:0
uv run .\code\test_all_algorithms.py --single CONNECTIVITY:1
```

### Quiet Mode

Run tests with reduced verbosity:

```bash
uv run .\code\test_all_algorithms.py --quiet
```

## Supported Algorithms

The test suite covers all 8 supported algorithm types:

### 1. **CONNECTIVITY**
- Tests graph connectivity and reachability
- Example: "Are nodes A and C connected?"

### 2. **SHORTEST_PATH**
- Tests shortest path finding (Dijkstra, BFS, Bellman-Ford)
- Example: "What's the shortest path from A to D?"

### 3. **CYCLE_DETECTION**
- Tests cycle detection in directed and undirected graphs
- Example: "Does this graph have a cycle?"

### 4. **TOPOLOGICAL_SORT**
- Tests topological ordering of DAGs
- Example: "What's the topological order of these tasks?"

### 5. **MAXIMUM_FLOW**
- Tests max flow algorithms (Ford-Fulkerson, Edmonds-Karp)
- Example: "What's the maximum flow from S to T?"

### 6. **BIPARTITE_MATCHING**
- Tests maximum matching in bipartite graphs
- Example: "Find the best worker-to-job assignment"

### 7. **HAMILTONIAN_PATH**
- Tests Hamiltonian path/cycle detection
- Example: "Find a path visiting all nodes exactly once"

### 8. **GNN_MESSAGE_PASSING**
- Tests graph neural network message passing simulation
- Example: "Simulate message passing from A with initial value 1"

## Test Structure

Each algorithm type has 3 test queries covering:
- Basic functionality
- Edge cases (weighted graphs, directed graphs, etc.)
- Real-world scenarios

## Output Format

The test suite provides:
- ✅ Success/❌ Failure indicators for each test
- Natural language response from the pipeline
- Algorithm used for each query
- Summary statistics (pass rate, per-algorithm results)

## Example Output

```
================================================================================
Testing: SHORTEST_PATH
================================================================================

[Test 1/3] Query: Graph: A->B, B->C, C->D. What's the shortest path from A to D?...
✅ SUCCESS
   Response: The shortest path from A to D is: A → B → C → D with length 3
   Algorithm: dijkstra

[Test 2/3] Query: Graph: 1--2 (weight 5), 1--3 (weight 2), 3--2 (weight 1)...
✅ SUCCESS
   Response: The shortest path from 1 to 2 is: 1 → 3 → 2 with total weight 3
   Algorithm: dijkstra
```

## Interactive Testing

For interactive testing with custom queries, use:

```bash
uv run .\code\simple_interactive.py
```

## Troubleshooting

### API Key Issues

Make sure your `.env` file contains:
```
OPENROUTER_API_KEY=your_key_here
```

### Import Errors

All import errors have been fixed. If you encounter any, ensure you're running from the project root:
```bash
cd "C:\Users\user\OneDrive - post.bgu.ac.il\Courses-Drive\sem7\Research Methods\LLM-understand-graph"
uv run .\code\test_all_algorithms.py
```

### Verbose Debugging

To see detailed execution logs for debugging, run without `--quiet`:
```bash
uv run .\code\test_all_algorithms.py
```

This will show:
- Phase-by-phase execution (Validation, Parsing, Algorithm Selection, Execution, Synthesis)
- Parser node/edge extraction
- Chooser task classification and algorithm selection
- Execution results
- Synthesis process

## Adding New Tests

To add new test cases, edit `test_all_algorithms.py` and add queries to the `TEST_QUERIES` dictionary:

```python
TEST_QUERIES = {
    "SHORTEST_PATH": [
        "Your new test query here...",
        # ... existing queries
    ],
}
```
