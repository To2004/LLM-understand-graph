# Interactive Pipeline Scripts

Two interactive scripts for testing the graph reasoning pipeline.

## 🚀 Quick Start

### Option 1: Simple Mode (Recommended for Quick Testing)

```bash
cd code
python simple_interactive.py
```

**Features:**
- ✅ Minimal interface
- ✅ Just type queries and get answers
- ✅ Type 'exit' to quit

**Example Session:**
```
Graph Reasoning Pipeline - Simple Mode
============================================================
Initializing pipeline...
✅ Ready! (Type 'exit' to quit)

📝 Your query: Graph: A--B--C. Is A connected to C?

⏳ Processing...

------------------------------------------------------------
✅ Yes, the nodes are connected.
   (Algorithm: bfs_connectivity)
------------------------------------------------------------

📝 Your query: exit

👋 Goodbye!
```

---

### Option 2: Full Interactive Mode (Advanced Features)

```bash
cd code
python interactive_pipeline.py
```

**Features:**
- ✅ Full-featured REPL interface
- ✅ Built-in help and examples
- ✅ Session statistics
- ✅ Verbose mode toggle
- ✅ Query history tracking

**Commands:**
- `help` - Show example queries
- `stats` - Show session statistics
- `verbose` - Toggle verbose mode
- `clear` - Clear screen
- `exit` or `quit` - Exit

**Example Session:**
```
======================================================================
  🧠 LLM GRAPH REASONING PIPELINE - Interactive Mode
======================================================================

Using: Llama 3.3 70B via OpenRouter (free)

Commands:
  - Type your graph query and press Enter
  - Type 'help' for example queries
  - Type 'stats' to see usage statistics
  - Type 'verbose' to toggle verbose mode
  - Type 'exit' or 'quit' to exit

======================================================================

🔄 Initializing pipeline...
✅ Pipeline ready!

🔍 Enter your query (or 'help'): help

======================================================================
  📚 EXAMPLE QUERIES
======================================================================

1. Connectivity:
   Graph: A--B--C. Is A connected to C?

2. Shortest Path:
   Graph: A--B--C--D. Find shortest path from A to D.

3. Cycle Detection:
   Graph: A->B, B->C, C->A. Does this graph have a cycle?

4. Path Finding:
   Given nodes 1,2,3,4 with edges 1-2, 2-3, 3-4. Find path from 1 to 4.

5. Complex Query:
   In a directed graph with edges A->B, B->C, C->D, D->B, is there a cycle?

======================================================================

🔍 Enter your query (or 'help'): Graph: A--B--C. Is A connected to C?

⏳ Processing query #1...

----------------------------------------------------------------------
📊 RESULT #1
----------------------------------------------------------------------
✅ Success: True

💬 Response:
   Yes, the nodes are connected.

🔧 Algorithm: bfs_connectivity

📈 Metadata:
   validation_confidence: 0.95
   decomposition_confidence: 0.9
   algorithm_confidence: 0.85
----------------------------------------------------------------------

🔍 Enter your query (or 'help'): stats

======================================================================
  📊 SESSION STATISTICS
======================================================================

Total Queries: 1
Successful: 1
Failed: 0
Success Rate: 100.0%

LLM Usage:
  Total Requests: 5
  Total Tokens: 2,450
======================================================================

🔍 Enter your query (or 'help'): exit

👋 Thanks for using the Graph Reasoning Pipeline!

======================================================================
  📊 SESSION STATISTICS
======================================================================

Total Queries: 1
Successful: 1
Failed: 0
Success Rate: 100.0%
======================================================================
```

---

## 📝 Example Queries to Try

### Connectivity
```
Graph: A--B--C. Is A connected to C?
Graph: A--B, C--D. Is A connected to D?
```

### Shortest Path
```
Graph: A--B--C--D. Find shortest path from A to D.
Given nodes 1,2,3,4,5 with edges 1-2, 2-3, 3-4, 4-5. What is the shortest path from 1 to 5?
```

### Cycle Detection
```
Graph: A->B, B->C, C->A. Does this graph have a cycle?
In a directed graph with edges A->B, B->C, is there a cycle?
```

### Weighted Graphs
```
Graph with nodes A,B,C and weighted edges: A-B (weight 5), B-C (weight 3). Find shortest path from A to C.
```

### Complex Queries
```
In a directed graph where node 1 points to 2, 2 points to 3, 3 points to 4, and 4 points back to 2, detect if there's a cycle.
```

---

## ⚙️ Prerequisites

1. **Set up API key** in `.env`:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
**Solution**: Make sure you're running from the `code` directory:
```bash
cd code
python simple_interactive.py
```

### "ValueError: Open Router API key not found"
**Solution**: Add your API key to `.env` file in the project root:
```bash
OPENROUTER_API_KEY=your_key_here
```

### Pipeline takes too long
**Solution**: This is normal - the LLM calls take 1-2 seconds each. Total time is ~6-10 seconds per query.

---

## 💡 Tips

1. **Start with simple queries** to test the pipeline
2. **Use 'help' command** to see example queries
3. **Enable verbose mode** to see detailed pipeline execution
4. **Check stats** to monitor your usage
5. **Try different graph formats** - the parser is flexible

---

## 🎯 What Each Script Does

### `simple_interactive.py`
- Minimal interface
- Just input → output
- Perfect for quick testing
- ~50 lines of code

### `interactive_pipeline.py`
- Full-featured REPL
- Help, stats, verbose mode
- Session tracking
- Error handling
- ~200 lines of code

---

Choose the one that fits your needs!
