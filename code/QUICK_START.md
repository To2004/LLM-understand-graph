# Quick Setup Guide for NLGraph Benchmark

## Step 1: Install Dependencies in Virtual Environment

Since you're using a virtual environment, install all dependencies there:

```powershell
# Make sure you're in the venv (you should see (llm-graph-reasoning) in your prompt)
cd c:\Users\user\Documents\GitHub\LLM-understand-graph\code

# Install all dependencies
pip install pydantic networkx openai python-dotenv requests scipy loguru pyyaml
```

## Step 2: Run the Benchmark

```powershell
# Test with easy difficulty, 3 samples per task
python run_nlgraph_benchmark.py --difficulty easy --max-samples 3

# Or test specific tasks
python run_nlgraph_benchmark.py --tasks shortest_path connectivity --difficulty easy --max-samples 3
```

## Step 3: Test with Different Models

The benchmark uses the model specified in your pipeline. To test different models, you can:

### Option A: Modify the benchmark script to test multiple models

Create `run_multi_model_benchmark.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter
from src.benchmarks.nlgraph import NLGraphBenchmark
from pathlib import Path

# Models to test
MODELS_TO_TEST = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

for model in MODELS_TO_TEST:
    print(f"\n{'='*80}")
    print(f"Testing Model: {model}")
    print(f"{'='*80}\n")
    
    # Initialize pipeline with specific model
    pipeline = GraphReasoningPipeline()
    pipeline.llm_client.model_name = model
    
    adapter = NLGraphAdapter(pipeline.orchestrator)
    benchmark = NLGraphBenchmark(Path("../data/NLGraph/NLGraph"))
    
    # Run small test
    benchmark.load_dataset(split='test', tasks=['connectivity'])
    samples = benchmark.filter_by_task('connectivity')[:3]
    
    results = adapter.run_batch(samples)
    print(f"\nModel: {model}")
    print(f"Accuracy: {results['accuracy']:.1f}%")
```

### Option B: Use command line with different models

Modify `run_nlgraph_benchmark.py` to accept a model parameter:

Add this argument to the argparse section:
```python
parser.add_argument(
    "--model",
    type=str,
    default="meta-llama/llama-3.3-70b-instruct:free",
    help="Model to use for evaluation"
)
```

Then update the pipeline initialization:
```python
from src.llms import OpenRouterClient
llm_client = OpenRouterClient(model_name=args.model)
pipeline = GraphReasoningPipeline(llm_client=llm_client, verbose=verbose)
```

## Available Free Models on OpenRouter

- `meta-llama/llama-3.3-70b-instruct:free` - Llama 3.3 70B (default)
- `deepseek/deepseek-r1:free` - DeepSeek R1 (reasoning model)
- `google/gemini-2.0-flash-exp:free` - Gemini 2.0 Flash
- `qwen/qwen-2.5-72b-instruct:free` - Qwen 2.5 72B
- `mistralai/mistral-7b-instruct:free` - Mistral 7B

## Troubleshooting

### If scipy import fails in venv:
```powershell
pip install --upgrade scipy
```

### If you get "No module named 'X'" errors:
```powershell
pip install pydantic networkx openai python-dotenv requests scipy loguru pyyaml
```

### To verify all packages are installed:
```powershell
python -c "import pydantic, networkx, openai, scipy, loguru, yaml; print('All packages installed!')"
```

## Quick Test Commands

```powershell
# Quick test (1 sample)
python run_nlgraph_benchmark.py --difficulty easy --max-samples 1 --tasks connectivity

# Full easy test (3 samples per task)
python run_nlgraph_benchmark.py --difficulty easy --max-samples 3

# Test all difficulties
python run_nlgraph_benchmark.py --difficulty easy --max-samples 3
python run_nlgraph_benchmark.py --difficulty medium --max-samples 3
python run_nlgraph_benchmark.py --difficulty hard --max-samples 3
```
