# Using OpenRouter with the Pipeline

## Quick Start

The pipeline now defaults to using **Llama 3.3 70B** from OpenRouter, which is free and high-performance.

### 1. Set Up Your API Key

```bash
# Add to your .env file
OPENROUTER_API_KEY=your_key_here
```

Get your free API key at: https://openrouter.ai/

### 2. Simplest Usage (Auto-defaults to Llama 3.3 70B)

```python
from src.pipeline import GraphReasoningPipeline

# No need to specify anything!
pipeline = GraphReasoningPipeline()

result = pipeline.run("Graph: A--B--C. Is A connected to C?")
print(result.natural_language_response)
```

### 3. Quick One-Liner

```python
from src.pipeline import quick_query

response = quick_query("Graph: A--B--C. Is A connected to C?")
print(response)
```

---

## Advanced: Using Different Models Per Agent

You can assign specialized models to each agent based on their strengths:

```python
from src.pipeline import GraphReasoningPipeline

agent_models = {
    "parser": "meta-llama/llama-3.3-70b-instruct:free",      # Best reasoning
    "chooser": "deepseek/deepseek-r1-0528:free",             # Advanced reasoning
    "synthesizer": "google/gemma-3-27b-it:free",             # Good NL generation
    "validator": "xiaomi/mimo-v2-flash:free",                # Fast validation
    "decomposer": "meta-llama/llama-3.3-70b-instruct:free"   # Structure understanding
}

pipeline = GraphReasoningPipeline(agent_models=agent_models, verbose=True)
result = pipeline.run("Your query here")
```

---

## Available Free Models on OpenRouter

All models below are **completely free** to use:

| Model | ID | Best For |
|-------|-----|----------|
| **Llama 3.3 70B** (Default) | `meta-llama/llama-3.3-70b-instruct:free` | Overall best - reasoning, parsing, general tasks |
| **DeepSeek R1** | `deepseek/deepseek-r1-0528:free` | Advanced reasoning, algorithm selection |
| **Devstral 2** | `mistralai/devstral-2512:free` | Code generation, 256K context |
| **MiMo V2 Flash** | `xiaomi/mimo-v2-flash:free` | Fast responses, large context |
| **Gemma 3 27B** | `google/gemma-3-27b-it:free` | Balanced, instruction following |
| **Qwen 3 Coder** | `qwen/qwen3-coder:free` | Code tasks, technical reasoning |

See `src/models/model_configs.py` for the complete list with details.

---

## Recommended Model Assignments by Task

Based on model strengths:

```python
RECOMMENDED_MODELS = {
    "parser": "meta-llama/llama-3.3-70b-instruct:free",    # Best reasoning for graph parsing
    "chooser": "deepseek/deepseek-r1-0528:free",           # Advanced reasoning for algorithm selection  
    "synthesizer": "google/gemma-3-27b-it:free",           # Good at natural language generation
    "validator": "xiaomi/mimo-v2-flash:free",              # Very fast for validation
    "decomposer": "meta-llama/llama-3.3-70b-instruct:free" # Good at understanding structure
}
```

---

## Examples

### Example 1: Basic Usage (Default Model)

```python
from src.pipeline import GraphReasoningPipeline

pipeline = GraphReasoningPipeline(verbose=True)
result = pipeline.run("Graph: A--B--C--D. Find shortest path from A to D.")

print(result.natural_language_response)
# Output: "The shortest path from A to D is: A → B → C → D"
```

### Example 2: Custom Model for All Agents

```python
from src.pipeline import GraphReasoningPipeline
from src.models import OpenRouterClient

# Use DeepSeek R1 for all agents
client = OpenRouterClient(model_name="deepseek/deepseek-r1-0528:free")
pipeline = GraphReasoningPipeline(client)

result = pipeline.run("Your query")
```

### Example 3: Different Models Per Agent

```python
from src.pipeline import GraphReasoningPipeline

agent_models = {
    "parser": "meta-llama/llama-3.3-70b-instruct:free",
    "chooser": "deepseek/deepseek-r1-0528:free",
    "synthesizer": "google/gemma-3-27b-it:free"
}

pipeline = GraphReasoningPipeline(agent_models=agent_models)
result = pipeline.run("Your query")
```

### Example 4: Batch Processing

```python
from src.pipeline import GraphReasoningPipeline

pipeline = GraphReasoningPipeline()

queries = [
    "Graph: A--B--C. Is A connected to C?",
    "Graph: A--B, C--D. Is A connected to D?",
    "Graph: A--B--C--D. Find shortest path from A to D."
]

results = pipeline.run_batch(queries)
for i, result in enumerate(results):
    print(f"Query {i+1}: {result.natural_language_response}")
```

---

## Why OpenRouter?

✅ **Free tier** with powerful models (Llama 3.3 70B, DeepSeek R1, etc.)  
✅ **No local setup** required (unlike Ollama)  
✅ **High performance** - 70B+ parameter models  
✅ **Large context windows** - up to 256K tokens  
✅ **Multiple model options** - choose the best for each task  
✅ **Simple API** - OpenAI-compatible interface  

---

## Troubleshooting

### API Key Not Found

```
ValueError: Open Router API key not found.
```

**Solution**: Add your API key to `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
```

### Rate Limiting

If you hit rate limits, try:
1. Using a smaller model (e.g., `google/gemma-3-27b-it:free`)
2. Adding delays between requests
3. Upgrading to a paid OpenRouter plan

---

## Performance Tips

1. **Use Llama 3.3 70B for critical agents** (parser, chooser) - best reasoning
2. **Use faster models for validation** (MiMo V2 Flash) - speed matters
3. **Use specialized models** - DeepSeek R1 for reasoning, Qwen for code
4. **Enable verbose mode** during development to see what's happening
5. **Batch similar queries** to reduce overhead

---

## Next Steps

- See `examples/basic_usage.py` for a complete working example
- See `examples/advanced_usage.py` for multi-model configuration
- Check `src/models/model_configs.py` for all available models
- Visit https://openrouter.ai/docs for API documentation

---

*Updated: 2026-01-12*
