# Dependencies Required for NLGraph Benchmark

## Required Python Packages

To run the NLGraph benchmark with your agent pipeline, you need to install the following dependencies:

```bash
pip install pydantic networkx openai python-dotenv
```

### Package Purposes

- **pydantic**: Data validation and settings management (used throughout the agent system)
- **networkx**: Graph data structures and algorithms (used by the executor)
- **openai**: OpenAI API client (for LLM calls)
- **python-dotenv**: Load environment variables from .env file

## Installation Steps

### Option 1: Install All at Once
```bash
pip install pydantic networkx openai python-dotenv
```

### Option 2: Install from Requirements File (if available)
```bash
pip install -r requirements.txt
```

## Verify Installation

After installation, verify the packages are installed:

```bash
python -c "import pydantic; import networkx; import openai; print('All dependencies installed!')"
```

## Running the Benchmark After Installation

Once dependencies are installed, you can run:

### Quick Test (3 samples per task, easy difficulty)
```bash
cd c:\Users\user\Documents\GitHub\LLM-understand-graph\code
python run_nlgraph_benchmark.py --difficulty easy --max-samples 3
```

### Comprehensive Test (3 samples per task per difficulty)
```bash
cd c:\Users\user\Documents\GitHub\LLM-understand-graph\code
python run_comprehensive_benchmark.py
```

### Manual Run for Each Difficulty
```bash
# Easy
python run_nlgraph_benchmark.py --difficulty easy --max-samples 3

# Medium
python run_nlgraph_benchmark.py --difficulty medium --max-samples 3

# Hard
python run_nlgraph_benchmark.py --difficulty hard --max-samples 3
```

## Environment Setup

Make sure you have your OpenAI API key configured:

1. Create a `.env` file in the code directory:
```bash
OPENAI_API_KEY=your-api-key-here
```

2. Or set it as an environment variable:
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

## Troubleshooting

### If pydantic import fails:
```bash
pip install --upgrade pydantic
```

### If networkx import fails:
```bash
pip install --upgrade networkx
```

### If you get API errors:
- Check your OpenAI API key is set correctly
- Verify you have API credits available
- Check your internet connection

## Expected Runtime

For the comprehensive benchmark (3 samples × 8 tasks × 3 difficulties = 72 samples):
- With 2-second delays between requests: ~2-3 minutes per difficulty level
- Total time: ~6-10 minutes for all three difficulty levels

## Next Steps

1. Install dependencies: `pip install pydantic networkx openai python-dotenv`
2. Set up your API key in `.env` file
3. Run the benchmark: `python run_nlgraph_benchmark.py --difficulty easy --max-samples 3`
