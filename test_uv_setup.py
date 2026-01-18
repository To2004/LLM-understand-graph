"""Test script to verify uv sync worked correctly"""

try:
    import networkx as nx
    import pydantic
    import langchain
    import numpy as np
    import pandas as pd
    
    print("✓ All core dependencies imported successfully!")
    print(f"  - NetworkX: {nx.__version__}")
    print(f"  - Pydantic: {pydantic.__version__}")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - Pandas: {pd.__version__}")
    
    # Test importing project modules
    import sys
    sys.path.insert(0, 'code')
    
    from src.agents import AgentParser, AgentChooser, AgentSynthesizer
    from src.agents import AgentOrchestrator, PromptValidator, PromptDecomposer
    
    print("✓ All project modules imported successfully!")
    print("\n✅ uv sync worked perfectly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
