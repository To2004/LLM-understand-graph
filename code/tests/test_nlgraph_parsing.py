"""
Standalone NLGraph Adapter Test

This script tests the NLGraph adapter's parsing functionality
by directly importing the module file.
"""

import sys
import os
import importlib.util

# Load the nlgraph_adapter module directly
adapter_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'src', 
    'agents', 
    'nlgraph_adapter.py'
)

spec = importlib.util.spec_from_file_location("nlgraph_adapter", adapter_path)
nlgraph_adapter = importlib.util.module_from_spec(spec)

# Mock pydantic.BaseModel before loading
class MockBaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

sys.modules['pydantic'] = type(sys)('pydantic')
sys.modules['pydantic'].BaseModel = MockBaseModel

# Now load the module
spec.loader.exec_module(nlgraph_adapter)

NLGraphAdapter = nlgraph_adapter.NLGraphAdapter


def test_question_parsing():
    """Test question parsing without full pipeline"""
    print("\n" + "="*80)
    print("Testing NLGraph Question Parsing")
    print("="*80)
    
    # Create a mock orchestrator for testing
    class MockOrchestrator:
        pass
    
    adapter = NLGraphAdapter(MockOrchestrator())
    
    # Test Case 1: Shortest Path Question
    print("\n" + "-"*80)
    print("Test 1: Shortest Path Question")
    print("-"*80)
    
    question1 = """In an undirected graph, the nodes are numbered from 0 to 6, and the edges are:
an edge between node 0 and node 1 with weight 1,
an edge between node 0 and node 6 with weight 1,
an edge between node 0 and node 5 with weight 1.
Q: Give the shortest path from node 4 to node 0.
A:"""
    
    graph_ctx, task_ctx = adapter.extract_graph_and_task(question1)
    
    print(f"Graph Context:\n{graph_ctx[:150]}...")
    print(f"\nTask Context:\n{task_ctx}")
    
    assert "nodes are numbered" in graph_ctx.lower()
    assert "shortest path" in task_ctx.lower()
    print("PASS Test 1")
    
    # Test Case 2: Connectivity Question
    print("\n" + "-"*80)
    print("Test 2: Connectivity Question")
    print("-"*80)
    
    question2 = """Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.
Graph: (0,6) (0,3) (0,2) (0,1) (1,6) (1,3) (1,2) (2,6) (2,3) (3,6) (4,5)
Q: Is there a path between node 0 and node 5?
A:"""
    
    graph_ctx, task_ctx = adapter.extract_graph_and_task(question2)
    
    print(f"Graph Context:\n{graph_ctx[:150]}...")
    print(f"\nTask Context:\n{task_ctx}")
    
    assert "graph" in graph_ctx.lower()
    assert "path" in task_ctx.lower()
    print("PASS Test 2")
    
    # Test Case 3: Answer Validation
    print("\n" + "-"*80)
    print("Test 3: Answer Validation")
    print("-"*80)
    
    # Test connectivity answer
    agent_response1 = "Yes, there is a path between node 0 and node 5"
    expected1 = "The answer is yes."
    
    matches1 = adapter._validate_answer(agent_response1, expected1)
    print(f"Agent: '{agent_response1}'")
    print(f"Expected: '{expected1}'")
    print(f"Matches: {matches1}")
    assert matches1, "Should match 'yes' answers"
    print("PASS Test 3a")
    
    # Test path answer
    agent_response2 = "The shortest path is 4,5,0 with weight 5"
    expected2 = "The shortest path from node 4 to node 0 is 4,5,0 with a total weight of 5"
    
    matches2 = adapter._validate_answer(agent_response2, expected2)
    print(f"\nAgent: '{agent_response2}'")
    print(f"Expected: '{expected2}'")
    print(f"Matches: {matches2}")
    assert matches2, "Should match path answers"
    print("PASS Test 3b")
    
    print("\n" + "="*80)
    print("All Tests Passed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        test_question_parsing()
    except AssertionError as e:
        print(f"\nFAIL Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
