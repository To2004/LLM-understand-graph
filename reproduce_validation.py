
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from src.benchmarks.nlgraph import NLGraphBenchmark
from src.agents.nlgraph_adapter import NLGraphAdapter

# Mock orchestrator
class MockOrchestrator:
    def execute(self, *args, **kwargs):
        pass

def reproduction():
    benchmark = NLGraphBenchmark(Path("code/../data/NLGraph/NLGraph"))
    benchmark.load_dataset(split='test', tasks=['shortest_path'])
    
    samples = benchmark.filter_by_task('shortest_path')
    # Filter by difficulty as in the log
    samples = [s for s in samples if s.get('difficulty') == 'easy']
    
    # The log said "Processing sample 3", but indices might be 0-based.
    # The log showed "[1/1] Processing sample 3..."
    # If max-samples was 1 and it processed sample 3, it probably sliced it or it was the 3rd sample in the filtered list.
    # Let's verify what sample 3 is.
    
    print(f"Total samples: {len(samples)}")

    # Print first few samples to debug
    print("\nListing first 5 samples:")
    for i, s in enumerate(samples[:5]):
        print(f"Sample {i}: ID={s.get('id')}")
        print(f"Question snippet: {s.get('question', '')[:100]}...")
        print("-" * 40)
    
    target_sample = None
    for s in samples:
        if s['id'] == '3':
            target_sample = s
            break
            
    if target_sample:
        print("\nFound target sample:")
        print(f"ID: {target_sample['id']}")
        print(f"Question: {target_sample['question'][:100]}...")
        print(f"Expected Answer: {target_sample['answer']}")
        
        adapter = NLGraphAdapter(MockOrchestrator())
        
        # Test validation with the answer from the log
        # Execution result: (['7', '8', '3'], 3.0)
        # Synthesized response: The shortest path is: ['7', '8', '3'] ? 3.0
        
        agent_response = "The shortest path is: ['7', '8', '3'] ? 3.0"
        expected_answer = target_sample['answer']
        
        print(f"\nEvaluating:")
        print(f"Agent Response: {agent_response}")
        print(f"Expected Answer: {expected_answer}")
        
        is_match = adapter._validate_answer(agent_response, expected_answer)
        print(f"Match: {is_match}")
        
        # Debug extraction
        print(f"Agent Path extracted: {adapter._extract_path(adapter._normalize_answer(agent_response))}")
        print(f"Expected Path extracted: {adapter._extract_path(adapter._normalize_answer(expected_answer))}")
        
    else:
        print("Could not find the target sample matching the log description.")

if __name__ == "__main__":
    reproduction()
