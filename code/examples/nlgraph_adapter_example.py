"""
Example: Using NLGraph Adapter with the Agent Pipeline

This script demonstrates how to use the NLGraph adapter to process
NLGraph benchmark questions.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter
from src.benchmarks.nlgraph import NLGraphBenchmark


def main():
    print("\n" + "="*80)
    print("NLGraph Adapter Example")
    print("="*80)
    
    # Initialize pipeline
    print("\n1. Initializing Graph Reasoning Pipeline...")
    pipeline = GraphReasoningPipeline(verbose=False)
    
    # Create NLGraph adapter
    print("2. Creating NLGraph Adapter...")
    adapter = NLGraphAdapter(pipeline.orchestrator)
    
    # Example 1: Process a single question
    print("\n" + "-"*80)
    print("Example 1: Single NLGraph Question")
    print("-"*80)
    
    question = """In an undirected graph, the nodes are numbered from 0 to 6, and the edges are:
an edge between node 0 and node 1 with weight 1,
an edge between node 0 and node 6 with weight 1,
an edge between node 0 and node 5 with weight 1,
an edge between node 1 and node 2 with weight 3,
an edge between node 1 and node 6 with weight 4,
an edge between node 1 and node 5 with weight 3,
an edge between node 2 and node 4 with weight 3,
an edge between node 2 and node 5 with weight 2,
an edge between node 3 and node 5 with weight 3,
an edge between node 4 and node 5 with weight 4.
Q: Give the shortest path from node 4 to node 0.
A:"""
    
    expected = "The shortest path from node 4 to node 0 is 4,5,0 with a total weight of 5"
    
    print(f"\nQuestion: {question[:100]}...")
    print(f"Expected: {expected}")
    
    result = adapter.process_nlgraph_question(question, expected)
    
    print(f"\n✅ Result:")
    print(f"   Success: {result.success}")
    print(f"   Algorithm: {result.algorithm_used}")
    print(f"   Response: {result.natural_language_response}")
    print(f"   Matches Expected: {result.matches_expected}")
    
    # Example 2: Load from benchmark and process batch
    print("\n" + "-"*80)
    print("Example 2: Batch Processing from Benchmark")
    print("-"*80)
    
    data_path = Path(__file__).parent.parent / 'data' / 'NLGraph' / 'NLGraph'
    
    if data_path.exists():
        print(f"\nLoading NLGraph benchmark from: {data_path}")
        benchmark = NLGraphBenchmark(data_path)
        
        # Load connectivity samples
        benchmark.load_dataset(split='test', tasks=['connectivity'])
        samples = benchmark.filter_by_task('connectivity')[:3]
        
        print(f"Loaded {len(samples)} connectivity samples")
        
        # Process batch
        print("\nProcessing batch...")
        batch_results = adapter.run_batch(samples, max_samples=3)
        
        print(f"\n✅ Batch Results:")
        print(f"   Total: {batch_results['total']}")
        print(f"   Correct: {batch_results['correct']}")
        print(f"   Accuracy: {batch_results['accuracy']:.1f}%")
        
        # Show individual results
        print("\n   Individual Results:")
        for r in batch_results['results']:
            status = "✅" if r['matches_expected'] else "❌"
            print(f"   {status} Sample {r['id']}: {r['algorithm_used']}")
    else:
        print(f"\n⚠️ NLGraph data not found at: {data_path}")
        print("Skipping batch example.")
    
    print("\n" + "="*80)
    print("Example Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
