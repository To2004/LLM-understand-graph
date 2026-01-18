"""
Test NLGraph Integration

This script tests the NLGraph adapter with sample questions from the benchmark.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter
from src.benchmarks.nlgraph import NLGraphBenchmark


def test_single_question():
    """Test with a single NLGraph question"""
    print("\n" + "="*80)
    print("Testing NLGraph Adapter - Single Question")
    print("="*80)
    
    # Initialize pipeline and adapter
    print("\nInitializing pipeline...")
    pipeline = GraphReasoningPipeline(verbose=True)
    
    print("Creating NLGraph adapter...")
    adapter = NLGraphAdapter(pipeline.orchestrator)
    
    # Test question (shortest path)
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
    
    expected_answer = "The shortest path from node 4 to node 0 is 4,5,0 with a total weight of 5"
    
    print("\n" + "-"*80)
    print("Question:")
    print(question[:200] + "...")
    print("\nExpected Answer:")
    print(expected_answer)
    print("-"*80)
    
    # Process question
    print("\nProcessing through NLGraph adapter...")
    result = adapter.process_nlgraph_question(question, expected_answer)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Success: {result.success}")
    print(f"Algorithm Used: {result.algorithm_used}")
    print(f"Agent Response: {result.natural_language_response}")
    print(f"Matches Expected: {result.matches_expected}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")
    
    print("="*80 + "\n")


def test_with_benchmark_data():
    """Test with actual NLGraph benchmark data"""
    print("\n" + "="*80)
    print("Testing NLGraph Adapter - Benchmark Data")
    print("="*80)
    
    # Check if data exists
    data_path = Path(__file__).parent.parent / 'data' / 'NLGraph' / 'NLGraph'
    
    if not data_path.exists():
        print(f"\n❌ NLGraph data not found at: {data_path}")
        print("Skipping benchmark test.")
        return
    
    # Initialize components
    print("\nInitializing pipeline...")
    pipeline = GraphReasoningPipeline(verbose=False)
    
    print("Creating NLGraph adapter...")
    adapter = NLGraphAdapter(pipeline.orchestrator)
    
    print("Loading NLGraph benchmark...")
    benchmark = NLGraphBenchmark(data_path)
    
    # Load a few samples from each task type
    print("\nLoading test samples...")
    benchmark.load_dataset(split='test', tasks=['shortest_path', 'connectivity'])
    
    # Test shortest path
    print("\n" + "-"*80)
    print("Testing SHORTEST_PATH task")
    print("-"*80)
    sp_samples = benchmark.filter_by_task('shortest_path')[:2]
    
    for i, sample in enumerate(sp_samples):
        print(f"\n[Sample {i+1}]")
        result = adapter.process_nlgraph_question(
            sample['question'],
            sample['answer']
        )
        
        print(f"Success: {result.success}")
        print(f"Matches: {result.matches_expected}")
        print(f"Response: {result.natural_language_response[:100]}...")
    
    # Test connectivity
    print("\n" + "-"*80)
    print("Testing CONNECTIVITY task")
    print("-"*80)
    conn_samples = benchmark.filter_by_task('connectivity')[:2]
    
    for i, sample in enumerate(conn_samples):
        print(f"\n[Sample {i+1}]")
        result = adapter.process_nlgraph_question(
            sample['question'],
            sample['answer']
        )
        
        print(f"Success: {result.success}")
        print(f"Matches: {result.matches_expected}")
        print(f"Response: {result.natural_language_response[:100]}...")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NLGraph integration")
    parser.add_argument(
        "--mode",
        choices=["single", "benchmark", "both"],
        default="single",
        help="Test mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["single", "both"]:
            test_single_question()
        
        if args.mode in ["benchmark", "both"]:
            test_with_benchmark_data()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
