"""
Example script demonstrating NLGraph dataset loading and usage.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchmarks.nlgraph import NLGraphBenchmark


def main():
    # Initialize NLGraph benchmark loader
    data_path = Path(__file__).parent.parent.parent / 'data' / 'NLGraph' / 'NLGraph'
    
    print(f"Loading NLGraph dataset from: {data_path}")
    benchmark = NLGraphBenchmark(data_path)
    
    # Load test split for specific tasks
    print("\n=== Loading shortest_path and connectivity tasks ===")
    samples = benchmark.load_dataset(split='test', tasks=['shortest_path', 'connectivity'])
    print(f"Loaded {len(samples)} samples")
    
    # Get statistics
    stats = benchmark.get_task_statistics()
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Tasks: {stats['tasks']}")
    print(f"Difficulties: {stats['difficulties']}")
    
    # Filter by task
    print("\n=== Shortest Path Samples ===")
    shortest_path_samples = benchmark.filter_by_task('shortest_path')
    print(f"Found {len(shortest_path_samples)} shortest path samples")
    
    if shortest_path_samples:
        sample = shortest_path_samples[0]
        print(f"\nSample ID: {sample['id']}")
        print(f"Task: {sample['task']}")
        print(f"Difficulty: {sample['difficulty']}")
        print(f"Question (first 200 chars): {sample['question'][:200]}...")
        print(f"Answer: {sample['answer']}")
    
    # Filter by difficulty
    print("\n=== Easy Samples ===")
    easy_samples = benchmark.filter_by_difficulty('easy')
    print(f"Found {len(easy_samples)} easy samples")
    
    # Load all tasks
    print("\n=== Loading All Tasks ===")
    all_samples = benchmark.load_dataset(split='test')
    print(f"Loaded {len(all_samples)} total samples across all tasks")
    
    # Show breakdown by task
    for task in benchmark.TASK_TYPES:
        task_samples = benchmark.filter_by_task(task)
        if task_samples:
            print(f"  {task}: {len(task_samples)} samples")


if __name__ == '__main__':
    main()
