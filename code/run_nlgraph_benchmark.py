"""
Run NLGraph Benchmark with Agent Pipeline

This script evaluates your model on the NLGraph benchmark dataset.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter
from src.benchmarks.nlgraph import NLGraphBenchmark


def run_nlgraph_benchmark(
    data_path: str,
    tasks: list = None,
    max_samples_per_task: int = 10,
    split: str = 'test',
    difficulty: str = None,
    output_dir: str = 'logs/nlgraph_results',
    verbose: bool = False
):
    """
    Run NLGraph benchmark evaluation.
    
    Args:
        data_path: Path to NLGraph dataset
        tasks: List of task types to evaluate (default: all)
        max_samples_per_task: Maximum samples per task type
        split: Dataset split ('test', 'train', 'main')
        difficulty: Filter by difficulty ('easy', 'medium', 'hard', or None for all)
        output_dir: Directory to save results
        verbose: Print detailed output
    """
    print("\n" + "="*80)
    print("NLGraph Benchmark Evaluation")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize components
    print("\n1. Initializing pipeline...")
    pipeline = GraphReasoningPipeline(verbose=verbose)
    
    print("2. Creating NLGraph adapter...")
    adapter = NLGraphAdapter(pipeline.orchestrator)
    
    print(f"3. Loading NLGraph benchmark from: {data_path}")
    benchmark = NLGraphBenchmark(Path(data_path))
    
    # Load dataset
    if tasks is None:
        tasks = benchmark.TASK_TYPES
    
    print(f"4. Loading tasks: {', '.join(tasks)}")
    benchmark.load_dataset(split=split, tasks=tasks)
    
    # Get statistics
    stats = benchmark.get_task_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    for task, count in stats['tasks'].items():
        print(f"  - {task}: {count} samples")
    
    # Run evaluation for each task
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'split': split,
        'max_samples_per_task': max_samples_per_task,
        'tasks': {}
    }
    
    total_correct = 0
    total_processed = 0
    
    for task in tasks:
        print("\n" + "="*80)
        print(f"Evaluating Task: {task.upper()}")
        print("="*80)
        
        # Get samples for this task
        samples = benchmark.filter_by_task(task)
        
        # Filter by difficulty if specified
        if difficulty:
            samples = [s for s in samples if s.get('difficulty') == difficulty]
            print(f"Filtering by difficulty: {difficulty}")
        
        samples = samples[:max_samples_per_task]
        
        if not samples:
            print(f"No samples found for task: {task}")
            continue
        
        print(f"Processing {len(samples)} samples...")
        
        # Process batch
        task_results = []
        task_correct = 0
        
        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] Sample {sample.get('id', i)}...")
            
            # Add delay to avoid rate limits
            if i > 0:
                time.sleep(2)  # 2 second delay between requests
            
            try:
                result = adapter.process_nlgraph_question(
                    sample['question'],
                    sample['answer']
                )
                
                task_results.append({
                    'id': sample.get('id', i),
                    'difficulty': sample.get('difficulty', 'unknown'),
                    'success': result.success,
                    'matches_expected': result.matches_expected,
                    'agent_response': result.natural_language_response,
                    'expected_answer': sample['answer'],
                    'algorithm_used': result.algorithm_used,
                    'error': result.error_message
                })
                
                if result.matches_expected:
                    task_correct += 1
                    total_correct += 1
                    print(f"  [OK] Correct")
                elif result.success:
                    print(f"  [WARN] Completed but answer may not match")
                    print(f"     Agent: {result.natural_language_response[:80]}...")
                    print(f"     Expected: {sample['answer'][:80]}...")
                else:
                    print(f"  [FAIL] Failed: {result.error_message}")
                
                total_processed += 1
                
            except Exception as e:
                print(f"  [ERROR] Exception: {str(e)}")
                task_results.append({
                    'id': sample.get('id', i),
                    'difficulty': sample.get('difficulty', 'unknown'),
                    'success': False,
                    'matches_expected': False,
                    'error': str(e)
                })
                total_processed += 1
        
        # Calculate task accuracy
        task_accuracy = (task_correct / len(samples) * 100) if samples else 0
        
        all_results['tasks'][task] = {
            'total': len(samples),
            'correct': task_correct,
            'accuracy': task_accuracy,
            'results': task_results
        }
        
        print(f"\n{task.upper()} Results:")
        print(f"  Accuracy: {task_accuracy:.1f}% ({task_correct}/{len(samples)})")
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_processed * 100) if total_processed > 0 else 0
    
    all_results['summary'] = {
        'total_processed': total_processed,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy
    }
    
    # Print summary
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"\nTotal Samples Processed: {total_processed}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    
    print("\nPer-Task Accuracy:")
    for task, task_data in all_results['tasks'].items():
        print(f"  {task:20} {task_data['accuracy']:5.1f}% ({task_data['correct']}/{task_data['total']})")
    
    # Save results
    json_file = os.path.join(output_dir, f"nlgraph_results_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_file}")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NLGraph benchmark evaluation")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/NLGraph/NLGraph",
        help="Path to NLGraph dataset"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=['shortest_path', 'connectivity', 'cycle', 'flow', 
                 'matching', 'hamilton', 'topology', 'GNN'],
        help="Task types to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples per task (default: 10)"
    )
    parser.add_argument(
        "--split",
        choices=['test', 'train', 'main'],
        default='test',
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--difficulty",
        choices=['easy', 'medium', 'hard'],
        help="Filter by difficulty level (default: all difficulties)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/nlgraph_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        run_nlgraph_benchmark(
            data_path=args.data_path,
            tasks=args.tasks,
            max_samples_per_task=args.max_samples,
            split=args.split,
            difficulty=args.difficulty,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
