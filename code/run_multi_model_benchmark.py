"""
Run NLGraph Benchmark with Multiple Models

This script tests multiple LLM models on the NLGraph benchmark.
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.pipeline import GraphReasoningPipeline
from src.agents.nlgraph_adapter import NLGraphAdapter
from src.benchmarks.nlgraph import NLGraphBenchmark
from src.llms import OpenRouterClient

# Models to test (all free models on OpenRouter)
MODELS_TO_TEST = [
    # "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    # "google/gemini-2.0-flash-exp:free",
    # "qwen/qwen-2.5-72b-instruct:free", # Currently 404
]

def run_multi_model_benchmark(
    models: list,
    tasks: list = None,
    max_samples: int = 3,
    difficulty: str = 'easy',
    output_dir: str = 'logs/multi_model_results'
):
    """
    Run benchmark with multiple models and compare results.
    """
    print("\n" + "="*80)
    print("Multi-Model NLGraph Benchmark Evaluation")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load benchmark data once
    benchmark = NLGraphBenchmark(Path("../data/NLGraph/NLGraph"))
    
    if tasks is None:
        tasks = ['connectivity', 'shortest_path']  # Start with these two
    
    benchmark.load_dataset(split='test', tasks=tasks)
    
    all_model_results = {}
    detailed_metrics = defaultdict(lambda: defaultdict(dict))
    
    for model_idx, model_name in enumerate(models):
        print(f"\n{'='*80}")
        print(f"Testing Model [{model_idx+1}/{len(models)}]: {model_name}")
        print(f"{'='*80}\n")
        
        model_start_time = time.time()
        
        try:
            # Initialize pipeline with specific model
            llm_client = OpenRouterClient(model_name=model_name)
            pipeline = GraphReasoningPipeline(llm_client=llm_client, verbose=False)
            adapter = NLGraphAdapter(pipeline.orchestrator)
            
            model_results = {}
            task_errors = defaultdict(list)
            task_times = {}
            task_tokens = {}
            
            # Reset client stats if possible or just track delta
            initial_tokens = getattr(llm_client, 'total_tokens_used', 0)
            
            for task in tasks:
                task_start = time.time()
                print(f"\nTask: {task.upper()}")
                samples = benchmark.filter_by_task(task)
                
                # Filter by difficulty
                if difficulty:
                    samples = [s for s in samples if s.get('difficulty') == difficulty]
                
                samples = samples[:max_samples]
                
                if not samples:
                    print(f"  No samples found")
                    continue
                
                print(f"  Processing {len(samples)} samples...")
                
                # Run batch with error tracking
                try:
                    results = adapter.run_batch(samples, max_samples=max_samples)
                    
                    # Track detailed results
                    for r in results['results']:
                        if r.get('error'):
                            task_errors[task].append(r['error'])
                    
                    model_results[task] = {
                        'accuracy': results['accuracy'],
                        'correct': results['correct'],
                        'total': results['total'],
                        'errors': len(task_errors[task])
                    }
                    
                    current_tokens = getattr(llm_client, 'total_tokens_used', 0)
                    tokens_this_task = current_tokens - initial_tokens
                    # Update initial for next task
                    initial_tokens = current_tokens
                    
                    task_times[task] = time.time() - task_start
                    task_tokens[task] = tokens_this_task
                    
                    print(f"  Accuracy: {results['accuracy']:.1f}% ({results['correct']}/{results['total']})")
                    print(f"  Time: {task_times[task]:.1f}s")
                    print(f"  Tokens: {tokens_this_task} (avg {tokens_this_task/results['total']:.0f}/sample)")
                    
                except Exception as e:
                    print(f"  [ERROR] Task failed: {str(e)}")
                    task_errors[task].append(str(e))
                    model_results[task] = {
                        'accuracy': 0.0,
                        'correct': 0,
                        'total': len(samples),
                        'errors': 1,
                        'error_message': str(e)
                    }
                
                # Add delay between tasks to avoid rate limits
                time.sleep(3)
            
            # Calculate overall accuracy for this model
            total_correct = sum(r['correct'] for r in model_results.values())
            total_samples = sum(r['total'] for r in model_results.values())
            overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
            model_time = time.time() - model_start_time
            
            # Calculate detailed analytics
            total_tokens = sum(task_tokens.values())
            
            # Analyze by graph size
            graph_size_stats = {'small (<10)': {'correct': 0, 'total': 0}, 
                              'medium (10-20)': {'correct': 0, 'total': 0}, 
                              'large (>20)': {'correct': 0, 'total': 0}}
            
            for task, res in all_model_results.get(model_name, {}).get('tasks', {}).items():
                # We need to access the raw results again, which are in the 'tasks' dict in my new structure
                # But wait, 'model_results' (local var) stores the summary, not the raw list.
                # I need to store the raw list in all_model_results to analyze it later or do it here.
                pass 
            
            # Let's just store the full raw results in a separate structure or simplisticly
            all_model_results[model_name] = {
                'overall_accuracy': overall_accuracy,
                'total_correct': total_correct,
                'total_samples': total_samples,
                'total_time': model_time,
                'total_tokens': total_tokens,
                'avg_time_per_sample': model_time / total_samples if total_samples > 0 else 0,
                'avg_tokens_per_sample': total_tokens / total_samples if total_samples > 0 else 0,
                'tasks': model_results,
                'error_summary': {task: len(errs) for task, errs in task_errors.items()}
            }
            
            print(f"\n{model_name} Overall: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")
            
        except Exception as e:
            print(f"  [ERROR] Failed to test {model_name}: {str(e)}")
            all_model_results[model_name] = {
                'error': str(e)
            }
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<50} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-"*80)
    
    for model_name, results in all_model_results.items():
        if 'error' in results:
            print(f"{model_name:<50} {'ERROR':>10} {results['error'][:15]:>15}")
        else:
            acc = results['overall_accuracy']
            correct = results['total_correct']
            total = results['total_samples']
            acc = results['overall_accuracy']
            correct = results['total_correct']
            total = results['total_samples']
            avg_tokens = results.get('avg_tokens_per_sample', 0)
            print(f"{model_name:<50} {acc:>9.1f}% {f'{correct}/{total}':>15}  {avg_tokens:>8.0f} toks/s")
    
    # Save results
    results_file = os.path.join(output_dir, f"multi_model_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'difficulty': difficulty,
            'max_samples': max_samples,
            'tasks': tasks,
            'results': all_model_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print("="*80 + "\n")
    
    return all_model_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NLGraph benchmark with multiple models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS_TO_TEST,
        help="Models to test"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=['shortest_path', 'connectivity', 'cycle', 'flow', 
                 'matching', 'hamilton', 'topology', 'GNN'],
        default=['connectivity', 'shortest_path'],
        help="Task types to evaluate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum samples per task"
    )
    parser.add_argument(
        "--difficulty",
        choices=['easy', 'medium', 'hard'],
        default='easy',
        help="Difficulty level"
    )
    
    args = parser.parse_args()
    
    try:
        run_multi_model_benchmark(
            models=args.models,
            tasks=args.tasks,
            max_samples=args.max_samples,
            difficulty=args.difficulty
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
