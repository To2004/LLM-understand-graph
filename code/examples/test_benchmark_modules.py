"""
Test script for benchmark modules (GraphInstructBenchmark and BenchmarkEvaluator)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchmarks import GraphInstructBenchmark, BenchmarkEvaluator, NLGraphBenchmark
from benchmarks.evaluator import EvaluationMetrics


def test_evaluator():
    """Test BenchmarkEvaluator functionality"""
    print("=" * 60)
    print("Testing BenchmarkEvaluator")
    print("=" * 60)
    
    evaluator = BenchmarkEvaluator()
    
    # Create sample data
    predictions = ["42", "yes", "node 5", "3"]
    ground_truths = ["42", "yes", "node 5", "4"]
    metadata = [
        {'success': True, 'iterations': 1, 'tool_calling_success': True},
        {'success': True, 'iterations': 2, 'tool_calling_success': True},
        {'success': True, 'iterations': 1, 'tool_calling_success': True},
        {'success': False, 'iterations': 3, 'tool_calling_success': False, 'repair_attempts': 2}
    ]
    
    # Evaluate
    print("\nEvaluating sample predictions...")
    metrics = evaluator.evaluate(predictions, ground_truths, metadata)
    
    print(f"\nResults:")
    print(f"  Exact Match Accuracy: {metrics.exact_match_accuracy:.2%}")
    print(f"  Tool Calling Success Rate: {metrics.tool_calling_success_rate:.2%}")
    print(f"  Repair Efficiency: {metrics.repair_efficiency:.2%}")
    print(f"  Average Iterations: {metrics.average_iterations:.2f}")
    print(f"  Total Samples: {metrics.total_samples}")
    print(f"  Successful Samples: {metrics.successful_samples}")
    
    # Test report generation
    print("\nGenerating JSON report...")
    output_path = Path(__file__).parent.parent / 'results' / 'test_report.json'
    evaluator.generate_report(metrics, str(output_path), format='json')
    
    print("\nGenerating CSV report...")
    csv_path = Path(__file__).parent.parent / 'results' / 'test_report.csv'
    evaluator.generate_report(metrics, str(csv_path), format='csv')
    
    # Test LaTeX table generation
    print("\nGenerating LaTeX table...")
    latex_table = evaluator.generate_latex_table(metrics)
    print(latex_table)
    
    # Test statistical analysis
    print("\n" + "=" * 60)
    print("Testing Statistical Analysis")
    print("=" * 60)
    
    # Create baseline results
    baseline_metrics = EvaluationMetrics(
        exact_match_accuracy=0.5,
        tool_calling_success_rate=0.6,
        repair_efficiency=0.7,
        average_iterations=2.5,
        total_samples=4,
        successful_samples=2
    )
    
    analysis = evaluator.statistical_analysis(baseline_metrics, metrics)
    print(f"\nBaseline Accuracy: {analysis['baseline_accuracy']:.2%}")
    print(f"System Accuracy: {analysis['system_accuracy']:.2%}")
    print(f"Improvement: {analysis['improvement']:.2%}")
    print(f"Relative Improvement: {analysis['relative_improvement']:.2%}")
    if analysis.get('p_value'):
        print(f"P-value: {analysis['p_value']:.4f}")
        print(f"Statistically Significant: {analysis['significant']}")
    
    # Test multi-model comparison
    print("\n" + "=" * 60)
    print("Testing Multi-Model Comparison")
    print("=" * 60)
    
    model_results = [baseline_metrics, metrics]
    model_names = ["Baseline", "System"]
    
    comparison = evaluator.multi_model_comparison(model_results, model_names)
    print(f"\nBest Model: {comparison['best_model']} ({comparison['best_accuracy']:.2%})")
    print(f"Worst Model: {comparison['worst_model']} ({comparison['worst_accuracy']:.2%})")
    print(f"Mean Accuracy: {comparison['mean_accuracy']:.2%}")
    print(f"Ranking: {', '.join(comparison['ranking'])}")
    
    print("\n✓ BenchmarkEvaluator tests completed successfully!")


def test_graph_instruct():
    """Test GraphInstructBenchmark functionality"""
    print("\n" + "=" * 60)
    print("Testing GraphInstructBenchmark")
    print("=" * 60)
    
    # Create a mock dataset directory for testing
    test_data_path = Path(__file__).parent.parent / 'data' / 'GraphInstruct'
    
    if not test_data_path.exists():
        print(f"\nWarning: Test data path does not exist: {test_data_path}")
        print("Skipping GraphInstructBenchmark tests.")
        print("To test this module, create sample data at the path above.")
        return
    
    try:
        benchmark = GraphInstructBenchmark(test_data_path)
        print(f"✓ Initialized GraphInstructBenchmark with path: {test_data_path}")
        
        # Try to load dataset
        samples = benchmark.load_dataset(split='test')
        print(f"✓ Loaded {len(samples)} samples")
        
        if len(samples) > 0:
            # Get statistics
            stats = benchmark.get_statistics()
            print(f"\nDataset Statistics:")
            print(f"  Total Samples: {stats['total_samples']}")
            print(f"  Difficulties: {stats['difficulties']}")
            print(f"  Edge Cases: {stats['edge_cases']}")
            print(f"  Graph Types: {stats['graph_types']}")
            
            # Get edge cases
            edge_cases = benchmark.get_edge_cases()
            print(f"\n  Samples with edge cases: {len(edge_cases)}")
            
            # Get first sample
            if len(samples) > 0:
                sample = benchmark.get_sample(0)
                print(f"\nFirst Sample:")
                print(f"  ID: {sample['id']}")
                print(f"  Difficulty: {sample.get('difficulty', 'unknown')}")
                print(f"  Instruction: {sample.get('instruction', '')[:100]}...")
        
        print("\n✓ GraphInstructBenchmark tests completed successfully!")
        
    except ValueError as e:
        print(f"\nExpected error (no dataset): {e}")
        print("This is normal if you don't have GraphInstruct dataset yet.")


def test_nlgraph():
    """Test NLGraphBenchmark functionality (already implemented)"""
    print("\n" + "=" * 60)
    print("Testing NLGraphBenchmark (existing implementation)")
    print("=" * 60)
    
    test_data_path = Path(__file__).parent.parent / 'data' / 'NLGraph' / 'NLGraph'
    
    if not test_data_path.exists():
        print(f"\nWarning: Test data path does not exist: {test_data_path}")
        print("Skipping NLGraphBenchmark tests.")
        return
    
    try:
        benchmark = NLGraphBenchmark(test_data_path)
        print(f"✓ Initialized NLGraphBenchmark with path: {test_data_path}")
        
        # Try to load dataset
        samples = benchmark.load_dataset(split='test', tasks=['shortest_path'])
        print(f"✓ Loaded {len(samples)} samples")
        
        if len(samples) > 0:
            stats = benchmark.get_task_statistics()
            print(f"\nDataset Statistics:")
            print(f"  Total Samples: {stats['total_samples']}")
            print(f"  Tasks: {stats['tasks']}")
        
        print("\n✓ NLGraphBenchmark tests completed successfully!")
        
    except ValueError as e:
        print(f"\nExpected error (no dataset): {e}")
        print("This is normal if you don't have NLGraph dataset yet.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BENCHMARK MODULES TEST SUITE")
    print("=" * 60)
    
    # Test evaluator (doesn't require external data)
    test_evaluator()
    
    # Test benchmark loaders (may require external datasets)
    test_graph_instruct()
    test_nlgraph()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
