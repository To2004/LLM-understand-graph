"""
Example usage of the LLM Graph Reasoning Framework
"""


def example_basic_usage():
    """
    Basic example of using the framework.
    """
    print("Basic Usage Example")
    print("=" * 60)
    
    try:
        from src.pipeline import GraphReasoningPipeline
        from src.llms import OpenRouterClient
        
        # Initialize LLM client with Llama 3.3 70B from OpenRouter
        print("\nInitializing OpenRouter client with Llama 3.3 70B...")
        llm_client = OpenRouterClient(model_name="meta-llama/llama-3.3-70b-instruct:free")
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = GraphReasoningPipeline(llm_client, verbose=True)
        
        # Run a query
        query = "Graph: Nodes A, B, C, D with edges A--B, B--C, C--D. Task: Is A connected to D?"
        
        print(f"\nQuery: {query}\n")
        result = pipeline.run(query)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Response: {result.natural_language_response}")
        print(f"Success: {result.success}")
        if result.algorithm_used:
            print(f"Algorithm Used: {result.algorithm_used}")
        if result.metadata:
            print(f"Metadata: {result.metadata}")
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("Make sure you're running from the code directory:")
        print("  cd code")
        print("  python examples/basic_usage.py")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()



def example_benchmark_evaluation():
    """
    Example of running benchmark evaluation.
    
    TODO [EXAMPLES-002]: Update when modules are implemented
    """
    print("\nBenchmark Evaluation Example")
    print("=" * 60)
    
    # TODO: Uncomment when implemented
    """
    from src.benchmarks import NLGraphBenchmark, BenchmarkEvaluator
    from src.agents import AgentOrchestrator
    
    # Load benchmark
    benchmark = NLGraphBenchmark("data/nlgraph")
    samples = benchmark.filter_by_task("connectivity")[:10]  # First 10 samples
    
    # Initialize system (assuming orchestrator is set up)
    # orchestrator = ...
    
    # Run evaluation
    predictions = []
    ground_truths = []
    
    for sample in samples:
        result = orchestrator.execute(sample['input'])
        predictions.append(result.solution)
        ground_truths.append(sample['ground_truth'])
    
    # Evaluate
    evaluator = BenchmarkEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truths, samples)
    
    print(f"Accuracy: {metrics.exact_match_accuracy:.2%}")
    print(f"Tool Success Rate: {metrics.tool_calling_success_rate:.2%}")
    """
    
    print("\n[TODO] Benchmark evaluation not yet implemented!")


def example_custom_algorithm():
    """
    Example of registering custom algorithm.
    
    TODO [EXAMPLES-003]: Update when modules are implemented
    """
    print("\nCustom Algorithm Example")
    print("=" * 60)
    
    # TODO: Uncomment when implemented
    """
    from src.algorithms import AlgorithmExecutor
    import networkx as nx
    
    # Define custom algorithm
    def my_custom_algorithm(graph, source, target):
        # Custom logic here
        return nx.shortest_path(graph, source, target)
    
    # Register algorithm
    executor = AlgorithmExecutor()
    executor.register_algorithm(
        name="my_algorithm",
        implementation=my_custom_algorithm,
        metadata={
            "complexity": "O(V + E)",
            "preconditions": ["graph_connected"]
        }
    )
    
    # Use in pipeline
    # ...
    """
    
    print("\n[TODO] Custom algorithms not yet supported!")


if __name__ == "__main__":
    example_basic_usage()
    example_benchmark_evaluation()
    example_custom_algorithm()
