"""
Advanced example: Using different models for different agents
"""


def example_multi_model_pipeline():
    """
    Example showing how to use different specialized models for each agent.
    
    This demonstrates the agent_models parameter which allows you to
    assign different models to different agents based on their strengths.
    """
    print("Multi-Model Pipeline Example")
    print("=" * 60)
    
    try:
        from src.pipeline import GraphReasoningPipeline
        
        # Define different models for different agents
        # Based on model strengths from model_configs.py
        agent_models = {
            "parser": "meta-llama/llama-3.3-70b-instruct:free",      # Best reasoning for parsing
            "chooser": "deepseek/deepseek-r1-0528:free",             # Advanced reasoning for algorithm selection
            "synthesizer": "google/gemma-3-27b-it:free",             # Good for natural language generation
            "validator": "xiaomi/mimo-v2-flash:free",                # Fast validation
            "decomposer": "meta-llama/llama-3.3-70b-instruct:free"   # Good at understanding structure
        }
        
        print("\nInitializing pipeline with specialized models per agent:")
        for agent, model in agent_models.items():
            model_short = model.split('/')[-1]
            print(f"  {agent:12s}: {model_short}")
        
        # Initialize pipeline with custom models
        pipeline = GraphReasoningPipeline(
            llm_client=None,  # Will auto-create clients
            agent_models=agent_models,
            verbose=True
        )
        
        # Run a query
        query = "Graph: A--B--C--D--E. Find the shortest path from A to E."
        
        print(f"\nQuery: {query}\n")
        result = pipeline.run(query)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Response: {result.natural_language_response}")
        print(f"Success: {result.success}")
        if result.algorithm_used:
            print(f"Algorithm Used: {result.algorithm_used}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


def example_simple_openrouter():
    """
    Simplest example: Just use OpenRouter with default model (Llama 3.3 70B).
    """
    print("\nSimple OpenRouter Example")
    print("=" * 60)
    
    try:
        from src.pipeline import GraphReasoningPipeline
        
        # No need to specify anything - defaults to OpenRouter + Llama 3.3 70B
        print("\nInitializing pipeline (auto-defaults to Llama 3.3 70B)...")
        pipeline = GraphReasoningPipeline(verbose=True)
        
        # Run a query
        query = "Graph: A--B, B--C, C--A. Does this graph have a cycle?"
        
        result = pipeline.run(query)
        
        print(f"\nResponse: {result.natural_language_response}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


def example_quick_query():
    """
    Example using the quick_query convenience function.
    """
    print("\nQuick Query Example")
    print("=" * 60)
    
    try:
        from src.pipeline import quick_query
        
        # One-liner query (uses OpenRouter + Llama 3.3 70B by default)
        print("\nRunning quick query...")
        response = quick_query("Graph: A--B--C. Is A connected to C?")
        
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all examples
    example_simple_openrouter()
    print("\n" + "=" * 80 + "\n")
    
    example_quick_query()
    print("\n" + "=" * 80 + "\n")
    
    example_multi_model_pipeline()
