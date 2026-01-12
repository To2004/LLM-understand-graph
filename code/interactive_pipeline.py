"""
Interactive Graph Reasoning Pipeline - REPL Interface

This script provides an interactive command-line interface for testing
the graph reasoning pipeline. Users can input queries and see results
in real-time.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphReasoningPipeline
from src.models import OpenRouterClient


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  🧠 LLM GRAPH REASONING PIPELINE - Interactive Mode")
    print("=" * 70)
    print("\nUsing: Llama 3.3 70B via OpenRouter (free)")
    print("\nCommands:")
    print("  - Type your graph query and press Enter")
    print("  - Type 'help' for example queries")
    print("  - Type 'stats' to see usage statistics")
    print("  - Type 'verbose' to toggle verbose mode")
    print("  - Type 'exit' or 'quit' to exit")
    print("\n" + "=" * 70 + "\n")


def print_help():
    """Print example queries."""
    print("\n" + "=" * 70)
    print("  📚 EXAMPLE QUERIES")
    print("=" * 70)
    print("\n1. Connectivity:")
    print("   Graph: A--B--C. Is A connected to C?")
    print("\n2. Shortest Path:")
    print("   Graph: A--B--C--D. Find shortest path from A to D.")
    print("\n3. Cycle Detection:")
    print("   Graph: A->B, B->C, C->A. Does this graph have a cycle?")
    print("\n4. Path Finding:")
    print("   Given nodes 1,2,3,4 with edges 1-2, 2-3, 3-4. Find path from 1 to 4.")
    print("\n5. Complex Query:")
    print("   In a directed graph with edges A->B, B->C, C->D, D->B, is there a cycle?")
    print("\n" + "=" * 70 + "\n")


def print_result(result, query_num):
    """Print formatted result."""
    print("\n" + "-" * 70)
    print(f"📊 RESULT #{query_num}")
    print("-" * 70)
    
    if result.success:
        print(f"✅ Success: {result.success}")
        print(f"\n💬 Response:")
        print(f"   {result.natural_language_response}")
        
        if result.algorithm_used:
            print(f"\n🔧 Algorithm: {result.algorithm_used}")
        
        if result.metadata:
            print(f"\n📈 Metadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")
    else:
        print(f"❌ Failed: {result.success}")
        print(f"\n⚠️  Error:")
        print(f"   {result.error_message}")
        print(f"\n💬 Response:")
        print(f"   {result.natural_language_response}")
    
    print("-" * 70 + "\n")


def print_stats(pipeline, query_count, successful_queries):
    """Print usage statistics."""
    print("\n" + "=" * 70)
    print("  📊 SESSION STATISTICS")
    print("=" * 70)
    print(f"\nTotal Queries: {query_count}")
    print(f"Successful: {successful_queries}")
    print(f"Failed: {query_count - successful_queries}")
    
    if query_count > 0:
        success_rate = (successful_queries / query_count) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    # Get LLM usage stats if available
    if hasattr(pipeline.llm_client, 'get_usage_stats'):
        usage = pipeline.llm_client.get_usage_stats()
        print(f"\nLLM Usage:")
        print(f"  Total Requests: {usage.get('total_requests', 0)}")
        print(f"  Total Tokens: {usage.get('total_tokens', 0):,}")
    
    print("=" * 70 + "\n")


def main():
    """Main interactive loop."""
    print_banner()
    
    # Initialize pipeline
    print("🔄 Initializing pipeline...")
    try:
        pipeline = GraphReasoningPipeline(verbose=False)
        print("✅ Pipeline ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("\nMake sure you have:")
        print("1. Set OPENROUTER_API_KEY in your .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        return
    
    # Session state
    query_count = 0
    successful_queries = 0
    verbose = False
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("🔍 Enter your query (or 'help'): ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for using the Graph Reasoning Pipeline!")
                print_stats(pipeline, query_count, successful_queries)
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'stats':
                print_stats(pipeline, query_count, successful_queries)
                continue
            
            elif user_input.lower() == 'verbose':
                verbose = not verbose
                pipeline.verbose = verbose
                print(f"\n🔧 Verbose mode: {'ON' if verbose else 'OFF'}\n")
                continue
            
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            
            # Process query
            query_count += 1
            print(f"\n⏳ Processing query #{query_count}...")
            
            try:
                result = pipeline.run(user_input)
                
                if result.success:
                    successful_queries += 1
                
                print_result(result, query_count)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Query interrupted by user.\n")
                query_count -= 1  # Don't count interrupted queries
                
            except Exception as e:
                print(f"\n❌ Error processing query: {e}\n")
                import traceback
                if verbose:
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            print_stats(pipeline, query_count, successful_queries)
            break
        
        except EOFError:
            print("\n\n👋 Exiting...")
            print_stats(pipeline, query_count, successful_queries)
            break


if __name__ == "__main__":
    main()
