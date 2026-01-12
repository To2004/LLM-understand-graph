"""
Simple Interactive Graph Query Tool

A minimal interactive interface for testing graph queries.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphReasoningPipeline


def main():
    print("\n" + "="*60)
    print("  Graph Reasoning Pipeline - Simple Mode")
    print("="*60)
    print("\nInitializing pipeline...")
    
    # Initialize pipeline
    pipeline = GraphReasoningPipeline()
    print("✅ Ready! (Type 'exit' to quit)\n")
    
    query_num = 0
    
    while True:
        # Get input
        query = input("\n📝 Your query: ").strip()
        
        # Exit condition
        if query.lower() in ['exit', 'quit', 'q', '']:
            print("\n👋 Goodbye!\n")
            break
        
        # Process query
        query_num += 1
        print(f"\n⏳ Processing...")
        
        try:
            result = pipeline.run(query)
            
            print("\n" + "-"*60)
            if result.success:
                print(f"✅ {result.natural_language_response}")
                if result.algorithm_used:
                    print(f"   (Algorithm: {result.algorithm_used})")
            else:
                print(f"❌ {result.natural_language_response}")
            print("-"*60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
