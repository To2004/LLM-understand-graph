"""
Example script demonstrating OpenRouter client usage with free models.

Before running:
1. Copy .env.example to .env
2. Add your OpenRouter API key to .env
3. Get a free key at: https://openrouter.ai/keys

Usage:
    python examples/test_openrouter.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from models.openrouter_client import OpenRouterClient
from models.model_configs import (
    get_recommended_model,
    get_model_id,
    list_available_models,
    print_model_info
)


def test_basic_generation():
    """Test basic text generation."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Text Generation")
    print("=" * 80)
    
    # Use the recommended model for general tasks
    model_id = get_recommended_model("general")
    print(f"Using model: {model_id}\n")
    
    client = OpenRouterClient(model_name=model_id)
    
    response = client.generate(
        prompt="What is a graph in computer science? Answer in 2 sentences.",
        system_message="You are a helpful computer science tutor."
    )
    
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.tokens_used}")
    print(f"Model: {response.model}")
    print()


def test_structured_output():
    """Test structured JSON generation."""
    print("\n" + "=" * 80)
    print("TEST 2: Structured JSON Output")
    print("=" * 80)
    
    # Use the best model for structured output
    model_id = get_recommended_model("parser")
    print(f"Using model: {model_id}\n")
    
    client = OpenRouterClient(model_name=model_id)
    
    schema = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"}
                    }
                }
            }
        }
    }
    
    prompt = """
    Extract the graph structure from this description:
    "There are three cities: A, B, and C. There is a road from A to B and from B to C."
    """
    
    result = client.generate_structured(
        prompt=prompt,
        schema=schema,
        system_message="You are a graph extraction expert."
    )
    
    print(f"Structured output: {result}")
    print()


def test_different_models():
    """Test different free models."""
    print("\n" + "=" * 80)
    print("TEST 3: Comparing Different Models")
    print("=" * 80)

    prompt = "What is the shortest path problem? Answer in one sentence."

    # Test the best free models (verified available)
    test_models = ["llama-3.3-70b", "deepseek-r1", "devstral-2", "gemma-3-27b"]

    for model_key in test_models:
        model_id = get_model_id(model_key)
        print(f"\nModel: {model_key}")
        print(f"ID: {model_id}")

        try:
            client = OpenRouterClient(model_name=model_id)
            response = client.generate(prompt)
            print(f"Response: {response.content}")
            print(f"Tokens: {response.tokens_used}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 80)


def test_batch_generation():
    """Test batch generation."""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Generation")
    print("=" * 80)
    
    model_id = get_recommended_model("general")
    print(f"Using model: {model_id}\n")
    
    client = OpenRouterClient(model_name=model_id)
    
    prompts = [
        "What is DFS? Answer in 5 words.",
        "What is BFS? Answer in 5 words.",
        "What is Dijkstra? Answer in 5 words."
    ]
    
    responses = client.batch_generate(prompts)
    
    for i, response in enumerate(responses):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Response: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print()


def main():
    """Run all tests."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenRouter API key to .env")
        print("3. Get a free key at: https://openrouter.ai/keys")
        return
    
    print("\n" + "=" * 80)
    print("OPENROUTER CLIENT TEST SUITE")
    print("=" * 80)
    
    # Show available models
    print_model_info()
    
    # Run tests
    try:
        test_basic_generation()
        test_structured_output()
        test_different_models()
        test_batch_generation()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure:")
        print("1. Your API key is valid")
        print("2. You have internet connection")
        print("3. OpenRouter service is available")


if __name__ == "__main__":
    main()
