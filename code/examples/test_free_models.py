"""
Test script for OpenRouter free models.
Queries multiple free models with a basic prompt to compare their responses.

Before running:
1. Copy .env.example to .env
2. Add your OpenRouter API key to .env
3. Get a free key at: https://openrouter.ai/keys

Usage:
    python examples/test_free_models.py
"""

import os
import sys
from pathlib import Path
from time import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from models.openrouter_client import OpenRouterClient
from models.model_configs import get_model_id, FREE_MODELS


# Test prompt - simple but tests reasoning and clarity
TEST_PROMPT = """Explain what a graph traversal algorithm is and give one example.
Be concise (2-3 sentences)."""


def test_model(model_key: str, model_info: dict) -> dict:
    """
    Test a single model with the test prompt.

    Args:
        model_key: Short model key
        model_info: Model configuration dict

    Returns:
        Dict with test results
    """
    model_id = model_info["id"]

    print(f"\n{'='*80}")
    print(f"Testing: {model_info['name']}")
    print(f"Key: {model_key}")
    print(f"ID: {model_id}")
    print(f"Context: {model_info['context_window']:,} tokens")
    print(f"{'='*80}\n")

    try:
        # Create client
        client = OpenRouterClient(model_name=model_id)

        # Time the request
        start_time = time()
        response = client.generate(
            prompt=TEST_PROMPT,
            system_message="You are a helpful computer science tutor."
        )
        elapsed_time = time() - start_time

        # Print results
        print(f"Response ({elapsed_time:.2f}s):")
        print(f"{response.content}\n")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Finish reason: {response.finish_reason}")

        return {
            "model_key": model_key,
            "model_name": model_info['name'],
            "success": True,
            "response": response.content,
            "tokens": response.tokens_used,
            "time": elapsed_time,
            "error": None
        }

    except Exception as e:
        print(f"ERROR: {str(e)}\n")
        return {
            "model_key": model_key,
            "model_name": model_info['name'],
            "success": False,
            "response": None,
            "tokens": 0,
            "time": 0,
            "error": str(e)
        }


def test_selected_models():
    """Test a curated selection of the best free models."""
    print("\n" + "="*80)
    print("TESTING SELECTED FREE MODELS")
    print("="*80)
    print(f"\nTest Prompt: {TEST_PROMPT}\n")

    # Select the most powerful and interesting free models (verified available)
    selected_models = [
        "llama-3.3-70b",      # Best overall
        "deepseek-r1",        # Best reasoning
        "devstral-2",         # Best for code with huge context
        "mimo-v2-flash",      # Fastest with large context
        "gemma-3-27b",        # Balanced general purpose
    ]

    results = []

    for model_key in selected_models:
        if model_key not in FREE_MODELS:
            print(f"\nWarning: Model {model_key} not found in configuration")
            continue

        model_info = FREE_MODELS[model_key]
        result = test_model(model_key, model_info)
        results.append(result)

    return results


def test_all_models():
    """Test all available free models."""
    print("\n" + "="*80)
    print("TESTING ALL FREE MODELS")
    print("="*80)
    print(f"\nTest Prompt: {TEST_PROMPT}\n")

    results = []

    for model_key, model_info in FREE_MODELS.items():
        result = test_model(model_key, model_info)
        results.append(result)

    return results


def print_summary(results: list):
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal models tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n" + "-"*80)
        print("SUCCESSFUL MODELS")
        print("-"*80)

        # Sort by response time
        successful.sort(key=lambda x: x["time"])

        for r in successful:
            print(f"\n{r['model_name']} ({r['model_key']})")
            print(f"  Time: {r['time']:.2f}s")
            print(f"  Tokens: {r['tokens']}")
            print(f"  Response length: {len(r['response'])} chars")

    if failed:
        print("\n" + "-"*80)
        print("FAILED MODELS")
        print("-"*80)

        for r in failed:
            print(f"\n{r['model_name']} ({r['model_key']})")
            print(f"  Error: {r['error']}")

    if successful:
        fastest = min(successful, key=lambda x: x["time"])
        most_tokens = max(successful, key=lambda x: x["tokens"])

        print("\n" + "-"*80)
        print("HIGHLIGHTS")
        print("-"*80)
        print(f"\nFastest: {fastest['model_name']} ({fastest['time']:.2f}s)")
        print(f"Most detailed: {most_tokens['model_name']} ({most_tokens['tokens']} tokens)")


def main():
    """Run the test suite."""
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

    print("\n" + "="*80)
    print("OPENROUTER FREE MODELS TEST")
    print("="*80)

    # Ask user which test to run
    print("\nSelect test mode:")
    print("1. Test selected powerful models (recommended - faster)")
    print("2. Test all available models (slower)")

    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"

    if choice == "2":
        results = test_all_models()
    else:
        results = test_selected_models()

    # Print summary
    print_summary(results)

    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
