"""
Simple script to verify OpenRouter API key is working.

Usage:
    python examples/check_api_key.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from models import OpenRouterClient, get_recommended_model


def check_api_key():
    """Quick test to verify API key works."""
    # Load environment
    load_dotenv()
    
    # Check if key exists
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in .env file")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenRouter API key to .env")
        print("3. Get a free key at: https://openrouter.ai/keys")
        return False
    
    if api_key == "your_api_key_here" or not api_key.strip():
        print("❌ ERROR: Please add your actual API key to .env file")
        print("   Current value is just a placeholder")
        print("\nGet a free key at: https://openrouter.ai/keys")
        return False
    
    print(f"✓ API key found (length: {len(api_key)})")
    
    # Try a simple API call
    print("\nTesting API connection...")
    try:
        model_id = get_recommended_model("general")
        print(f"Using model: {model_id}")
        
        client = OpenRouterClient(model_name=model_id)
        response = client.generate(
            prompt="Say 'Hello' in one word.",
            system_message="You are a helpful assistant."
        )
        
        print("\n✓ API connection successful!")
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Model: {response.model}")
        
        print("\n✓ Everything is working correctly!")
        print("You can now run: python examples/test_openrouter.py")
        return True
        
    except Exception as e:
        print(f"\n❌ API call failed: {e}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- No internet connection")
        print("- OpenRouter service is down")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("OpenRouter API Key Check")
    print("=" * 60)
    print()
    
    success = check_api_key()
    
    print()
    print("=" * 60)
    
    sys.exit(0 if success else 1)
