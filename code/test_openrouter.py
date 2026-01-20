"""
Test OpenRouter API Connection and Model Availability

This script helps diagnose OpenRouter API issues by:
1. Checking if the API key is set
2. Testing the connection with a simple request
3. Trying alternative models if the default fails
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llms.openrouter_client import OpenRouterClient

# Load environment variables
load_dotenv()

def test_api_connection():
    """Test OpenRouter API connection and model availability."""
    
    print("=" * 80)
    print("OpenRouter API Connection Test")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("\nTo fix this:")
        print("1. Get an API key from https://openrouter.ai/keys")
        print("2. Set it as an environment variable:")
        print("   Windows PowerShell: $env:OPENROUTER_API_KEY='your-key-here'")
        print("   Windows CMD: set OPENROUTER_API_KEY=your-key-here")
        print("   Or create a .env file in the code directory with:")
        print("   OPENROUTER_API_KEY=your-key-here")
        return False
    
    print(f"\n[OK] API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test models in order of preference
    test_models = [
        ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (Free)"),
        ("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (Free)"),
        ("google/gemini-flash-1.5", "Gemini Flash 1.5"),
        ("qwen/qwen-2-7b-instruct:free", "Qwen 2 7B (Free)"),
    ]
    
    print("\n" + "=" * 80)
    print("Testing Models")
    print("=" * 80)
    
    working_models = []
    
    for model_name, display_name in test_models:
        print(f"\n📝 Testing: {display_name}")
        print(f"   Model ID: {model_name}")
        
        try:
            client = OpenRouterClient(model_name=model_name)
            
            # Simple test prompt
            response = client.generate(
                prompt="What is 2+2? Answer with just the number.",
                system_message="You are a helpful assistant."
            )
            
            print(f"   ✅ SUCCESS!")
            print(f"   Response: {response.content[:100]}")
            print(f"   Tokens used: {response.tokens_used}")
            
            working_models.append((model_name, display_name))
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)[:200]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if working_models:
        print(f"\n✅ Found {len(working_models)} working model(s):")
        for model_name, display_name in working_models:
            print(f"   - {display_name}")
            print(f"     Model ID: {model_name}")
        
        print("\n💡 Recommended action:")
        print(f"   Use this model in your pipeline:")
        print(f"   pipeline = GraphReasoningPipeline(")
        print(f"       llm_client=OpenRouterClient(model_name=\"{working_models[0][0]}\")")
        print(f"   )")
        
        return True
    else:
        print("\n❌ No working models found!")
        print("\nPossible issues:")
        print("1. OpenRouter service might be experiencing issues")
        print("2. Your API key might be invalid or expired")
        print("3. You might have exceeded your rate limits")
        print("\nTroubleshooting:")
        print("- Check https://openrouter.ai/activity for your API usage")
        print("- Try again in a few minutes")
        print("- Consider using a paid model if free models are unavailable")
        
        return False


if __name__ == "__main__":
    success = test_api_connection()
    sys.exit(0 if success else 1)
