
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
url = f"{base_url}/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/GraphReasoning/LLM-Graph",
    "X-Title": "LLM Graph Reasoning Framework",
}

models = [
    "deepseek/deepseek-r1-0528:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

print(f"Testing URL: {url}")
for model in models:
    print(f"\nTesting Model: {model}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
        else:
            print("Success!")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(1)
