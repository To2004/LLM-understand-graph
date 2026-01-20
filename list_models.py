
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    # Try to find it in the user's env file manually if needed, or just warn
    pass

url = "https://openrouter.ai/api/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        models = response.json()['data']
        # Filter for free models or specific ones we care about
        free_models = [m['id'] for m in models if 'free' in m['id']]
        print(f"Found {len(free_models)} free models:")
        for m in sorted(free_models):
            print(m)
        
        # Also check for our specific targets to see if they exist under slightly different names
        print("\nChecking specific targets:")
        for m in models:
            if "deepseek" in m['id'].lower() and "r1" in m['id'].lower():
                print(f"Found DeepSeek candidate: {m['id']}")
            if "qwen" in m['id'].lower() and "2.5" in m['id'].lower() and "72b" in m['id'].lower():
                print(f"Found Qwen candidate: {m['id']}")

    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Error: {e}")
