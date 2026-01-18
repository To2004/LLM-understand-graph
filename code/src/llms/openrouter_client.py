"""
Open Router API client implementation
"""

import os
import json
import time
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .base import BaseLLMClient, LLMResponse

# Load environment variables from .env file
load_dotenv()


class OpenRouterClient(BaseLLMClient):
    """
    Open Router API client for accessing various open-source models.
    Uses direct HTTP requests to OpenRouter API.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/llama-3.3-70b-instruct:free",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        """
        Initialize Open Router client.
        
        Args:
            model_name: Open Router model identifier
            api_key: Open Router API key (defaults to OPENROUTER_API_KEY env var)
            base_url: Open Router API base URL
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        super().__init__(model_name, **kwargs)
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Open Router API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url.rstrip('/')
        
        # Set default parameters
        self.max_tokens = kwargs.get('max_tokens', 8192)
    
    def generate(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Open Router API.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with generated text
        """
        # Build messages list
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Get parameters
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/GraphReasoning/LLM-Graph", # Optional
            "X-Title": "LLM Graph Reasoning Framework", # Optional
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Call Open Router API
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Check for rate limits (429)
                if response.status_code == 429:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', 'Rate limit exceeded')
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit ({error_msg}). Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"Open Router rate limit exceeded after {max_retries} retries: {error_msg}")
                
                # Check for other errors
                response.raise_for_status()
                
                # Extract response
                result = response.json()
                
                content = result['choices'][0]['message']['content']
                finish_reason = result['choices'][0]['finish_reason']
                
                # Usage stats might be missing in some responses
                usage = result.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
    
                # Track usage
                self.total_tokens_used += tokens_used
                self.total_requests += 1
    
                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=tokens_used,
                    finish_reason=finish_reason,
                    metadata={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )
                
            except requests.exceptions.RequestException as e:
                # Handle connection errors differently? For now, just fail or basic retry
                # If it's not a rate limit (handled above)
                if attempt < max_retries - 1 and isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                     delay = base_delay * (2 ** attempt)
                     print(f"Connection error: {e}. Retrying in {delay}s...")
                     time.sleep(delay)
                     continue
                
                raise RuntimeError(f"Open Router API call failed: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Open Router API call failed: {str(e)}")
    
    def generate_structured(
        self, 
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output matching schema.
        
        Args:
            prompt: User prompt
            schema: JSON schema for output structure
            system_message: Optional system message
            
        Returns:
            Parsed JSON output matching schema
        """
        # Add schema instruction to prompt
        schema_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        # Generate response
        response = self.generate(schema_prompt, system_message)
        
        # Parse JSON from response
        try:
            # Try to extract JSON from markdown code blocks if present
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}\nResponse: {response.content}")
