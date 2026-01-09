"""
Open Router API client implementation
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI
from .base import BaseLLMClient, LLMResponse


class OpenRouterClient(BaseLLMClient):
    """
    Open Router API client for accessing various open-source models.
    Uses OpenAI-compatible API interface.
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
        
        # Initialize OpenAI client with Open Router endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        
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
        
        try:
            # Call Open Router API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract response
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            tokens_used = response.usage.total_tokens

            # Track usage
            self.total_tokens_used += tokens_used
            self.total_requests += 1

            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
            
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
        import json
        
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
