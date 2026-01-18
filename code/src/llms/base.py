"""
Base LLM client interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized LLM response"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Dict[str, Any] = {}


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    TODO: Team Member Assignment - [MODELS TEAM]
    
    Priority: HIGH
    Estimated Time: 1-2 weeks
    """
    
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        """
        Initialize LLM client.

        Args:
            model_name: Name/ID of the model
            temperature: Sampling temperature (0.0 for deterministic)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.config = kwargs

        # Configure retry logic
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)

        # Token tracking
        self.total_tokens_used = 0
        self.total_requests = 0
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with generated text
            
        TODO [MODELS-002]:
            - Implement model-specific API call
            - Handle errors and retries
            - Parse response into standard format
            - Track token usage
        """
        pass
    
    @abstractmethod
    def generate_structured(
        self, 
        prompt: str,
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate structured JSON output matching schema.
        
        Args:
            prompt: User prompt
            schema: JSON schema for output validation
            system_message: Optional system message
            
        Returns:
            JSON string that matches the schema
            
        TODO [MODELS-003]:
            - Use JSON mode or function calling
            - Validate output against schema
            - Retry if output doesn't match schema
            - Return as JSON string (caller will parse)
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None
    ) -> List[LLMResponse]:
        """
        Generate for multiple prompts (sequential by default, can be overridden for parallel).

        Args:
            prompts: List of prompts to generate from
            system_message: Optional system message for all prompts

        Returns:
            List of LLMResponse objects in same order as prompts
        """
        # Default implementation: sequential processing
        # Subclasses can override for parallel/async processing
        results = []
        for prompt in prompts:
            response = self.generate(prompt, system_message)
            results.append(response)
        return results

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get token usage statistics.

        Returns:
            Dictionary with total_tokens and total_requests
        """
        return {
            "total_tokens": self.total_tokens_used,
            "total_requests": self.total_requests
        }
