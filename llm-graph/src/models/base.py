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
            
        TODO [MODELS-001]:
            - Store model configuration
            - Initialize client connection
            - Set default parameters
            - Configure retry logic
        """
        self.model_name = model_name
        self.temperature = temperature
        self.config = kwargs
        # TODO: Implement initialization
        pass
    
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
    ) -> Dict[str, Any]:
        """
        Generate structured output matching schema.
        
        TODO [MODELS-003]:
            - Use JSON mode or function calling
            - Validate output against schema
            - Retry if output doesn't match schema
            - Parse into Python dict
        """
        pass
    
    def batch_generate(
        self, 
        prompts: List[str],
        system_message: Optional[str] = None
    ) -> List[LLMResponse]:
        """
        Generate for multiple prompts (can be parallel).
        
        TODO [MODELS-004]:
            - Implement batch processing
            - Use async calls if supported
            - Handle rate limits
            - Return results in order
        """
        # TODO: Implement batch generation
        raise NotImplementedError()
