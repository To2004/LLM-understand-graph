"""
LLM model integrations and adapters
"""

from .base import BaseLLMClient, LLMResponse
from .openrouter_client import OpenRouterClient
from .model_configs import (
    get_model_id,
    get_recommended_model,
    list_available_models,
    FREE_MODELS,
    TASK_RECOMMENDATIONS
)

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "OpenRouterClient",
    "get_model_id",
    "get_recommended_model",
    "list_available_models",
    "FREE_MODELS",
    "TASK_RECOMMENDATIONS"
]
