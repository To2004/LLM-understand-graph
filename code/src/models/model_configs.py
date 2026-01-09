"""
Recommended model configurations for OpenRouter.
Free models with strong performance.
"""

from typing import Dict, Any


# Recommended free models on OpenRouter (as of January 2025)
# Verified available models from https://openrouter.ai/collections/free-models
FREE_MODELS = {
    "llama-3.3-70b": {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Meta Llama 3.3 70B Instruct",
        "description": "Best overall free model - excellent reasoning and instruction following",
        "context_window": 131000,
        "strengths": ["reasoning", "instruction_following", "structured_output"],
        "recommended_for": ["parsing", "algorithm_selection", "complex_reasoning"]
    },
    "deepseek-r1": {
        "id": "deepseek/deepseek-r1-0528:free",
        "name": "DeepSeek R1",
        "description": "Powerful reasoning model with strong problem-solving capabilities",
        "context_window": 164000,
        "strengths": ["advanced_reasoning", "problem_solving", "math", "logical_thinking"],
        "recommended_for": ["complex_reasoning", "algorithm_design", "math_problems"]
    },
    "devstral-2": {
        "id": "mistralai/devstral-2512:free",
        "name": "Mistral Devstral 2",
        "description": "Specialized for agentic coding with 123B parameters and 256K context",
        "context_window": 262000,
        "strengths": ["code_generation", "agentic_coding", "large_context"],
        "recommended_for": ["code_tasks", "algorithm_implementation", "complex_coding"]
    },
    "mimo-v2-flash": {
        "id": "xiaomi/mimo-v2-flash:free",
        "name": "Xiaomi MiMo V2 Flash",
        "description": "Fast MoE model with 309B params (15B active), hybrid attention",
        "context_window": 262000,
        "strengths": ["speed", "large_context", "efficient", "balanced"],
        "recommended_for": ["general_tasks", "quick_responses", "large_documents"]
    },
    "qwen3-coder": {
        "id": "qwen/qwen3-coder:free",
        "name": "Qwen 3 Coder",
        "description": "Specialized for code generation and technical tasks",
        "context_window": 262000,
        "strengths": ["code_generation", "technical_reasoning", "math", "large_context"],
        "recommended_for": ["code_tasks", "algorithm_implementation", "technical_docs"]
    },
    "gemma-3-27b": {
        "id": "google/gemma-3-27b-it:free",
        "name": "Google Gemma 3 27B",
        "description": "Latest Gemma model with strong balanced performance",
        "context_window": 131000,
        "strengths": ["instruction_following", "balanced_performance", "reasoning"],
        "recommended_for": ["general_tasks", "instruction_following", "analysis"]
    },
    "glm-4.5-air": {
        "id": "z-ai/glm-4.5-air:free",
        "name": "GLM 4.5 Air",
        "description": "Lightweight MoE model with efficient architecture",
        "context_window": 131000,
        "strengths": ["efficiency", "balanced", "speed"],
        "recommended_for": ["quick_tasks", "general_purpose", "simple_reasoning"]
    },
    "deepseek-chimera": {
        "id": "tngtech/deepseek-r1t2-chimera:free",
        "name": "DeepSeek TNG R1T2 Chimera",
        "description": "671B-parameter MoE model from DeepSeek checkpoints",
        "context_window": 164000,
        "strengths": ["advanced_reasoning", "large_scale", "versatile"],
        "recommended_for": ["complex_tasks", "detailed_analysis", "research"]
    },
    "kat-coder-pro": {
        "id": "kwaipilot/kat-coder-pro:free",
        "name": "Kwai KAT Coder Pro",
        "description": "Professional coding model with large context",
        "context_window": 256000,
        "strengths": ["code_generation", "large_context", "professional_coding"],
        "recommended_for": ["code_tasks", "large_codebases", "refactoring"]
    },
    "nemotron-nano-12b": {
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "name": "NVIDIA Nemotron Nano 12B VL",
        "description": "NVIDIA's small efficient model with vision capabilities",
        "context_window": 128000,
        "strengths": ["efficiency", "multimodal", "vision"],
        "recommended_for": ["quick_tasks", "vision_tasks", "efficient_processing"]
    }
}


# Default model recommendations by task type
TASK_RECOMMENDATIONS = {
    "parser": "llama-3.3-70b",  # Best reasoning for graph parsing
    "chooser": "deepseek-r1",  # Advanced reasoning for algorithm selection
    "verifier": "mimo-v2-flash",  # Very fast with large context for verification
    "general": "gemma-3-27b",  # Good balanced default for most tasks
    "code": "devstral-2",  # Best for code-related tasks with large context
    "reasoning": "deepseek-r1",  # Best for complex reasoning
}


def get_model_id(model_key: str) -> str:
    """
    Get the full model ID from a short key.
    
    Args:
        model_key: Short model key (e.g., "llama-3.3-70b")
        
    Returns:
        Full OpenRouter model ID
        
    Example:
        >>> get_model_id("llama-3.3-70b")
        'meta-llama/llama-3.3-70b-instruct:free'
    """
    if model_key in FREE_MODELS:
        return FREE_MODELS[model_key]["id"]
    # If already a full ID, return as-is
    return model_key


def get_recommended_model(task_type: str = "general") -> str:
    """
    Get recommended model ID for a specific task type.
    
    Args:
        task_type: Type of task (parser, chooser, verifier, general)
        
    Returns:
        Full OpenRouter model ID
        
    Example:
        >>> get_recommended_model("parser")
        'meta-llama/llama-3.3-70b-instruct:free'
    """
    model_key = TASK_RECOMMENDATIONS.get(task_type, "general")
    return get_model_id(model_key)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all available free models with their details.
    
    Returns:
        Dictionary of model configurations
    """
    return FREE_MODELS


def print_model_info():
    """Print information about all available free models."""
    print("=" * 80)
    print("AVAILABLE FREE MODELS ON OPENROUTER")
    print("=" * 80)
    print()
    
    for key, config in FREE_MODELS.items():
        print(f"Key: {key}")
        print(f"  Name: {config['name']}")
        print(f"  ID: {config['id']}")
        print(f"  Description: {config['description']}")
        print(f"  Context Window: {config['context_window']:,} tokens")
        print(f"  Strengths: {', '.join(config['strengths'])}")
        print(f"  Recommended for: {', '.join(config['recommended_for'])}")
        print()
    
    print("=" * 80)
    print("TASK RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    for task, model_key in TASK_RECOMMENDATIONS.items():
        model_name = FREE_MODELS[model_key]["name"]
        print(f"  {task.capitalize()}: {model_name}")
    print()


if __name__ == "__main__":
    print_model_info()
