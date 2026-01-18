"""
Prompt Validator: Validates that user prompts are graph-related queries
"""

from typing import Optional
from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of prompt validation"""
    is_valid: bool
    rejection_reason: Optional[str] = None
    confidence: float


class PromptValidator:
    """
    Validates that user prompts contain graph reasoning tasks.
    
    Rejects non-graph queries like "What is the capital of France?"
    """
    
    VALIDATION_PROMPT = """You are a validator for a graph reasoning system.

Your task is to determine if the user's input contains a graph reasoning problem.

A VALID graph query must contain:
1. A graph description (nodes, edges, connections) OR reference to a graph
2. A task/question about the graph (connectivity, paths, cycles, etc.)

VALID examples:
- "Graph: A--B--C. Is A connected to C?"
- "Given nodes 1,2,3 with edges 1-2, 2-3. Find shortest path from 1 to 3."
- "In a graph with edges A->B, B->C, does a cycle exist?"

INVALID examples:
- "What is the capital of France?"
- "Solve 2 + 2"
- "Write me a poem"
- "How does photosynthesis work?"

User Input: {prompt}

Respond in JSON format:
{{
    "is_valid": true/false,
    "rejection_reason": "reason if invalid, null if valid",
    "confidence": 0.0-1.0
}}

Response:"""

    def __init__(self, llm_client):
        """
        Initialize validator with LLM client.
        
        Args:
            llm_client: The LLM client for semantic validation
        """
        self.llm_client = llm_client
        
    def validate(self, prompt: str) -> ValidationResult:
        """
        Check if prompt contains a graph reasoning task.
        
        Args:
            prompt: User input to validate
            
        Returns:
            ValidationResult with validation status
        """
        # First, quick keyword check
        if self._check_graph_keywords(prompt):
            # If keywords found, do LLM validation for semantic check
            return self._llm_validation(prompt)
        else:
            # No graph keywords, likely invalid
            llm_result = self._llm_validation(prompt)
            # If LLM says valid despite no keywords, trust it
            # Otherwise, reject
            return llm_result
    
    def _check_graph_keywords(self, prompt: str) -> bool:
        """
        Check for graph-related keywords.
        
        Args:
            prompt: User input
            
        Returns:
            True if graph keywords found
        """
        graph_keywords = [
            'graph', 'node', 'nodes', 'edge', 'edges', 'vertex', 'vertices',
            'path', 'cycle', 'connected', 'connectivity', 'shortest',
            'neighbor', 'adjacent', 'directed', 'undirected', 'weighted',
            'flow', 'matching', 'bipartite', 'tree', 'forest'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in graph_keywords)
    
    def _llm_validation(self, prompt: str) -> ValidationResult:
        """
        Use LLM to validate prompt semantically.
        
        Args:
            prompt: User input
            
        Returns:
            ValidationResult from LLM analysis
        """
        validation_prompt = self.VALIDATION_PROMPT.format(prompt=prompt)
        
        try:
            response = self.llm_client.generate(validation_prompt)
            
            # Parse JSON response - extract content from LLMResponse
            import json
            # Extract JSON from response content
            response_text = response.content.strip()
            
            # Try to find JSON in the response
            if '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result_dict = json.loads(json_str)
                
                return ValidationResult(
                    is_valid=result_dict.get('is_valid', False),
                    rejection_reason=result_dict.get('rejection_reason'),
                    confidence=result_dict.get('confidence', 0.5)
                )
            else:
                # Fallback: if response contains "valid" or similar
                if 'valid' in response_text.lower() and 'invalid' not in response_text.lower():
                    return ValidationResult(is_valid=True, confidence=0.7)
                else:
                    return ValidationResult(
                        is_valid=False,
                        rejection_reason="Could not parse validation response",
                        confidence=0.5
                    )
                    
        except Exception as e:
            # On error, be conservative and reject
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Validation error: {str(e)}",
                confidence=0.3
            )
