"""
Prompt Decomposer: Splits user input into graph_context and task_context
"""

from typing import Tuple
from pydantic import BaseModel


class DecompositionResult(BaseModel):
    """Result of prompt decomposition"""
    graph_context: str
    task_context: str
    confidence: float


class PromptDecomposer:
    """
    Splits user input into two distinct chunks:
    - graph_context: The graph structure description
    - task_context: The query/task to perform
    """
    
    DECOMPOSITION_PROMPT = """You are a prompt decomposer for a graph reasoning system.

Your task is to split the user's input into TWO parts:
1. **graph_context**: The description of the graph structure (nodes, edges, weights, etc.)
2. **task_context**: The question or task to perform on the graph

Examples:

Input: "Graph: A--B, B--C. Task: Is A connected to C?"
Output:
{{
    "graph_context": "Graph: A--B, B--C",
    "task_context": "Is A connected to C?",
    "confidence": 0.95
}}

Input: "Given nodes 1,2,3,4 with edges 1-2, 2-3, 3-4. Find the shortest path from 1 to 4."
Output:
{{
    "graph_context": "nodes 1,2,3,4 with edges 1-2, 2-3, 3-4",
    "task_context": "Find the shortest path from 1 to 4",
    "confidence": 0.9
}}

Input: "In a directed graph where A points to B, B points to C, and C points to A, detect if there's a cycle."
Output:
{{
    "graph_context": "directed graph where A points to B, B points to C, and C points to A",
    "task_context": "detect if there's a cycle",
    "confidence": 0.85
}}

Now decompose this input:

User Input: {prompt}

Respond ONLY with JSON in this format:
{{
    "graph_context": "...",
    "task_context": "...",
    "confidence": 0.0-1.0
}}

Response:"""

    def __init__(self, llm_client):
        """
        Initialize decomposer with LLM client.
        
        Args:
            llm_client: The LLM client for decomposition
        """
        self.llm_client = llm_client
        
    def decompose(self, prompt: str) -> DecompositionResult:
        """
        Split prompt into graph_context and task_context.
        
        Args:
            prompt: User input to decompose
            
        Returns:
            DecompositionResult with both contexts
        """
        # Try LLM-based decomposition first
        try:
            return self._llm_decomposition(prompt)
        except Exception as e:
            # Fallback to heuristic decomposition
            return self._heuristic_decomposition(prompt)
    
    def _llm_decomposition(self, prompt: str) -> DecompositionResult:
        """
        Use LLM to decompose prompt.
        
        Args:
            prompt: User input
            
        Returns:
            DecompositionResult from LLM analysis
        """
        decomposition_prompt = self.DECOMPOSITION_PROMPT.format(prompt=prompt)
        
        response = self.llm_client.generate(decomposition_prompt)
        
        # Parse JSON response - extract content from LLMResponse
        import json
        response_text = response.content.strip()
        
        # Extract JSON from response
        if '{' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            result_dict = json.loads(json_str)
            
            return DecompositionResult(
                graph_context=result_dict.get('graph_context', ''),
                task_context=result_dict.get('task_context', ''),
                confidence=result_dict.get('confidence', 0.5)
            )
        else:
            raise ValueError("Could not parse JSON from LLM response")
    
    def _heuristic_decomposition(self, prompt: str) -> DecompositionResult:
        """
        Fallback heuristic decomposition using simple rules.
        
        Args:
            prompt: User input
            
        Returns:
            DecompositionResult using heuristics
        """
        # Look for common separators
        separators = [
            'Task:', 'task:', 'Question:', 'question:', 'Query:', 'query:',
            'Find', 'find', 'Is', 'is', 'Does', 'does', 'Check', 'check'
        ]
        
        graph_context = prompt
        task_context = prompt
        
        # Try to split on separators
        for sep in separators:
            if sep in prompt:
                parts = prompt.split(sep, 1)
                if len(parts) == 2:
                    graph_context = parts[0].strip()
                    task_context = sep + parts[1].strip() if sep.endswith(':') else parts[1].strip()
                    break
        
        # If no separator found, try to extract task from end
        if graph_context == task_context:
            # Look for question marks or imperative verbs at the end
            sentences = prompt.split('.')
            if len(sentences) > 1:
                graph_context = '.'.join(sentences[:-1]).strip()
                task_context = sentences[-1].strip()
        
        return DecompositionResult(
            graph_context=graph_context,
            task_context=task_context,
            confidence=0.6  # Lower confidence for heuristic
        )
    
    def _extract_graph_description(self, prompt: str) -> str:
        """
        Extract the graph structure description.
        
        Args:
            prompt: User input
            
        Returns:
            Graph description portion
        """
        # This is a helper method for future enhancements
        return prompt
    
    def _extract_task_query(self, prompt: str) -> str:
        """
        Extract the task/query portion.
        
        Args:
            prompt: User input
            
        Returns:
            Task query portion
        """
        # This is a helper method for future enhancements
        return prompt
