"""
Agent Synthesizer: Converts raw algorithm results into natural language responses
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SynthesisResult(BaseModel):
    """Result of synthesis with natural language response"""
    natural_language_response: str
    raw_result: Any
    metadata: Optional[Dict[str, Any]] = None


class AgentSynthesizer:
    """
    Converts raw algorithm output to natural language responses.
    
    Examples:
        - raw_result=4 → "The shortest path has length 4."
        - raw_result=True → "Yes, the nodes are connected."
        - raw_result=[A, B, C] → "The path is: A → B → C"
    """
    
    SYNTHESIS_TEMPLATE = """You are a helpful assistant that explains graph algorithm results in natural language.

Original Query: {task_query}
Algorithm Used: {algorithm_name}
Raw Result: {raw_result}

Provide a clear, concise natural language answer to the user's query based on the algorithm result.
Be direct and informative. Do not add unnecessary explanations unless the result is complex.

Response:"""

    def __init__(self, llm_client):
        """
        Initialize synthesizer with LLM client.
        
        Args:
            llm_client: The LLM client for response generation
        """
        self.llm_client = llm_client
    
    def synthesize(
        self,
        raw_result: Any,
        task_query: str,
        algorithm_name: str,
        graph_structure: Optional[Any] = None
    ) -> SynthesisResult:
        """
        Convert raw algorithm output to natural language.
        
        Args:
            raw_result: Raw output from algorithm
            task_query: Original task query
            algorithm_name: Name of algorithm used
            graph_structure: Optional graph structure for context
            
        Returns:
            SynthesisResult with natural language response
        """
        # Try rule-based formatting first for common cases
        formatted_response = self._try_rule_based_formatting(
            raw_result, task_query, algorithm_name
        )
        
        if formatted_response:
            return SynthesisResult(
                natural_language_response=formatted_response,
                raw_result=raw_result,
                metadata={'method': 'rule_based'}
            )
        
        # Fall back to LLM-based synthesis for complex cases
        return self._llm_synthesis(raw_result, task_query, algorithm_name)
    
    def _try_rule_based_formatting(
        self,
        raw_result: Any,
        task_query: str,
        algorithm_name: str
    ) -> Optional[str]:
        """
        Try to format result using simple rules.
        
        Args:
            raw_result: Raw result
            task_query: Task query
            algorithm_name: Algorithm name
            
        Returns:
            Formatted string or None if rules don't apply
        """
        # Boolean results
        if isinstance(raw_result, bool):
            return self._format_boolean_result(raw_result, task_query)
        
        # Numeric results
        if isinstance(raw_result, (int, float)):
            return self._format_numeric_result(raw_result, algorithm_name, task_query)
        
        # List/path results
        if isinstance(raw_result, (list, tuple)):
            return self._format_path_result(raw_result, task_query)
        
        # None/null results
        if raw_result is None:
            return "No result found. The query may not be applicable to this graph."
        
        return None
    
    def _format_boolean_result(self, value: bool, task_query: str) -> str:
        """
        Format yes/no results.
        
        Args:
            value: Boolean result
            task_query: Original query
            
        Returns:
            Natural language response
        """
        if value:
            # Try to extract what we're confirming
            if 'connected' in task_query.lower():
                return "Yes, the nodes are connected."
            elif 'cycle' in task_query.lower():
                return "Yes, a cycle exists in the graph."
            elif 'bipartite' in task_query.lower():
                return "Yes, the graph is bipartite."
            elif 'path' in task_query.lower():
                return "Yes, a path exists."
            else:
                return "Yes."
        else:
            if 'connected' in task_query.lower():
                return "No, the nodes are not connected."
            elif 'cycle' in task_query.lower():
                return "No, there is no cycle in the graph."
            elif 'bipartite' in task_query.lower():
                return "No, the graph is not bipartite."
            elif 'path' in task_query.lower():
                return "No, no path exists."
            else:
                return "No."
    
    def _format_numeric_result(
        self,
        value: float,
        algorithm_name: str,
        task_query: str
    ) -> str:
        """
        Format numeric results based on context.
        
        Args:
            value: Numeric result
            algorithm_name: Algorithm used
            task_query: Original query
            
        Returns:
            Natural language response
        """
        # Shortest path length
        if 'shortest' in task_query.lower() or 'shortest_path' in algorithm_name:
            if value == float('inf'):
                return "No path exists between the nodes."
            return f"The shortest path has length {value}."
        
        # Flow
        if 'flow' in task_query.lower() or 'flow' in algorithm_name:
            return f"The maximum flow is {value}."
        
        # Generic numeric result
        return f"The result is {value}."
    
    def _format_path_result(self, path: List[str], task_query: str) -> str:
        """
        Format path lists into readable strings.
        
        Args:
            path: List of nodes in path
            task_query: Original query
            
        Returns:
            Natural language response
        """
        if not path:
            return "No path found."
        
        if len(path) == 1:
            return f"The path contains only node {path[0]}."
        
        # Format as A → B → C
        path_str = " → ".join(str(node) for node in path)
        
        if 'shortest' in task_query.lower():
            return f"The shortest path is: {path_str}"
        elif 'cycle' in task_query.lower():
            return f"A cycle was found: {path_str}"
        else:
            return f"The path is: {path_str}"
    
    def _llm_synthesis(
        self,
        raw_result: Any,
        task_query: str,
        algorithm_name: str
    ) -> SynthesisResult:
        """
        Use LLM to synthesize natural language response.
        
        Args:
            raw_result: Raw result
            task_query: Task query
            algorithm_name: Algorithm name
            
        Returns:
            SynthesisResult with LLM-generated response
        """
        prompt = self.SYNTHESIS_TEMPLATE.format(
            task_query=task_query,
            algorithm_name=algorithm_name,
            raw_result=str(raw_result)
        )
        
        try:
            response = self.llm_client.generate(prompt)
            
            # Clean up response - extract content from LLMResponse
            response_text = response.content.strip()
            
            # Remove common prefixes
            prefixes_to_remove = [
                "Response:", "Answer:", "Result:", "Output:",
                "The answer is:", "The result is:"
            ]
            for prefix in prefixes_to_remove:
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix):].strip()
            
            return SynthesisResult(
                natural_language_response=response_text,
                raw_result=raw_result,
                metadata={'method': 'llm_based'}
            )
            
        except Exception as e:
            # Fallback to simple string representation
            return SynthesisResult(
                natural_language_response=f"The result is: {raw_result}",
                raw_result=raw_result,
                metadata={'method': 'fallback', 'error': str(e)}
            )
