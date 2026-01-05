"""
Agent Chooser: Classifies tasks and selects appropriate graph algorithms

This module analyzes natural language queries to determine the type of graph
problem and selects the most appropriate algorithm to solve it.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pydantic import BaseModel
from agents.prompts import ChooserPrompts


class TaskType(Enum):
    """Supported graph reasoning task types"""
    CONNECTIVITY = "connectivity"
    CYCLE_DETECTION = "cycle_detection"
    TOPOLOGICAL_SORT = "topological_sort"
    SHORTEST_PATH = "shortest_path"
    MAXIMUM_FLOW = "maximum_flow"
    BIPARTITE_MATCHING = "bipartite_matching"
    HAMILTONIAN_PATH = "hamiltonian_path"
    GNN_MESSAGE_PASSING = "gnn_message_passing"


class AlgorithmChoice(BaseModel):
    """Selected algorithm with parameters"""
    task_type: TaskType
    algorithm_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: Optional[str] = None


class AgentChooser:
    """
    Classifies graph tasks and selects appropriate algorithms.
    """
    
    def __init__(self, llm_client, algorithm_registry: Optional[Dict] = None):
        """
        Initialize the algorithm chooser.
        
        Args:
            llm_client: The LLM client for task classification
            algorithm_registry: Optional mapping of tasks to algorithms (uses default if None)
        """
        self.llm_client = llm_client
        self.algorithm_registry = algorithm_registry
    
    def choose_algorithm(
        self, 
        task_query: str,
        graph_structure: Optional[Any] = None
    ) -> AlgorithmChoice:
        """
        Select the appropriate algorithm for a task query.
        
        Args:
            task_query: Natural language description of the task
            graph_structure: Optional GraphStructure from parser for better selection
            
        Returns:
            AlgorithmChoice with selected algorithm and parameters
        """
        # Classify task type and extract base parameters
        task_type, base_parameters, confidence = self._classify_task_type(task_query)
        
        # Select specific algorithm based on task and graph properties
        algorithm_name, reasoning = self._select_algorithm(
            task_type,
            base_parameters,
            graph_structure
        )
        
        return AlgorithmChoice(
            task_type=task_type,
            algorithm_name=algorithm_name,
            parameters=base_parameters,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _classify_task_type(self, query: str) -> tuple:
        """
        Classify the task type from query using LLM.
        
        Returns:
            Tuple of (TaskType, parameters dict, confidence score)
        """
        prompt = ChooserPrompts.format_task_classification_prompt(query)
        
        response = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_TASK_CLASSIFICATION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        parsed = self._parse_json_response(response)
        
        # Convert string to TaskType enum
        task_type = TaskType(parsed["task_type"])
        
        return (
            task_type,
            parsed.get("parameters", {}),
            parsed.get("confidence", 0.0)
        )
    
    def _extract_parameters(
        self, 
        query: str, 
        nodes: List[str]
    ) -> Dict[str, Any]:
        """
        Extract algorithm-specific parameters from query.
        
        Args:
            query: Natural language query
            nodes: List of node identifiers in the graph
            
        Returns:
            Dictionary of extracted parameters
        """
        prompt = ChooserPrompts.format_parameter_extraction_prompt(query, nodes)
        
        response = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_PARAMETER_EXTRACTION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        return self._parse_json_response(response)
    
    def _select_algorithm(
        self, 
        task_type: TaskType, 
        parameters: Dict[str, Any],
        graph_structure: Optional[Any] = None
    ) -> tuple:
        """
        Select specific algorithm based on task and graph properties.
        
        Args:
            task_type: Classified task type
            parameters: Extracted parameters
            graph_structure: Optional GraphStructure from parser
            
        Returns:
            Tuple of (algorithm_name, reasoning)
        """
        directed = graph_structure.directed
        weighted = graph_structure.weighted
        num_nodes = len(graph_structure.nodes)
        num_edges = len(graph_structure.edges)
        
        # Use LLM to select the best algorithm
        prompt = ChooserPrompts.format_algorithm_selection_prompt(
            task_type=task_type.value,
            parameters=parameters,
            directed=directed,
            weighted=weighted,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
        
        response = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_ALGORITHM_SELECTION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        parsed = self._parse_json_response(response)
        
        return (
            parsed["algorithm_name"],
            parsed.get("reasoning", "")
        )
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON string from LLM response.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Parsed dictionary
            
        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    def validate_choice(
        self, 
        choice: AlgorithmChoice, 
        graph_structure: Any
    ) -> bool:
        """
        Validate that chosen algorithm is applicable to the graph.
        
        Args:
            choice: Algorithm choice to validate
            graph_structure: The GraphStructure from parser
            
        Returns:
            True if valid, False otherwise
        """
        # Check if task type has registered algorithms
        if choice.task_type not in self.algorithm_registry:
            return False
        
        # Check if chosen algorithm is in registry for this task
        available = self.algorithm_registry[choice.task_type]
        if choice.algorithm_name not in available:
            return False
        
        # Validate required parameters are present based on task type
        if choice.task_type in [TaskType.CONNECTIVITY, TaskType.SHORTEST_PATH]:
            # These require source and target nodes
            if "source" not in choice.parameters or "target" not in choice.parameters:
                return False
            
            # Verify nodes exist in graph
            if graph_structure:
                nodes = set(graph_structure.nodes)
                if choice.parameters["source"] not in nodes:
                    return False
                if choice.parameters["target"] not in nodes:
                    return False
        
        if choice.task_type == TaskType.MAXIMUM_FLOW:
            # Flow requires source and sink
            if "source" not in choice.parameters or "sink" not in choice.parameters:
                return False
            
            # Verify nodes exist in graph
            if graph_structure:
                nodes = set(graph_structure.nodes)
                if choice.parameters["source"] not in nodes:
                    return False
                if choice.parameters["sink"] not in nodes:
                    return False
        
        return True
