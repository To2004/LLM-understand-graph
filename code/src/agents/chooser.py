"""
Agent Chooser: Selects appropriate graph algorithms for given tasks

This module classifies task types and selects the correct classical graph 
algorithm to execute (connectivity, shortest path, maximum flow, etc.).
"""

from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel


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


class AgentChooser:
    """
    Classifies graph tasks and selects appropriate algorithms.
    
    TODO: Team Member Assignment - [CHOOSER TEAM]
    
    Priority: HIGH
    Estimated Time: 2-3 weeks
    """
    
    def __init__(self, llm_client, algorithm_registry: Dict = None):
        """
        Initialize the algorithm chooser.
        
        Args:
            llm_client: The LLM client for task classification
            algorithm_registry: Mapping of tasks to algorithms
            
        TODO [CHOOSER-001]:
            - Load algorithm registry from config
            - Initialize task classification prompts
            - Set up few-shot examples for classification
            - Configure confidence thresholds
        """
        self.llm_client = llm_client
        self.algorithm_registry = algorithm_registry or {}
        # TODO: Implement initialization
        pass
    
    def choose_algorithm(self, task_query: str) -> AlgorithmChoice:
        """
        Select the appropriate algorithm for a task query.
        
        Args:
            task_query: Natural language description of the task
            
        Returns:
            AlgorithmChoice with selected algorithm and parameters
            
        TODO [CHOOSER-002]:
            - Classify task type using LLM
            - Extract algorithm parameters from query
            - Map task to specific algorithm implementation
            - Calculate confidence score
            - Handle ambiguous queries
        
        Example:
            >>> query = "Find the shortest path from A to D"
            >>> choice = chooser.choose_algorithm(query)
            >>> choice.algorithm_name
            'dijkstra'
        """
        # TODO: Implement algorithm selection
        raise NotImplementedError("AgentChooser.choose_algorithm() not yet implemented")
    
    def _classify_task_type(self, query: str) -> TaskType:
        """
        Classify the task type from query.
        
        TODO [CHOOSER-003]:
            - Use LLM with classification prompt
            - Implement keyword matching for common patterns
            - Handle multi-task queries
            - Return confidence scores
        """
        # TODO: Implement task classification
        raise NotImplementedError()
    
    def _extract_parameters(
        self, 
        query: str, 
        task_type: TaskType
    ) -> Dict[str, Any]:
        """
        Extract algorithm-specific parameters from query.
        
        TODO [CHOOSER-004]:
            - Extract source/target nodes for path queries
            - Extract capacity constraints for flow problems
            - Parse optional parameters (weighted, directed, etc.)
            - Validate parameter consistency
        """
        # TODO: Implement parameter extraction
        raise NotImplementedError()
    
    def _select_algorithm(
        self, 
        task_type: TaskType, 
        graph_properties: Dict[str, Any]
    ) -> str:
        """
        Select specific algorithm based on task and graph properties.
        
        TODO [CHOOSER-005]:
            - Choose BFS vs DFS for connectivity
            - Choose Dijkstra vs Bellman-Ford for shortest path
            - Consider graph size and density
            - Handle special cases (negative weights, etc.)
        """
        # TODO: Implement algorithm selection logic
        raise NotImplementedError()
    
    def validate_choice(
        self, 
        choice: AlgorithmChoice, 
        graph_structure: Any
    ) -> bool:
        """
        Validate that chosen algorithm is applicable to the graph.
        
        TODO [CHOOSER-006]:
            - Check if graph satisfies algorithm preconditions
            - Verify required parameters are present
            - Warn about potential performance issues
            - Suggest alternatives if needed
        """
        # TODO: Implement validation
        raise NotImplementedError()
