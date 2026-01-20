"""
Agent Chooser: Classifies tasks and selects appropriate graph algorithms

This module analyzes natural language queries to determine the type of graph
problem and selects the most appropriate algorithm to solve it.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pydantic import BaseModel
from .prompts import ChooserPrompts


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
        
        # Validate and repair parameters if graph structure is available
        if graph_structure:
            base_parameters = self._validate_and_repair_parameters(
                task_query, 
                base_parameters, 
                graph_structure
            )
        
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

    def _validate_and_repair_parameters(
        self,
        query: str,
        parameters: Dict[str, Any],
        graph_structure: Any
    ) -> Dict[str, Any]:
        """
        Validate parameters against graph structure and repair if needed.
        
        Args:
            query: Original task query
            parameters: Extracted parameters to validate
            graph_structure: Graph structure with node list
            
        Returns:
            Validated (and potentially repaired) parameters
        """
        if not graph_structure:
            return parameters
            
        # Get node list safely
        nodes = []
        if hasattr(graph_structure, 'nodes'):
            nodes = graph_structure.nodes
        elif isinstance(graph_structure, dict) and 'nodes' in graph_structure:
            nodes = graph_structure['nodes']
        else:
            return parameters # Cannot validate
            
        nodes_set = set(nodes)
        needs_repair = False
        
        # Check source
        if 'source' in parameters:
            if parameters['source'] not in nodes_set:
                print(f"[Chooser] Invalid source '{parameters['source']}' not in graph. Repairing...")
                needs_repair = True
                
        # Check target
        if 'target' in parameters:
            if parameters['target'] not in nodes_set:
                print(f"[Chooser] Invalid target '{parameters['target']}' not in graph. Repairing...")
                needs_repair = True

        # Check sink
        if 'sink' in parameters:
            if parameters['sink'] not in nodes_set:
                print(f"[Chooser] Invalid sink '{parameters['sink']}' not in graph. Repairing...")
                needs_repair = True
                
        if needs_repair:
            # Re-extract with knowledge of valid nodes
            print(f"[Chooser] Attempting to repair parameters using valid nodes: {nodes}")
            try:
                new_params = self._extract_parameters(query, list(nodes) if isinstance(nodes, list) else list(nodes_set))
                # Only keep found parameters that are actually valid now
                if 'found_parameters' in new_params:
                    del new_params['found_parameters']
                    
                print(f"[Chooser] Repaired parameters: {new_params}")
                # Update parameters with new values
                parameters.update(new_params)
            except Exception as e:
                print(f"[Chooser] Failed to repair parameters: {e}")
                
        return parameters
    
    def _classify_task_type(self, query: str) -> tuple:
        """
        Classify the task type from query using LLM.
        
        Returns:
            Tuple of (TaskType, parameters dict, confidence score)
        """
        print(f"[Chooser] Classifying task type for query: {query[:100]}...")
        prompt = ChooserPrompts.format_task_classification_prompt(query)
        
        # generate_structured returns a dict, not a string
        parsed = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_TASK_CLASSIFICATION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        # Convert string to TaskType enum
        task_type = TaskType(parsed["task_type"])
        print(f"[Chooser] Classified as: {task_type.value} (confidence: {parsed.get('confidence', 0.0)})")
        
        # Normalize parameters for specific task types
        parameters = parsed.get("parameters", {})
        
        # Fix: Maximum flow expects 'sink' not 'target'
        if task_type == TaskType.MAXIMUM_FLOW:
            if 'target' in parameters and 'sink' not in parameters:
                parameters['sink'] = parameters.pop('target')
                print(f"[Chooser] Normalized 'target' → 'sink' for maximum flow")
        
        return (
            task_type,
            parameters,
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
        
        # generate_structured returns a dict, not a string
        parsed = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_PARAMETER_EXTRACTION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        return parsed
    
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
        # Handle case where graph_structure is not available (parallel execution)
        if graph_structure is None:
            directed = True  # Default assumption
            weighted = False  # Default assumption
            num_nodes = 0  # Unknown
            num_edges = 0  # Unknown
            print("[Chooser] No graph structure available, using defaults")
        else:
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
        
        # generate_structured returns a dict, not a string
        parsed = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ChooserPrompts.SCHEMA_ALGORITHM_SELECTION,
            system_message=ChooserPrompts.SYSTEM_MESSAGE
        )
        
        # Handle unimplemented algorithms with fallbacks
        algorithm_name = parsed.get("algorithm_name")
        reasoning = parsed.get("reasoning", "")
        
        if algorithm_name is None or algorithm_name == "None" or algorithm_name == "":
            # Provide fallback algorithms for unimplemented task types
            if task_type == TaskType.HAMILTONIAN_PATH:
                algorithm_name = "has_cycle"
                reasoning = "Hamiltonian path algorithm not implemented. Using cycle detection as fallback to check for Hamiltonian cycles."
                print(f"[Chooser] Fallback: Using has_cycle for Hamiltonian path")
            elif task_type == TaskType.GNN_MESSAGE_PASSING:
                algorithm_name = "find_all_paths"
                reasoning = "GNN message passing not implemented. Using path finding as a simple fallback."
                print(f"[Chooser] Fallback: Using find_all_paths for GNN message passing")
            else:
                # No fallback available
                algorithm_name = "is_connected"
                reasoning = f"No algorithm available for {task_type.value}. Using basic connectivity check as fallback."
                print(f"[Chooser] Warning: No algorithm for {task_type.value}, using is_connected as fallback")
        
        print(f"[Chooser] Selected algorithm: {algorithm_name}")
        print(f"[Chooser] Reasoning: {reasoning}")
        
        return (
            algorithm_name,
            reasoning
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
