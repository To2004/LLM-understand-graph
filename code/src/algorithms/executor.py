"""
Algorithm Executor: Unified interface for executing graph algorithms

This module provides a single entry point for executing classical graph
algorithms using NetworkX, with proper error handling and result formatting.
"""

from typing import Any, Dict
import networkx as nx
from .connectivity import ConnectivityAlgorithms
from .cycles import CycleAlgorithms
from .shortest_path import ShortestPathAlgorithms
from .flow import FlowAlgorithms
from .matching import MatchingAlgorithms


class AlgorithmExecutor:
    """
    Executes classical graph algorithms via NetworkX.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM]
    
    Priority: MEDIUM
    Estimated Time: 2 weeks
    """
    
    def __init__(self):
        """
        Initialize algorithm executor with all available algorithms.
        
        Registers all implemented algorithms with their metadata.
        """
        self.algorithms = {}
        self._register_all_algorithms()
    
    def _register_all_algorithms(self):
        """Register all available graph algorithms."""
        # Connectivity algorithms
        self.register_algorithm(
            'is_connected',
            ConnectivityAlgorithms.is_connected,
            {'category': 'connectivity', 'complexity': 'O(V+E)'}
        )
        self.register_algorithm(
            'find_all_paths',
            ConnectivityAlgorithms.find_all_paths,
            {'category': 'connectivity', 'complexity': 'exponential'}
        )
        self.register_algorithm(
            'connected_components',
            ConnectivityAlgorithms.connected_components,
            {'category': 'connectivity', 'complexity': 'O(V+E)'}
        )
        
        # Cycle algorithms
        self.register_algorithm(
            'has_cycle',
            CycleAlgorithms.has_cycle,
            {'category': 'cycles', 'complexity': 'O(V+E)'}
        )
        self.register_algorithm(
            'topological_sort',
            CycleAlgorithms.topological_sort,
            {'category': 'cycles', 'complexity': 'O(V+E)', 'requires_dag': True}
        )
        self.register_algorithm(
            'find_all_cycles',
            CycleAlgorithms.find_all_cycles,
            {'category': 'cycles', 'complexity': 'exponential'}
        )
        
        # Shortest path algorithms
        self.register_algorithm(
            'dijkstra',
            ShortestPathAlgorithms.dijkstra,
            {'category': 'shortest_path', 'complexity': 'O((V+E)logV)', 'no_negative_weights': True}
        )
        self.register_algorithm(
            'bellman_ford',
            ShortestPathAlgorithms.bellman_ford,
            {'category': 'shortest_path', 'complexity': 'O(VE)', 'allows_negative_weights': True}
        )
        self.register_algorithm(
            'all_pairs_shortest_path',
            ShortestPathAlgorithms.all_pairs_shortest_path,
            {'category': 'shortest_path', 'complexity': 'O(V^3)'}
        )
        
        # Flow algorithms
        self.register_algorithm(
            'maximum_flow',
            FlowAlgorithms.maximum_flow,
            {'category': 'flow', 'complexity': 'O(V*E^2)', 'requires_directed': True}
        )
        self.register_algorithm(
            'minimum_cut',
            FlowAlgorithms.minimum_cut,
            {'category': 'flow', 'complexity': 'O(V*E^2)', 'requires_directed': True}
        )
        self.register_algorithm(
            'min_cost_flow',
            FlowAlgorithms.min_cost_flow,
            {'category': 'flow', 'complexity': 'polynomial', 'requires_directed': True}
        )
        
        # Matching algorithms
        self.register_algorithm(
            'maximum_matching',
            MatchingAlgorithms.maximum_matching,
            {'category': 'matching', 'complexity': 'O(V*E)'}
        )
        self.register_algorithm(
            'bipartite_matching',
            MatchingAlgorithms.bipartite_matching,
            {'category': 'matching', 'complexity': 'O(sqrt(V)*E)', 'requires_bipartite': True}
        )
        self.register_algorithm(
            'is_bipartite',
            MatchingAlgorithms.is_bipartite,
            {'category': 'matching', 'complexity': 'O(V+E)'}
        )
    
    def execute(
        self, 
        algorithm_name: str,
        graph: nx.Graph,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a graph algorithm.
        
        Args:
            algorithm_name: Name of algorithm to execute
            graph: NetworkX graph object
            parameters: Algorithm-specific parameters
            
        Returns:
            Algorithm result (path, value, boolean, etc.)
            
        Raises:
            ValueError: If algorithm not found or preconditions not met
            
        Example:
            >>> G = nx.Graph([('A','B'), ('B','C')])
            >>> result = executor.execute('dijkstra', G, {'source': 'A', 'target': 'C'})
            >>> result
            (['A', 'B', 'C'], 2.0)
        """
        # Check if algorithm exists
        if algorithm_name not in self.algorithms:
            available = ', '.join(self.algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available: {available}")
        
        # Get algorithm info
        algo_info = self.algorithms[algorithm_name]
        implementation = algo_info['implementation']
        metadata = algo_info['metadata']
        
        # Validate preconditions
        self._validate_preconditions(algorithm_name, graph, parameters, metadata)
        
        try:
            # Execute the algorithm
            result = implementation(graph, **parameters)
            return result
        except Exception as e:
            raise Exception(f"Error executing {algorithm_name}: {str(e)}")
    
    def _validate_preconditions(
        self, 
        algorithm_name: str,
        graph: nx.Graph,
        parameters: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Validate algorithm preconditions.
        
        Args:
            algorithm_name: Name of the algorithm
            graph: NetworkX graph
            parameters: Algorithm parameters
            metadata: Algorithm metadata with requirements
            
        Raises:
            ValueError: If preconditions are not met
        """
        # Check if directed graph is required
        if metadata.get('requires_directed', False) and not graph.is_directed():
            raise ValueError(f"{algorithm_name} requires a directed graph")
        
        # Check if DAG is required
        if metadata.get('requires_dag', False):
            if not graph.is_directed() or not nx.is_directed_acyclic_graph(graph):
                raise ValueError(f"{algorithm_name} requires a directed acyclic graph (DAG)")
        
        # Check if bipartite is required
        if metadata.get('requires_bipartite', False) and not nx.is_bipartite(graph):
            raise ValueError(f"{algorithm_name} requires a bipartite graph")
        
        # Check for negative weights if not allowed
        if metadata.get('no_negative_weights', False):
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1)
                if weight < 0:
                    raise ValueError(f"{algorithm_name} does not support negative edge weights")
        
        # Validate required parameters based on algorithm
        if algorithm_name in ['dijkstra', 'bellman_ford', 'is_connected']:
            if 'source' not in parameters or 'target' not in parameters:
                raise ValueError(f"{algorithm_name} requires 'source' and 'target' parameters")
            if parameters['source'] not in graph.nodes():
                raise ValueError(f"Source node '{parameters['source']}' not in graph")
            if parameters['target'] not in graph.nodes():
                raise ValueError(f"Target node '{parameters['target']}' not in graph")
        
        elif algorithm_name in ['maximum_flow', 'minimum_cut']:
            if 'source' not in parameters or 'sink' not in parameters:
                raise ValueError(f"{algorithm_name} requires 'source' and 'sink' parameters")
        
        return True
    
    def register_algorithm(
        self, 
        name: str, 
        implementation: callable,
        metadata: Dict[str, Any]
    ):
        """
        Register a new algorithm implementation.
        
        Args:
            name: Unique name for the algorithm
            implementation: Callable function that implements the algorithm
            metadata: Dictionary with algorithm info (complexity, preconditions, etc.)
        """
        self.algorithms[name] = {
            'implementation': implementation,
            'metadata': metadata
        }
