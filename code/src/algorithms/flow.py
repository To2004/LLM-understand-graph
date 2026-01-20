"""
Flow algorithms: Maximum flow, minimum cut
"""

import networkx as nx
from typing import Dict, Any, Tuple


class FlowAlgorithms:
    """
    Flow and cut graph algorithms.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM - Flow]
    
    Priority: LOW
    Estimated Time: 1 week
    """
    
    @staticmethod
    def maximum_flow(
        graph: nx.DiGraph,
        source: str,
        sink: str,
        capacity: str = 'capacity'
    ) -> Tuple[float, Dict]:
        """
        Compute maximum flow from source to sink.
        
        Args:
            graph: NetworkX directed graph
            source: Source node
            sink: Sink node
            capacity: Edge attribute for capacity (default: 'capacity')
            
        Returns:
            Tuple of (flow_value, flow_dict)
            flow_dict maps edge (u, v) to flow amount
            
        Note:
            Uses the Edmonds-Karp algorithm (BFS-based Ford-Fulkerson)
        """
        try:
            # Check if capacity attribute exists, fallback to weight if needed
            # Common issue: parsed graphs might put capacity in 'weight' attribute
            test_edge = next(iter(graph.edges(data=True)), None)
            if test_edge and capacity not in test_edge[2] and 'weight' in test_edge[2]:
                capacity = 'weight'

            # Compute maximum flow using Edmonds-Karp algorithm
            flow_value, flow_dict = nx.maximum_flow(
                graph, 
                source, 
                sink, 
                capacity=capacity,
                flow_func=nx.algorithms.flow.edmonds_karp
            )
            return (flow_value, flow_dict)
        except nx.NetworkXError as e:
            raise nx.NetworkXError(f"Maximum flow error: {e}")
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found: {e}")
    
    @staticmethod
    def minimum_cut(
        graph: nx.DiGraph,
        source: str,
        sink: str,
        capacity: str = 'capacity'
    ) -> Tuple[float, Tuple[set, set]]:
        """
        Compute minimum cut.
        
        Args:
            graph: NetworkX directed graph
            source: Source node
            sink: Sink node
            capacity: Edge attribute for capacity (default: 'capacity')
            
        Returns:
            Tuple of (cut_value, (reachable_set, non_reachable_set))
            The cut_value equals the maximum flow value (max-flow min-cut theorem)
            
        Note:
            The minimum cut partitions nodes into two sets, verifying max-flow min-cut theorem
        """
        try:
            # Check if capacity attribute exists, fallback to weight if needed
            test_edge = next(iter(graph.edges(data=True)), None)
            if test_edge and capacity not in test_edge[2] and 'weight' in test_edge[2]:
                capacity = 'weight'
                
            # Compute minimum cut
            cut_value, partition = nx.minimum_cut(
                graph,
                source,
                sink,
                capacity=capacity
            )
            return (cut_value, partition)
        except nx.NetworkXError as e:
            raise nx.NetworkXError(f"Minimum cut error: {e}")
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found: {e}")
    
    @staticmethod
    def min_cost_flow(
        graph: nx.DiGraph,
        demand: Dict[str, int],
        capacity: str = 'capacity',
        weight: str = 'weight'
    ) -> Dict:
        """
        Compute minimum cost flow.
        
        Args:
            graph: NetworkX directed graph
            demand: Dictionary mapping nodes to supply (positive) or demand (negative)
                   Sum of all demands must equal zero (flow conservation)
            capacity: Edge attribute for capacity (default: 'capacity')
            weight: Edge attribute for cost per unit flow (default: 'weight')
            
        Returns:
            Flow dictionary mapping edges (u, v) to flow amount
            
        Raises:
            NetworkXUnfeasible: If no feasible flow exists
            
        Note:
            Positive demand means supply node, negative means demand node.
            Uses network simplex algorithm.
        """
        try:
            # Add demand attribute to nodes
            for node, dem in demand.items():
                if node in graph:
                    graph.nodes[node]['demand'] = dem
            
            # Check if capacity/weight attribute exists, fallback logic akin to max flow
            test_edge = next(iter(graph.edges(data=True)), None)
            if test_edge:
                if capacity not in test_edge[2] and 'weight' in test_edge[2]:
                    # If capacity is missing but weight is present, 
                    # we must be careful. Min cost flow needs BOTH capacity and cost (weight).
                    # If we only have one attribute, we typically can't do min cost flow easily 
                    # unless we assume unit cost or infinite capacity.
                    # As a heuristic for this generic agent: 
                    # if 'weight' is present but 'capacity' is not, assume 'weight' is capacity 
                    # and cost is 1 (or vice versa? tricky).
                    # For now, let's assume 'weight' in graph map corresponds to 'capacity' arg
                    # if we are doing standard flow, but min_cost_flow is more complex.
                    # Let's simple check logic: if we don't have capacity, try to use weight as capacity
                    # and default cost (weight param) to 1 if missing? 
                    # NetworkX defaults weight to 0 and capacity to infinite if missing.
                    pass 

            # Compute minimum cost flow
            flow_dict = nx.min_cost_flow(
                graph,
                capacity=capacity,
                weight=weight
            )
            
            return flow_dict
        except nx.NetworkXUnfeasible as e:
            raise nx.NetworkXUnfeasible(f"No feasible flow exists: {e}")
        except nx.NetworkXError as e:
            raise nx.NetworkXError(f"Min-cost flow error: {e}")
