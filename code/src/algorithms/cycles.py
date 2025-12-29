"""
Cycle detection and topological sorting algorithms
"""

import networkx as nx
from typing import List, Optional, Tuple


class CycleAlgorithms:
    """
    Cycle detection and topological sorting algorithms.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM - Cycles]
    
    Priority: MEDIUM
    Estimated Time: 1 week
    """
    
    @staticmethod
    def has_cycle(graph: nx.Graph) -> Tuple[bool, Optional[List[str]]]:
        """
        Detect if graph contains a cycle.
        
        Args:
            graph: NetworkX graph (directed or undirected)
            
        Returns:
            Tuple of (has_cycle, cycle), where cycle is a list of nodes forming
            a cycle if found, None otherwise
        """
        try:
            # find_cycle returns list of edges forming a cycle
            cycle_edges = nx.find_cycle(graph)
            # Extract nodes from edges
            cycle_nodes = [edge[0] for edge in cycle_edges]
            # Add the last node to complete the cycle
            if cycle_edges:
                cycle_nodes.append(cycle_edges[-1][1])
            return (True, cycle_nodes)
        except nx.NetworkXNoCycle:
            return (False, None)
    
    @staticmethod
    def topological_sort(graph: nx.DiGraph) -> List[str]:
        """
        Compute topological ordering of DAG.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            List of nodes in topological order
            
        Raises:
            NetworkXError: If graph contains a cycle (not a DAG)
        """
        try:
            # Verify it's a DAG and get topological order
            return list(nx.topological_sort(graph))
        except nx.NetworkXError as e:
            # This is raised if the graph contains a cycle
            raise nx.NetworkXError(f"Graph contains a cycle, cannot perform topological sort: {e}")
    
    @staticmethod
    def find_all_cycles(
        graph: nx.Graph,
        max_cycles: int = 100
    ) -> List[List[str]]:
        """
        Find all simple cycles in graph.
        
        Args:
            graph: NetworkX graph (directed or undirected)
            max_cycles: Maximum number of cycles to return (for performance)
            
        Returns:
            List of cycles, where each cycle is a list of nodes.
            Sorted by cycle length (shortest first).
        """
        cycles = []
        
        try:
            if graph.is_directed():
                # For directed graphs, use simple_cycles
                cycle_generator = nx.simple_cycles(graph)
            else:
                # For undirected graphs, use cycle_basis
                # cycle_basis returns fundamental cycles
                cycle_generator = nx.cycle_basis(graph)
            
            # Collect up to max_cycles
            for i, cycle in enumerate(cycle_generator):
                if i >= max_cycles:
                    break
                cycles.append(list(cycle))
            
            # Sort by cycle length (shortest first)
            cycles.sort(key=len)
            
        except Exception:
            # Return empty list if any error occurs
            pass
        
        return cycles
