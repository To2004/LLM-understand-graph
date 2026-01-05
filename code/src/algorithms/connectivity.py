"""
Connectivity algorithms: BFS, DFS, connected components
"""

import networkx as nx
from typing import List, Any, Optional


class ConnectivityAlgorithms:
    """
    Connectivity-related graph algorithms.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM - Connectivity]
    
    Priority: MEDIUM
    Estimated Time: 1 week
    """
    
    @staticmethod
    def is_connected(
        graph: nx.Graph, 
        source: str, 
        target: str
    ) -> tuple[bool, Optional[List[str]]]:
        """
        Check if two nodes are connected and return path if exists.
        
        Uses BFS-based path finding from NetworkX.
        
        Args:
            graph: NetworkX graph
            source: Source node
            target: Target node
            
        Returns:
            Tuple of (is_connected, path), where path is None if not connected
        """
        try:
            # Use NetworkX shortest_path for BFS-based path finding
            path = nx.shortest_path(graph, source=source, target=target)
            return (True, path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return (False, None)
    
    @staticmethod
    def find_all_paths(
        graph: nx.Graph,
        source: str,
        target: str,
        max_paths: int = 10
    ) -> List[List[str]]:
        """
        Find all simple paths between two nodes.
        
        Args:
            graph: NetworkX graph
            source: Source node
            target: Target node
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths (each path is a list of nodes), sorted by length
        """
        try:
            # Find all simple paths (no repeated nodes)
            all_paths = nx.all_simple_paths(graph, source=source, target=target)
            
            # Convert generator to list and limit number of paths
            paths = []
            for i, path in enumerate(all_paths):
                if i >= max_paths:
                    break
                paths.append(path)
            
            # Sort paths by length (shortest first)
            paths.sort(key=len)
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    @staticmethod
    def connected_components(graph: nx.Graph) -> List[List[str]]:
        """
        Find all connected components.
        
        Args:
            graph: NetworkX graph (directed or undirected)
            
        Returns:
            List of components, where each component is a list of nodes.
            Components are sorted by size (largest first).
        """
        # Handle directed vs undirected graphs
        if graph.is_directed():
            # For directed graphs, use weakly connected components
            components = nx.weakly_connected_components(graph)
        else:
            # For undirected graphs, use connected components
            components = nx.connected_components(graph)
        
        # Convert to list of lists and sort by size (largest first)
        component_list = [list(comp) for comp in components]
        component_list.sort(key=len, reverse=True)
        
        return component_list
