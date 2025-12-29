"""
Shortest path algorithms: Dijkstra, Bellman-Ford, A*
"""

import networkx as nx
from typing import List, Dict, Any, Optional


class ShortestPathAlgorithms:
    """
    Shortest path graph algorithms.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM - Shortest Path]
    
    Priority: MEDIUM
    Estimated Time: 1 week
    """
    
    @staticmethod
    def dijkstra(
        graph: nx.Graph,
        source: str,
        target: str,
        weight: str = 'weight'
    ) -> tuple[List[str], float]:
        """
        Compute shortest path using Dijkstra's algorithm.
        
        Args:
            graph: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge attribute to use as weight (default: 'weight')
            
        Returns:
            Tuple of (path, total_distance)
            
        Raises:
            ValueError: If graph contains negative weights
            NetworkXNoPath: If no path exists
        """
        # Check for negative weights
        for u, v, data in graph.edges(data=True):
            if weight in data and data[weight] < 0:
                raise ValueError("Dijkstra's algorithm does not support negative weights")
        
        try:
            # Compute shortest path
            path = nx.dijkstra_path(graph, source=source, target=target, weight=weight)
            # Compute path length
            length = nx.dijkstra_path_length(graph, source=source, target=target, weight=weight)
            return (path, length)
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found: {e}")
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(f"No path between {source} and {target}")
    
    @staticmethod
    def bellman_ford(
        graph: nx.Graph,
        source: str,
        target: str,
        weight: str = 'weight'
    ) -> tuple[List[str], float]:
        """
        Compute shortest path allowing negative weights.
        
        Args:
            graph: NetworkX graph
            source: Source node
            target: Target node
            weight: Edge attribute to use as weight (default: 'weight')
            
        Returns:
            Tuple of (path, total_distance)
            
        Raises:
            NetworkXError: If graph contains negative cycle
            NetworkXNoPath: If no path exists
        """
        try:
            # Compute shortest path using Bellman-Ford
            path = nx.bellman_ford_path(graph, source=source, target=target, weight=weight)
            # Compute path length
            length = nx.bellman_ford_path_length(graph, source=source, target=target, weight=weight)
            return (path, length)
        except nx.NetworkXError as e:
            # This includes negative cycle detection
            raise nx.NetworkXError(f"Bellman-Ford error: {e}")
        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found: {e}")
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(f"No path between {source} and {target}")
    
    @staticmethod
    def all_pairs_shortest_path(
        graph: nx.Graph,
        weight: str = 'weight'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute shortest paths between all node pairs.
        
        Args:
            graph: NetworkX graph
            weight: Edge attribute to use as weight (default: 'weight')
            
        Returns:
            Dictionary mapping source -> target -> {'path': [...], 'length': float}
            
        Note:
            Uses Floyd-Warshall for dense graphs, Johnson's for sparse graphs.
        """
        result = {}
        
        # Compute all pairs shortest path lengths and paths
        try:
            # Get path lengths using appropriate algorithm
            lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))
            paths = dict(nx.all_pairs_dijkstra_path(graph, weight=weight))
            
            # Combine into unified structure
            for source in paths:
                result[source] = {}
                for target in paths[source]:
                    result[source][target] = {
                        'path': paths[source][target],
                        'length': lengths[source].get(target, float('inf'))
                    }
                    
        except Exception as e:
            # Fallback to basic approach if error
            for source in graph.nodes():
                result[source] = {}
                for target in graph.nodes():
                    try:
                        path = nx.shortest_path(graph, source=source, target=target, weight=weight)
                        length = nx.shortest_path_length(graph, source=source, target=target, weight=weight)
                        result[source][target] = {'path': path, 'length': length}
                    except nx.NetworkXNoPath:
                        result[source][target] = {'path': None, 'length': float('inf')}
        
        return result
