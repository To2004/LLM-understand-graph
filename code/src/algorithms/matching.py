"""
Matching algorithms: Bipartite matching, maximum matching
"""

import networkx as nx
from typing import Dict, Set, Tuple, Optional


class MatchingAlgorithms:
    """
    Graph matching algorithms.
    
    TODO: Team Member Assignment - [ALGORITHMS TEAM - Matching]
    
    Priority: LOW
    Estimated Time: 1 week
    """
    
    @staticmethod
    def maximum_matching(graph: nx.Graph) -> Set[tuple]:
        """
        Compute maximum matching.
        
        Args:
            graph: NetworkX graph (undirected)
            
        Returns:
            Set of edges forming a maximum matching.
            A matching is a set of edges with no common vertices.
            
        Note:
            Uses maximum weight matching with unit weights for unweighted graphs.
            For weighted graphs, uses edge 'weight' attribute.
        """
        try:
            # Use max_weight_matching which works for both weighted and unweighted
            # For unweighted graphs, it computes maximum cardinality matching
            matching = nx.max_weight_matching(graph, maxcardinality=True)
            return matching
        except Exception as e:
            # Return empty set if any error
            return set()
    
    @staticmethod
    def bipartite_matching(
        graph: nx.Graph,
        top_nodes: Set[str]
    ) -> Dict[str, str]:
        """
        Compute maximum bipartite matching.
        
        Args:
            graph: NetworkX graph
            top_nodes: Set of nodes in one partition of the bipartite graph
            
        Returns:
            Dictionary mapping nodes from one side to the other in the matching
            
        Raises:
            ValueError: If graph is not bipartite
            
        Note:
            Uses Hopcroft-Karp algorithm for maximum cardinality bipartite matching
        """
        try:
            # First verify the graph is bipartite
            if not nx.is_bipartite(graph):
                raise ValueError("Graph is not bipartite")
            
            # Compute maximum bipartite matching
            matching = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
            
            return matching
        except nx.AmbiguousSolution:
            # If top_nodes doesn't properly partition the graph
            raise ValueError("top_nodes does not form a valid bipartite partition")
    
    @staticmethod
    def is_bipartite(graph: nx.Graph) -> Tuple[bool, Optional[Dict]]:
        """
        Check if graph is bipartite.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Tuple of (is_bipartite, coloring)
            If bipartite, coloring is a dict mapping nodes to 0 or 1 (the two partitions)
            If not bipartite, coloring is None
            
        Note:
            A graph is bipartite if its nodes can be colored with two colors
            such that no adjacent nodes have the same color.
            Handles disconnected graphs by checking each component.
        """
        try:
            # Check if graph is bipartite
            is_bip = nx.is_bipartite(graph)
            
            if is_bip:
                # Get the bipartite coloring (0 or 1 for each node)
                coloring = nx.bipartite.color(graph)
                return (True, coloring)
            else:
                return (False, None)
        except nx.NetworkXError:
            # Not bipartite
            return (False, None)
