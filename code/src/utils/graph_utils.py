"""
Graph utility functions for conversion and validation
"""

import networkx as nx
from typing import Dict, List, Any


class GraphUtils:
    """
    Utility functions for graph operations.
    
    Provides conversion between dictionary representations and NetworkX graphs.
    """
    
    @staticmethod
    def dict_to_networkx(graph_dict: Dict[str, Any]) -> nx.Graph:
        """
        Convert dictionary representation to NetworkX graph.
        
        Expected input format (Unified Intermediate Format):
        {
            "nodes": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"]],
            "directed": false,
            "weighted": false,
            "weights": {}  # Optional: {"A-B": 1.5, "B-C": 2.0}
        }
        
        Args:
            graph_dict: Dictionary with graph data
            
        Returns:
            NetworkX Graph or DiGraph object
        """
        # Choose graph type based on directed flag
        directed = graph_dict.get('directed', False)
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        nodes = graph_dict.get('nodes', [])
        G.add_nodes_from(nodes)
        
        # Add edges
        edges = graph_dict.get('edges', [])
        weighted = graph_dict.get('weighted', False)
        weights = graph_dict.get('weights', {})
        
        if weighted and weights:
            # Add edges with weights
            for edge in edges:
                if len(edge) == 2:
                    source, target = edge[0], edge[1]
                    # Try to find weight for this edge
                    # Try to find weight for this edge
                    weight_key = f"{source}-{target}"
                    
                    if weight_key in weights:
                        weight = weights[weight_key]
                    elif (source, target) in weights:
                        weight = weights[(source, target)]
                    else:
                        # Try reverse key for undirected graphs or if parser flipped them
                        weight_key_rev = f"{target}-{source}"
                        if weight_key_rev in weights:
                            weight = weights[weight_key_rev]
                        elif (target, source) in weights:
                            weight = weights[(target, source)]
                        else:
                            # Warning if weighted but weight not found
                            if weighted:
                                print(f"Warning: Weight not found for edge {source}-{target}, using default 1.0")
                                # DEBUG: Print details for the first few failures to avoid spam
                                if not hasattr(GraphUtils, '_debug_count'):
                                    GraphUtils._debug_count = 0
                                if GraphUtils._debug_count < 3:
                                    print(f"DEBUG: Lookup Key: '{weight_key}'")
                                    print(f"DEBUG: Source: '{source}' (type: {type(source)}), Target: '{target}' (type: {type(target)})")
                                    print(f"DEBUG: Available keys (sample): {list(weights.keys())[:5]}")
                                    GraphUtils._debug_count += 1
                            weight = 1.0
                            
                    G.add_edge(source, target, weight=weight)
        else:
            # Add edges without weights
            for edge in edges:
                if len(edge) == 2:
                    G.add_edge(edge[0], edge[1])
        
        # Add metadata as graph attributes
        metadata = graph_dict.get('metadata', {})
        if metadata:
            for key, value in metadata.items():
                G.graph[key] = value
        
        return G
    
    @staticmethod
    def networkx_to_dict(graph: nx.Graph) -> Dict[str, Any]:
        """
        Convert NetworkX graph to dictionary.
        
        Returns unified intermediate format for serialization.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary in unified format
        """
        # Extract nodes
        nodes = list(graph.nodes())
        
        # Extract edges
        edges = [[str(u), str(v)] for u, v in graph.edges()]
        
        # Determine if directed
        directed = isinstance(graph, nx.DiGraph)
        
        # Check if weighted (look for 'weight' attribute on edges)
        weighted = False
        weights = {}
        
        for u, v, data in graph.edges(data=True):
            if 'weight' in data:
                weighted = True
                weight_key = f"{u}-{v}"
                weights[weight_key] = data['weight']
        
        # Extract graph metadata
        metadata = dict(graph.graph) if graph.graph else None
        
        result = {
            'nodes': [str(n) for n in nodes],
            'edges': edges,
            'directed': directed,
            'weighted': weighted
        }
        
        if weighted:
            result['weights'] = weights
        
        if metadata:
            result['metadata'] = metadata
        
        return result
    
    @staticmethod
    def validate_graph(graph: nx.Graph) -> bool:
        """
        Validate graph structure.
        
        Checks:
        - No self-loops (unless explicitly allowed)
        - All edges reference valid nodes
        - Weights are valid numbers (if weighted)
        - Warns about isolated nodes
        
        Args:
            graph: NetworkX graph to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            print(f"Warning: Graph contains self-loops: {self_loops}")
        
        # Check that all edges reference valid nodes
        nodes = set(graph.nodes())
        for u, v in graph.edges():
            if u not in nodes or v not in nodes:
                raise ValueError(f"Edge ({u}, {v}) references non-existent node")
        
        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            print(f"Warning: Graph contains isolated nodes: {isolated}")
        
        # Check weights if present
        for u, v, data in graph.edges(data=True):
            if 'weight' in data:
                weight = data['weight']
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Edge ({u}, {v}) has invalid weight: {weight}")
                if weight < 0:
                    print(f"Warning: Edge ({u}, {v}) has negative weight: {weight}")
        
        return True
    
    @staticmethod
    def serialize_for_llm(graph: nx.Graph, format: str = "incident") -> str:
        """
        Serialize graph for LLM consumption.
        
        Formats:
        - "incident": Incident-based format (A: [B, C])
        - "adjacency": Adjacency list format
        - "edge_list": Simple edge list format
        
        Args:
            graph: NetworkX graph
            format: Output format
            
        Returns:
            String representation optimized for LLM
        """
        if format == "edge_list":
            # Simple edge list: A-B, B-C, C-D
            edges = [f"{u}-{v}" for u, v in graph.edges()]
            return ", ".join(edges)
        
        elif format == "adjacency":
            # Adjacency list: A: B, C; B: A, C, D
            adj_list = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    adj_list.append(f"{node}: {', '.join(str(n) for n in neighbors)}")
            return "; ".join(adj_list)
        
        else:  # incident format (default)
            # Incident format: A: [B, C]
            incident_list = []
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                incident_list.append(f"{node}: [{', '.join(str(n) for n in neighbors)}]")
            return "\n".join(incident_list)

