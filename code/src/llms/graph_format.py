"""
Unified Graph Format: Standardized representation between Parser and Graph Builder
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator


class UnifiedGraphFormat(BaseModel):
    """
    Standardized graph representation format.
    
    This is the contract between Agent Parser and Graph Builder.
    Ensures consistent data flow through the pipeline.
    """
    nodes: List[str] = Field(..., description="List of node identifiers")
    edges: List[List[str]] = Field(..., description="List of edges as [source, target] pairs")
    directed: bool = Field(default=False, description="Whether graph is directed")
    weighted: bool = Field(default=False, description="Whether graph has edge weights")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Edge weights as 'A-B': weight or tuple keys"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional graph properties"
    )
    
    @validator('edges')
    def validate_edges(cls, v, values):
        """Ensure all edges are pairs"""
        for edge in v:
            if len(edge) != 2:
                raise ValueError(f"Edge must be a pair, got: {edge}")
        return v
    
    @validator('weights')
    def validate_weights(cls, v, values):
        """Ensure weights are provided if graph is weighted"""
        if values.get('weighted') and not v:
            # Allow empty weights dict, will be populated later
            return {}
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary for Graph Builder"""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_graph_structure(cls, graph_structure) -> 'UnifiedGraphFormat':
        """
        Convert from AgentParser's GraphStructure to UnifiedGraphFormat.
        
        Args:
            graph_structure: GraphStructure from parser
            
        Returns:
            UnifiedGraphFormat instance
        """
        # Convert edges from tuples to lists
        edges_list = [[str(e[0]), str(e[1])] for e in graph_structure.edges]
        
        # Convert weights dict keys from tuples to strings
        weights_dict = None
        if graph_structure.weighted and graph_structure.weights:
            weights_dict = {}
            for edge_tuple, weight in graph_structure.weights.items():
                # Convert tuple key to string "A-B"
                key = f"{edge_tuple[0]}-{edge_tuple[1]}"
                weights_dict[key] = weight
        
        return cls(
            nodes=[str(n) for n in graph_structure.nodes],
            edges=edges_list,
            directed=graph_structure.directed,
            weighted=graph_structure.weighted,
            weights=weights_dict,
            metadata=graph_structure.metadata
        )
    
    def to_graph_structure(self):
        """
        Convert back to GraphStructure format if needed.
        
        Returns:
            GraphStructure-like object
        """
        # Import here to avoid circular dependency
        from agents.parser import GraphStructure
        
        # Convert edges back to tuples
        edges_tuples = [tuple(e) for e in self.edges]
        
        # Convert weights back to tuple keys
        weights_dict = None
        if self.weighted and self.weights:
            weights_dict = {}
            for key, weight in self.weights.items():
                # Parse "A-B" back to tuple
                parts = key.split('-', 1)
                if len(parts) == 2:
                    weights_dict[(parts[0], parts[1])] = weight
        
        return GraphStructure(
            nodes=self.nodes,
            edges=edges_tuples,
            directed=self.directed,
            weighted=self.weighted,
            weights=weights_dict,
            metadata=self.metadata
        )
