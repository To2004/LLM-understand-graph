"""
Agent Parser: Extracts graph structure from natural language descriptions

This module converts natural-language graph descriptions into structured 
representations (nodes, edges, directions, weights) following GraphNL templates.
"""

from typing import Dict, List, Any, Optional
import json
from pydantic import BaseModel
from agents.prompts import ParsingPrompts

class GraphStructure(BaseModel):
    """Structured representation of a graph extracted from natural language"""
    nodes: List[str]
    edges: List[tuple]
    directed: bool
    weighted: bool
    weights: Optional[Dict[tuple, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentParser:
    """
    Parses natural language graph descriptions into structured representations.
    
    TODO: Team Member Assignment - [PARSER TEAM]
    
    Priority: HIGH
    Estimated Time: 2-3 weeks
    """
    
    def __init__(self, llm_client, template_type: str = "graphnl"):
        """
        Initialize the parser with an LLM client.
        
        Args:
            llm_client: The LLM client for parsing assistance
            template_type: Type of template to use (graphnl, custom, etc.)
        """
        self.llm_client = llm_client
        self.template_type = template_type
    
    def parse(self, natural_language_input: str) -> GraphStructure:
        """
        Parse natural language description into structured graph.
        
        Args:
            natural_language_input: Natural language description of graph
            
        Returns:
            GraphStructure object with extracted nodes, edges, etc.
        """
        # Extract nodes first
        nodes = self._extract_nodes(natural_language_input)
        
        # Extract edges given the nodes
        edges_data = self._extract_edges(natural_language_input, nodes)
        
        # Build the graph structure
        graph_structure = GraphStructure(
            nodes=nodes,
            edges=edges_data["edges"],
            directed=edges_data["directed"],
            weighted=edges_data["weighted"],
            weights=edges_data.get("weights"),
            metadata={"source": "llm_parsed"}
        )
        
        # Validate the structure
        if not self._validate_structure(graph_structure):
            raise ValueError("Parsed graph structure is invalid")
        
        return graph_structure
    
    def _extract_nodes(self, text: str) -> List[str]:
        """
        Extract node identifiers from text using LLM.
        """
        prompt = ParsingPrompts.format_node_extraction_prompt(text)
        
        response = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ParsingPrompts.SCHEMA_NODE_EXTRACTION,
            system_message=ParsingPrompts.SYSTEM_MESSAGE
        )
        
        # Parse LLM response text to dict
        parsed = self._parse_json_response(response)
        
        return parsed["nodes"]
    
    def _extract_edges(self, text: str, nodes: List[str]) -> Dict[str, Any]:
        """
        Extract edges from text given known nodes.
        
        Returns:
            Dictionary with edges (as tuples), directed flag, weighted flag, and optional weights
        """
        prompt = ParsingPrompts.format_edge_extraction_prompt(text, nodes)
        
        response = self.llm_client.generate_structured(
            prompt=prompt,
            schema=ParsingPrompts.SCHEMA_EDGE_EXTRACTION,
            system_message=ParsingPrompts.SYSTEM_MESSAGE
        )
        
        # Parse and convert edges format
        parsed = self._parse_json_response(response)
        return self._convert_edges_format(parsed)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON string from LLM response.
        Simple JSON parsing only - no conversion.
        
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
    
    def _convert_edges_format(self, edges_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert edge data from LLM format to GraphStructure format.
        Responsible for all data type conversions.
        
        Args:
            edges_data: Raw edges data with arrays
            
        Returns:
            Converted edges data with tuples and proper weight keys
        """
        if "edges" not in edges_data:
            raise ValueError("Missing 'edges' key in edge extraction response")
        
        converted = edges_data.copy()
        
        # Convert edge arrays to tuples for GraphStructure
        converted["edges"] = [tuple(edge) for edge in edges_data["edges"]]
        
        # Convert weight keys from string format to tuple format if present
        if edges_data.get("weighted") and edges_data.get("weights"):
            converted["weights"] = self._convert_weight_keys(
                edges_data["weights"],
                converted["edges"]
            )
        
        return converted
    
    def _convert_weight_keys(
        self, 
        weights: Dict[str, float], 
        edges: List[tuple]
    ) -> Dict[tuple, float]:
        """
        Convert weight dictionary keys from string format to tuple format.
        
        Args:
            weights: Dictionary with string keys like "A-B"
            edges: List of edge tuples to match against
            
        Returns:
            Dictionary with tuple keys
            
        Raises:
            ValueError: If weighted edges are missing weight values
        """
        converted_weights = {}
        missing_weights = []
        
        for edge in edges:
            # Try standard format "A-B"
            key = f"{edge[0]}-{edge[1]}"
            if key in weights:
                converted_weights[edge] = weights[key]
            else:
                # Try reverse for undirected graphs
                reverse_key = f"{edge[1]}-{edge[0]}"
                if reverse_key in weights:
                    converted_weights[edge] = weights[reverse_key]
                else:
                    missing_weights.append(edge)
        
        # Warn or fail if weights are missing
        if missing_weights:
            raise ValueError(
                f"Weighted graph missing weights for edges: {missing_weights}"
            )
        
        return converted_weights
    
    def _validate_structure(self, structure: GraphStructure) -> bool:
        """
        Validate that parsed structure is consistent.
        """
        # Check that we have nodes
        if not structure.nodes:
            return False
        
        # Check all edges reference valid nodes
        node_set = set(structure.nodes)
        for edge in structure.edges:
            if len(edge) != 2:
                return False
            if edge[0] not in node_set or edge[1] not in node_set:
                return False
        
        # Check weight consistency - if weighted, all edges must have weights
        if structure.weighted:
            if not structure.weights:
                return False
            
            for edge in structure.edges:
                # Weights dict uses tuple keys
                if edge not in structure.weights:
                    # Try reverse for undirected graphs
                    edge_reversed = (edge[1], edge[0])
                    if not structure.directed and edge_reversed not in structure.weights:
                        return False
                    elif structure.directed:
                        return False
        
        return True
    
    def parse_with_repair(
        self, 
        natural_language_input: str, 
        max_retries: int = 3
    ) -> GraphStructure:
        """
        Parse with automatic retry on failure.
        """
        errors = []
        
        for attempt in range(max_retries):
            try:
                return self.parse(natural_language_input)
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                errors.append(f"Attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
                continue
        
        error_summary = "\n".join(errors)
        raise ValueError(
            f"Failed to parse after {max_retries} attempts:\n{error_summary}"
        )
