"""
Graph parsing templates and prompts
"""

from typing import Dict, Any


class ParsingPrompts:
    """
    LLM prompts for graph parsing.
    
    TODO: Team Member Assignment - [PARSER TEAM - Prompts]
    
    TODO [PARSER-PROMPTS-001]:
        - Design effective parsing prompts
        - Add few-shot examples
        - Test prompt variations
        - Optimize for token efficiency
    """
    
    SYSTEM_MESSAGE = """You are an expert at extracting graph structures from natural language descriptions.
    
Your task is to parse textual graph descriptions and extract:
1. List of nodes (vertices)
2. List of edges with their connections
3. Graph properties (directed/undirected, weighted/unweighted)
4. Edge weights if present

Output your response as structured JSON."""

    NODE_EXTRACTION = """Extract all nodes from this graph description:

"{text}"

Return a JSON object with format:
{{
    "nodes": ["A", "B", "C", ...],
    "node_count": <number>
}}"""

    EDGE_EXTRACTION = """Given these nodes: {nodes}

Extract all edges from this description:
"{text}"

Return a JSON object with format:
{{
    "edges": [["A", "B"], ["B", "C"], ...],
    "directed": <true/false>,
    "weighted": <true/false>,
    "weights": {{"A-B": 2, "B-C": 3, ...}} // if weighted
}}"""

    FULL_PARSING = """Parse this complete graph description:

"{text}"

Extract and return JSON with format:
{{
    "nodes": ["A", "B", ...],
    "edges": [["A", "B"], ...],
    "directed": <true/false>,
    "weighted": <true/false>,
    "weights": {{"A-B": 2, ...}}, // if weighted
    "metadata": {{}} // any additional info
}}

Examples:

Input: "Nodes A, B, C with edges A--B, B--C"
Output: {{"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]], "directed": false, "weighted": false}}

Input: "Directed graph: A->B (weight 5), B->C (weight 3)"
Output: {{"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]], "directed": true, "weighted": true, "weights": {{"A-B": 5, "B-C": 3}}}}
"""
