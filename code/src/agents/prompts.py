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

    SCHEMA_NODE_EXTRACTION = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of node identifiers found in the graph"
            },
            "node_count": {
                "type": "integer",
                "description": "Total number of nodes"
            }
        },
        "required": ["nodes", "node_count"]
    }

    SCHEMA_EDGE_EXTRACTION = {
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "List of edges as [source, target] pairs"
            },
            "directed": {
                "type": "boolean",
                "description": "Whether the graph is directed"
            },
            "weighted": {
                "type": "boolean",
                "description": "Whether the graph has weighted edges"
            },
            "weights": {
                "type": "object",
                "description": "Edge weights as dictionary with 'source-target' keys",
                "additionalProperties": {"type": "number"}
            }
        },
        "required": ["edges", "directed", "weighted"]
    }

    SCHEMA_FULL_PARSING = {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of all node identifiers"
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "description": "List of edges as [source, target] pairs"
            },
            "directed": {
                "type": "boolean",
                "description": "Whether the graph is directed"
            },
            "weighted": {
                "type": "boolean",
                "description": "Whether edges have weights"
            },
            "weights": {
                "type": "object",
                "description": "Optional edge weights as 'source-target': weight",
                "additionalProperties": {"type": "number"}
            },
            "metadata": {
                "type": "object",
                "description": "Additional graph information",
                "additionalProperties": True
            }
        },
        "required": ["nodes", "edges", "directed", "weighted"]
    }

    @staticmethod
    def format_node_extraction_prompt(text: str) -> str:
        """Format the node extraction prompt with input text."""
        return ParsingPrompts.NODE_EXTRACTION.format(text=text)

    @staticmethod
    def format_edge_extraction_prompt(text: str, nodes: list) -> str:
        """Format the edge extraction prompt with text and known nodes."""
        return ParsingPrompts.EDGE_EXTRACTION.format(
            text=text,
            nodes=", ".join(nodes)
        )

    @staticmethod
    def format_full_parsing_prompt(text: str) -> str:
        """Format the full parsing prompt with input text."""
        return ParsingPrompts.FULL_PARSING.format(text=text)


class ChooserPrompts:
    """
    LLM prompts for task classification and algorithm selection.
    
    TODO: Team Member Assignment - [CHOOSER TEAM - Prompts]
    """
    
    SYSTEM_MESSAGE = """You are an expert at classifying graph reasoning tasks and selecting appropriate algorithms.

Your task is to:
1. Identify the type of graph problem from a natural language query
2. Extract relevant parameters (source/target nodes, constraints, etc.)
3. Select the most appropriate algorithm

Output your response as structured JSON."""

    TASK_CLASSIFICATION = """Classify this graph task query:

"{query}"

Determine the task type and extract parameters.

Return a JSON object with format:
{{
    "task_type": "connectivity|shortest_path|cycle_detection|topological_sort|maximum_flow|bipartite_matching|hamiltonian_path|gnn_message_passing",
    "parameters": {{
        "source": "...",  // if applicable
        "target": "...",  // if applicable
        // other task-specific parameters
    }},
    "confidence": 0.95  // your confidence in this classification (0.0-1.0)
}}

Examples:

Query: "Is node A connected to node D?"
Output: {{"task_type": "connectivity", "parameters": {{"source": "A", "target": "D"}}, "confidence": 0.98}}

Query: "Find the shortest path from A to Z"
Output: {{"task_type": "shortest_path", "parameters": {{"source": "A", "target": "Z"}}, "confidence": 0.99}}

Query: "Does this graph have a cycle?"
Output: {{"task_type": "cycle_detection", "parameters": {{}}, "confidence": 0.97}}

Query: "What is the maximum flow from S to T?"
Output: {{"task_type": "maximum_flow", "parameters": {{"source": "S", "sink": "T"}}, "confidence": 0.95}}
"""

    ALGORITHM_SELECTION = """Given this task classification and graph properties, select the best algorithm:

Task: {task_type}
Parameters: {parameters}
Graph Properties:
- Directed: {directed}
- Weighted: {weighted}
- Number of nodes: {num_nodes}
- Number of edges: {num_edges}

Return a JSON object with format:
{{
    "algorithm_name": "...",  // specific algorithm to use
    "reasoning": "...",       // why this algorithm is appropriate
    "confidence": 0.9
}}

Available algorithms by task:
- connectivity: is_connected, find_all_paths, connected_components
- shortest_path: dijkstra (no negative weights), bellman_ford (allows negative weights), all_pairs_shortest_path
- cycle_detection: has_cycle, find_all_cycles
- topological_sort: topological_sort (requires DAG)
- maximum_flow: maximum_flow, minimum_cut
- bipartite_matching: maximum_matching, bipartite_matching, is_bipartite
- hamiltonian_path: NOT IMPLEMENTED (use has_cycle as fallback for Hamiltonian cycles)
- gnn_message_passing: NOT IMPLEMENTED (use find_all_paths as fallback)

IMPORTANT: If an algorithm is not available, return null for algorithm_name.
"""

    PARAMETER_EXTRACTION = """Extract algorithm parameters from this query:

"{query}"

Known nodes in graph: {nodes}

Return a JSON object with format:
{{
    "source": "...",      // source node (if applicable)
    "target": "...",      // target/destination node (if applicable)
    "sink": "...",        // sink node for flow problems (if applicable)
    "constraints": {{}},  // any additional constraints
    "found_parameters": ["source", "target"]  // list of parameters found
}}

If a parameter cannot be determined, omit it from the response.
"""

    SCHEMA_TASK_CLASSIFICATION = {
        "type": "object",
        "properties": {
            "task_type": {
                "type": "string",
                "enum": [
                    "connectivity",
                    "shortest_path",
                    "cycle_detection",
                    "topological_sort",
                    "maximum_flow",
                    "bipartite_matching",
                    "hamiltonian_path",
                    "gnn_message_passing"
                ],
                "description": "The type of graph reasoning task"
            },
            "parameters": {
                "type": "object",
                "description": "Task-specific parameters extracted from query",
                "additionalProperties": True
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score for classification"
            }
        },
        "required": ["task_type", "parameters", "confidence"]
    }

    SCHEMA_ALGORITHM_SELECTION = {
        "type": "object",
        "properties": {
            "algorithm_name": {
                "type": "string",
                "description": "Name of the selected algorithm"
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation for algorithm selection"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in algorithm choice"
            }
        },
        "required": ["algorithm_name", "reasoning", "confidence"]
    }

    SCHEMA_PARAMETER_EXTRACTION = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source node identifier"
            },
            "target": {
                "type": "string",
                "description": "Target/destination node identifier"
            },
            "sink": {
                "type": "string",
                "description": "Sink node for flow problems"
            },
            "constraints": {
                "type": "object",
                "description": "Additional constraints",
                "additionalProperties": True
            },
            "found_parameters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of parameters successfully extracted"
            }
        }
    }

    @staticmethod
    def format_task_classification_prompt(query: str) -> str:
        """Format the task classification prompt."""
        return ChooserPrompts.TASK_CLASSIFICATION.format(query=query)

    @staticmethod
    def format_algorithm_selection_prompt(
        task_type: str,
        parameters: dict,
        directed: bool,
        weighted: bool,
        num_nodes: int,
        num_edges: int
    ) -> str:
        """Format the algorithm selection prompt."""
        return ChooserPrompts.ALGORITHM_SELECTION.format(
            task_type=task_type,
            parameters=str(parameters),
            directed=directed,
            weighted=weighted,
            num_nodes=num_nodes,
            num_edges=num_edges
        )

    @staticmethod
    def format_parameter_extraction_prompt(query: str, nodes: list) -> str:
        """Format the parameter extraction prompt."""
        return ChooserPrompts.PARAMETER_EXTRACTION.format(
            query=query,
            nodes=", ".join(nodes)
        )
