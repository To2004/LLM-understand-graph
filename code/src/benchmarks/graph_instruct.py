"""
GraphInstruct benchmark dataset loader
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import networkx as nx


class GraphInstructBenchmark:
    """
    GraphInstruct dataset handler for diverse edge cases.
    
    Supports loading and filtering instruction-following samples with
    various graph edge cases including disconnected components, self-loops,
    and multi-edges.
    """
    
    def __init__(self, data_path: Path):
        """
        Initialize GraphInstruct benchmark.
        
        Args:
            data_path: Path to GraphInstruct dataset directory
        """
        self.data_path = Path(data_path)
        self.samples = []
        self.samples_by_difficulty = {}
        
        # Validate path exists
        if not self.data_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.data_path}")
    
    def load_dataset(self, split: str = 'test') -> List[Dict[str, Any]]:
        """
        Load GraphInstruct dataset.
        
        Args:
            split: Dataset split to load ('train', 'test', or 'validation')
            
        Returns:
            List of samples with instructions, graphs, and metadata
        """
        self.samples = []
        self.samples_by_difficulty = {'easy': [], 'medium': [], 'hard': []}
        
        # Look for JSON files in the dataset directory
        json_file = self.data_path / f"{split}.json"
        
        if not json_file.exists():
            # Try alternative naming conventions
            json_file = self.data_path / f"graphinstruct_{split}.json"
        
        if json_file.exists():
            self.samples = self._load_json_file(json_file)
        else:
            print(f"Warning: Dataset file not found: {json_file}")
            # Try loading from subdirectories
            for difficulty in ['easy', 'medium', 'hard']:
                difficulty_file = self.data_path / difficulty / f"{split}.json"
                if difficulty_file.exists():
                    samples = self._load_json_file(difficulty_file)
                    for sample in samples:
                        sample['difficulty'] = difficulty
                    self.samples.extend(samples)
                    self.samples_by_difficulty[difficulty] = samples
        
        return self.samples
    
    def _load_json_file(self, json_file: Path) -> List[Dict[str, Any]]:
        """
        Load samples from a JSON file.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            List of sample dictionaries
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of samples
                for item in data:
                    sample = self._parse_sample(item)
                    if sample:
                        samples.append(sample)
            elif isinstance(data, dict):
                # Dictionary of samples
                for key, value in data.items():
                    sample = self._parse_sample(value, sample_id=key)
                    if sample:
                        samples.append(sample)
            
            return samples
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return []
    
    def _parse_sample(self, item: Dict[str, Any], sample_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parse a single sample from the dataset.
        
        Args:
            item: Sample data dictionary
            sample_id: Optional sample identifier
            
        Returns:
            Parsed sample dictionary or None if parsing fails
        """
        try:
            sample = {
                'id': sample_id or item.get('id', 'unknown'),
                'instruction': item.get('instruction', item.get('question', '')),
                'answer': item.get('answer', item.get('output', '')),
                'difficulty': item.get('difficulty', 'unknown'),
                'graph_type': item.get('graph_type', 'unknown')
            }
            
            # Parse graph if available
            if 'graph' in item:
                sample['graph'] = self._parse_graph(item['graph'])
            elif 'edges' in item:
                sample['graph'] = self._build_graph_from_edges(
                    item['edges'],
                    item.get('nodes', [])
                )
            
            # Detect edge cases
            if 'graph' in sample and sample['graph']:
                sample['edge_cases'] = self._detect_edge_cases(sample['graph'])
            
            return sample
        except Exception as e:
            print(f"Error parsing sample: {e}")
            return None
    
    def _parse_graph(self, graph_data: Any) -> Optional[nx.Graph]:
        """
        Parse graph from various formats.
        
        Args:
            graph_data: Graph data in various formats (dict, list, string)
            
        Returns:
            NetworkX graph or None if parsing fails
        """
        try:
            G = nx.Graph()
            
            if isinstance(graph_data, dict):
                # Dictionary format with nodes and edges
                if 'nodes' in graph_data:
                    G.add_nodes_from(graph_data['nodes'])
                if 'edges' in graph_data:
                    G.add_edges_from(graph_data['edges'])
            elif isinstance(graph_data, list):
                # List of edges
                G.add_edges_from(graph_data)
            elif isinstance(graph_data, str):
                # String representation (e.g., edge list)
                edges = eval(graph_data)  # Careful: only use with trusted data
                G.add_edges_from(edges)
            
            return G if G.number_of_nodes() > 0 else None
        except Exception as e:
            print(f"Error parsing graph: {e}")
            return None
    
    def _build_graph_from_edges(self, edges: List, nodes: Optional[List] = None) -> nx.Graph:
        """
        Build NetworkX graph from edge list.
        
        Args:
            edges: List of edges (tuples or lists)
            nodes: Optional list of nodes
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        if nodes:
            G.add_nodes_from(nodes)
        
        G.add_edges_from(edges)
        
        return G
    
    def _detect_edge_cases(self, graph: nx.Graph) -> List[str]:
        """
        Detect edge cases in a graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            List of detected edge case types
        """
        edge_cases = []
        
        # Check for disconnected components
        if not nx.is_connected(graph):
            edge_cases.append('disconnected')
        
        # Check for self-loops
        if graph.number_of_selfloops() > 0:
            edge_cases.append('self_loops')
        
        # Check for isolated nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            edge_cases.append('isolated_nodes')
        
        # Check for multi-edges (if using MultiGraph)
        # For regular Graph, this won't apply, but we can check degree
        max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0
        if max_degree > graph.number_of_nodes() - 1:
            edge_cases.append('high_degree')
        
        return edge_cases
    
    def get_edge_cases(self) -> List[Dict[str, Any]]:
        """
        Get samples with edge cases.
        
        Returns:
            Filtered list of samples containing edge cases
        """
        edge_case_samples = []
        
        for sample in self.samples:
            if 'edge_cases' in sample and sample['edge_cases']:
                edge_case_samples.append(sample)
        
        return edge_case_samples
    
    def filter_by_edge_case_type(self, edge_case_type: str) -> List[Dict[str, Any]]:
        """
        Filter samples by specific edge case type.
        
        Args:
            edge_case_type: Type of edge case ('disconnected', 'self_loops', etc.)
            
        Returns:
            Filtered list of samples with specified edge case
        """
        return [
            s for s in self.samples
            if 'edge_cases' in s and edge_case_type in s['edge_cases']
        ]
    
    def filter_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Filter samples by difficulty level.
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            Filtered list of samples with specified difficulty
        """
        return [s for s in self.samples if s.get('difficulty') == difficulty]
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Get single sample by index.
        
        Args:
            index: Index of sample to retrieve
            
        Returns:
            Sample dictionary
        """
        if 0 <= index < len(self.samples):
            return self.samples[index]
        raise IndexError(f"Sample index {index} out of range [0, {len(self.samples)})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded dataset.
        
        Returns:
            Dictionary with sample counts, edge case distributions, etc.
        """
        stats = {
            'total_samples': len(self.samples),
            'difficulties': {},
            'edge_cases': {},
            'graph_types': {}
        }
        
        for sample in self.samples:
            # Count difficulties
            difficulty = sample.get('difficulty', 'unknown')
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
            
            # Count edge cases
            for edge_case in sample.get('edge_cases', []):
                stats['edge_cases'][edge_case] = stats['edge_cases'].get(edge_case, 0) + 1
            
            # Count graph types
            graph_type = sample.get('graph_type', 'unknown')
            stats['graph_types'][graph_type] = stats['graph_types'].get(graph_type, 0) + 1
        
        return stats
