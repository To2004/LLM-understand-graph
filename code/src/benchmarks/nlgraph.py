"""
NLGraph benchmark dataset loader and processor
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import networkx as nx


class NLGraphBenchmark:
    """
    NLGraph benchmark dataset handler.
    
    Supports loading and filtering samples from the NLGraph dataset across
    multiple task types: shortest_path, connectivity, cycle, flow, matching,
    hamilton, topology, and GNN.
    """
    
    TASK_TYPES = [
        'shortest_path', 'connectivity', 'cycle', 'flow',
        'matching', 'hamilton', 'topology', 'GNN'
    ]
    
    def __init__(self, data_path: Path):
        """
        Initialize NLGraph benchmark.
        
        Args:
            data_path: Path to NLGraph dataset (e.g., 'data/NLGraph/NLGraph')
        """
        self.data_path = Path(data_path)
        self.samples = []
        self.samples_by_task = {}
        
        # Validate path exists
        if not self.data_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.data_path}")
    
    def load_dataset(self, split: str = 'test', tasks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load complete NLGraph dataset.
        
        Args:
            split: Dataset split to load ('train', 'test', or 'main')
            tasks: List of task types to load (defaults to all)
            
        Returns:
            List of samples with questions, answers, and metadata
        """
        if tasks is None:
            tasks = self.TASK_TYPES
        
        self.samples = []
        self.samples_by_task = {}
        
        for task in tasks:
            task_path = self.data_path / task
            if not task_path.exists():
                print(f"Warning: Task path not found: {task_path}")
                continue
            
            # Load JSON file with questions and answers
            json_file = task_path / f"{split}.json"
            if json_file.exists():
                task_samples = self._load_task_json(task, json_file)
                self.samples.extend(task_samples)
                self.samples_by_task[task] = task_samples
            else:
                print(f"Warning: JSON file not found: {json_file}")
        
        return self.samples
    
    def _load_task_json(self, task: str, json_file: Path) -> List[Dict[str, Any]]:
        """
        Load samples from a task JSON file.
        
        Args:
            task: Task type name
            json_file: Path to JSON file
            
        Returns:
            List of sample dictionaries
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            for key, value in data.items():
                sample = {
                    'id': key,
                    'task': task,
                    'question': value.get('question', ''),
                    'answer': value.get('answer', ''),
                    'difficulty': value.get('difficulty', 'unknown')
                }
                
                # Parse graph information from question if available
                if task in ['shortest_path', 'connectivity', 'cycle']:
                    sample['graph'] = self._parse_graph_from_question(sample['question'], task)
                
                samples.append(sample)
            
            return samples
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return []
    
    def filter_by_task(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Filter samples by task type.
        
        Args:
            task_type: Task type to filter by (e.g., 'shortest_path', 'connectivity')
            
        Returns:
            Filtered list of samples for the specified task
        """
        return [s for s in self.samples if s.get('task') == task_type]
    
    def filter_by_complexity(
        self, 
        min_nodes: int = 0,
        max_nodes: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Filter samples by graph complexity.
        
        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            
        Returns:
            Filtered list of samples within the node count range
        """
        filtered = []
        for sample in self.samples:
            graph = sample.get('graph')
            if graph and isinstance(graph, nx.Graph):
                num_nodes = graph.number_of_nodes()
                if min_nodes <= num_nodes <= max_nodes:
                    filtered.append(sample)
            elif 'difficulty' in sample:
                # Fallback: use difficulty as proxy for complexity
                filtered.append(sample)
        
        return filtered
    
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
        Get single sample with metadata.
        
        Args:
            index: Index of sample to retrieve
            
        Returns:
            Sample dictionary with question, answer, task type, and metadata
        """
        if 0 <= index < len(self.samples):
            return self.samples[index]
        raise IndexError(f"Sample index {index} out of range [0, {len(self.samples)})")
    
    def _parse_graph_from_question(self, question: str, task: str) -> Optional[nx.Graph]:
        """
        Parse graph structure from natural language question.
        
        Args:
            question: Question text containing graph description
            task: Task type for context
            
        Returns:
            NetworkX graph or None if parsing fails
        """
        try:
            lines = question.split('\n')
            G = nx.Graph()
            
            # Find the line that specifies nodes
            for line in lines:
                if 'nodes are numbered from' in line.lower():
                    # Extract node range
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'to':
                            try:
                                max_node = int(parts[i+1].rstrip(',.'))
                                G.add_nodes_from(range(max_node + 1))
                                break
                            except:
                                pass
                
                # Parse edges
                if 'edge between' in line.lower() or 'road between' in line.lower():
                    # Extract edge information
                    parts = line.split()
                    try:
                        # Find node numbers
                        nodes = []
                        weight = None
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                nodes.append(int(part))
                            elif 'weight' in parts[i-1:i+1] or 'length' in parts[i-1:i+1]:
                                if part.isdigit():
                                    weight = int(part)
                        
                        if len(nodes) >= 2:
                            if weight is not None:
                                G.add_edge(nodes[0], nodes[1], weight=weight)
                            else:
                                G.add_edge(nodes[0], nodes[1])
                    except:
                        pass
            
            if G.number_of_nodes() > 0:
                return G
        except Exception as e:
            pass
        
        return None
    
    def load_graph_file(self, task: str, difficulty: str, graph_id: int) -> Optional[nx.Graph]:
        """
        Load graph from file in the graph/ directory.
        
        Args:
            task: Task type
            difficulty: Difficulty level
            graph_id: Graph file ID
            
        Returns:
            NetworkX graph or None if file not found
        """
        graph_file = self.data_path / task / 'graph' / difficulty / 'standard' / f'graph{graph_id}.txt'
        
        if not graph_file.exists():
            return None
        
        try:
            with open(graph_file, 'r') as f:
                first_line = f.readline().split()
                n, m = int(first_line[0]), int(first_line[1])
                
                G = nx.Graph()
                G.add_nodes_from(range(n))
                
                # Read edges
                for _ in range(m):
                    parts = f.readline().split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        weight = int(parts[2]) if len(parts) > 2 else 1
                        G.add_edge(u, v, weight=weight)
                
                return G
        except Exception as e:
            print(f"Error loading graph file {graph_file}: {e}")
            return None
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded dataset.
        
        Returns:
            Dictionary with task counts, difficulty distributions, etc.
        """
        stats = {
            'total_samples': len(self.samples),
            'tasks': {},
            'difficulties': {}
        }
        
        for sample in self.samples:
            task = sample.get('task', 'unknown')
            difficulty = sample.get('difficulty', 'unknown')
            
            stats['tasks'][task] = stats['tasks'].get(task, 0) + 1
            stats['difficulties'][difficulty] = stats['difficulties'].get(difficulty, 0) + 1
        
        return stats
