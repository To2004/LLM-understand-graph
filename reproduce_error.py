
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'code'))

import networkx as nx
from src.algorithms.executor import AlgorithmExecutor

# Mocking the scenario
# The log said: "NetworkX graph created: 9 nodes, 12 edges"
# And params: {'source': '8', 'target': '9'}
# This strongly suggests nodes are 0-8 (total 9), and 9 is out of bounds.

def test_repro():
    executor = AlgorithmExecutor()
    
    # Create a graph with 9 nodes (0-8)
    G = nx.Graph()
    G.add_nodes_from([str(i) for i in range(9)])
    # Add some edges randomly to match 12 edges count
    edges = [
        ('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), 
        ('4', '5'), ('5', '6'), ('6', '7'), ('7', '8'),
        ('0', '2'), ('2', '4'), ('4', '6'), ('6', '8')
    ]
    G.add_edges_from(edges)
    
    print(f"Graph nodes: {G.nodes()}")
    print(f"Graph edges: {len(G.edges())}")
    
    params = {'source': '8', 'target': '9'}
    print(f"Executing is_connected with params: {params}")
    
    try:
        executor.execute('is_connected', G, params)
        print("Success!")
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_repro()
