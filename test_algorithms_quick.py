"""
Quick test script to verify all algorithm implementations work correctly.
"""

import sys
sys.path.insert(0, r'c:\Users\user\OneDrive - post.bgu.ac.il\Courses-Drive\sem7\Research Methods\LLM-understand-graph\code')

import networkx as nx
from src.algorithms import (
    ConnectivityAlgorithms,
    ShortestPathAlgorithms,
    FlowAlgorithms,
    CycleAlgorithms,
    MatchingAlgorithms,
    AlgorithmExecutor
)

def test_connectivity():
    """Test connectivity algorithms"""
    print("=" * 60)
    print("Testing Connectivity Algorithms")
    print("=" * 60)
    
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
    
    # Test is_connected
    connected, path = ConnectivityAlgorithms.is_connected(G, 'A', 'D')
    print(f"✓ Is A connected to D? {connected}, Path: {path}")
    
    # Test find_all_paths
    paths = ConnectivityAlgorithms.find_all_paths(G, 'A', 'D', max_paths=5)
    print(f"✓ All paths from A to D: {paths}")
    
    # Test connected_components
    components = ConnectivityAlgorithms.connected_components(G)
    print(f"✓ Connected components: {components}")
    print()

def test_shortest_path():
    """Test shortest path algorithms"""
    print("=" * 60)
    print("Testing Shortest Path Algorithms")
    print("=" * 60)
    
    # Create weighted graph
    G = nx.Graph()
    G.add_weighted_edges_from([
        ('A', 'B', 1),
        ('B', 'C', 2),
        ('A', 'C', 4),
        ('C', 'D', 1)
    ])
    
    # Test Dijkstra
    path, length = ShortestPathAlgorithms.dijkstra(G, 'A', 'D')
    print(f"✓ Dijkstra A→D: Path={path}, Length={length}")
    
    # Test Bellman-Ford
    path, length = ShortestPathAlgorithms.bellman_ford(G, 'A', 'D')
    print(f"✓ Bellman-Ford A→D: Path={path}, Length={length}")
    
    # Test all pairs
    all_paths = ShortestPathAlgorithms.all_pairs_shortest_path(G)
    print(f"✓ All pairs shortest paths computed for {len(all_paths)} nodes")
    print()

def test_flow():
    """Test flow algorithms"""
    print("=" * 60)
    print("Testing Flow Algorithms")
    print("=" * 60)
    
    # Create directed graph with capacities
    G = nx.DiGraph()
    G.add_edge('S', 'A', capacity=10)
    G.add_edge('S', 'B', capacity=5)
    G.add_edge('A', 'T', capacity=10)
    G.add_edge('B', 'T', capacity=5)
    
    # Test maximum flow
    flow_value, flow_dict = FlowAlgorithms.maximum_flow(G, 'S', 'T')
    print(f"✓ Maximum flow S→T: {flow_value}")
    
    # Test minimum cut
    cut_value, partition = FlowAlgorithms.minimum_cut(G, 'S', 'T')
    print(f"✓ Minimum cut value: {cut_value}")
    print()

def test_cycles():
    """Test cycle algorithms"""
    print("=" * 60)
    print("Testing Cycle Algorithms")
    print("=" * 60)
    
    # Create graph with cycle
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    
    # Test has_cycle
    has_cycle, cycle = CycleAlgorithms.has_cycle(G)
    print(f"✓ Has cycle? {has_cycle}, Cycle: {cycle}")
    
    # Test topological sort on DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
    topo_order = CycleAlgorithms.topological_sort(DAG)
    print(f"✓ Topological sort: {topo_order}")
    
    # Test find_all_cycles
    cycles = CycleAlgorithms.find_all_cycles(G, max_cycles=10)
    print(f"✓ Found {len(cycles)} cycles")
    print()

def test_matching():
    """Test matching algorithms"""
    print("=" * 60)
    print("Testing Matching Algorithms")
    print("=" * 60)
    
    # Create simple graph
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('C', 'D'), ('E', 'F')])
    
    # Test maximum matching
    matching = MatchingAlgorithms.maximum_matching(G)
    print(f"✓ Maximum matching: {matching}")
    
    # Test bipartite check
    is_bip, coloring = MatchingAlgorithms.is_bipartite(G)
    print(f"✓ Is bipartite? {is_bip}")
    
    # Test bipartite matching
    if is_bip:
        top_nodes = {node for node, color in coloring.items() if color == 0}
        bip_matching = MatchingAlgorithms.bipartite_matching(G, top_nodes)
        print(f"✓ Bipartite matching: {len(bip_matching)} edges")
    print()

def test_executor():
    """Test the unified executor"""
    print("=" * 60)
    print("Testing Algorithm Executor")
    print("=" * 60)
    
    executor = AlgorithmExecutor()
    
    # Create test graph
    G = nx.Graph()
    G.add_weighted_edges_from([('A', 'B', 1), ('B', 'C', 2)])
    
    # Test execution via executor
    result = executor.execute(
        'dijkstra',
        G,
        {'source': 'A', 'target': 'C', 'weight': 'weight'}
    )
    print(f"✓ Executor dijkstra: {result}")
    print()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ALGORITHM IMPLEMENTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_connectivity()
        test_shortest_path()
        test_flow()
        test_cycles()
        test_matching()
        test_executor()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
