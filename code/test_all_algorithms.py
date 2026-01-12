"""
Algorithm Test Suite

This script tests all supported graph algorithms with example queries.
Run this to verify that the pipeline can handle all algorithm types.
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphReasoningPipeline


class Logger:
    """Dual logger that writes to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        """Write to both console and file"""
        print(message, end='')
        self.file.write(message)
        self.file.flush()
    
    def close(self):
        """Close the log file"""
        self.file.close()


# Test queries for each algorithm type
TEST_QUERIES = {
    "CONNECTIVITY": [
        "Graph: A--B, B--C, D--E. Are nodes A and C connected?",
        "Graph: 1->2, 2->3, 3->1, 4->5. Is there a path from node 1 to node 4?",
        "Graph: X--Y--Z. Can I reach Z from X?",
    ],
    
    "SHORTEST_PATH": [
        "Graph: A->B, B->C, C->D. What's the shortest path from A to D?",
        "Graph: 1--2 (weight 5), 1--3 (weight 2), 3--2 (weight 1). Find shortest path from 1 to 2.",
        "Graph: Start->Mid1, Start->Mid2, Mid1->End, Mid2->End. What's the path length from Start to End?",
    ],
    
    "CYCLE_DETECTION": [
        "Graph: A->B, B->C, C->A. Does this graph have a cycle?",
        "Graph: 1->2, 2->3, 3->4. Is there a cycle in this graph?",
        "Graph: X->Y, Y->Z, Z->X, X->W. Find all cycles.",
    ],
    
    "TOPOLOGICAL_SORT": [
        "Graph: Task1->Task2, Task1->Task3, Task2->Task4, Task3->Task4. What's the topological order?",
        "Graph: A->B, A->C, B->D, C->D. Give me a valid ordering of nodes.",
        "Graph: Course1->Course2, Course1->Course3, Course2->Course4. In what order should I take these courses?",
    ],
    
    "MAXIMUM_FLOW": [
        "Graph: S->A (capacity 10), S->B (capacity 5), A->T (capacity 10), B->T (capacity 5). What's the maximum flow from S to T?",
        "Graph: Source->Node1 (cap 20), Source->Node2 (cap 10), Node1->Sink (cap 15), Node2->Sink (cap 10). Find max flow from Source to Sink.",
        "Graph: Start->Mid1 (10), Start->Mid2 (15), Mid1->End (8), Mid2->End (12). Maximum flow from Start to End?",
    ],
    
    "BIPARTITE_MATCHING": [
        "Graph: Worker1--Job1, Worker1--Job2, Worker2--Job2, Worker3--Job3. Find maximum matching.",
        "Graph: Student1--Project1, Student1--Project2, Student2--Project2, Student3--Project3. What's the best assignment?",
        "Graph: A--X, A--Y, B--Y, C--Z. Find maximum bipartite matching.",
    ],
    
    "HAMILTONIAN_PATH": [
        "Graph: A--B, B--C, C--D, D--A, A--C. Is there a Hamiltonian path?",
        "Graph: 1--2, 2--3, 3--4, 4--1, 1--3, 2--4. Find a path visiting all nodes exactly once.",
        "Graph: X--Y, Y--Z, Z--W, W--X. Does a Hamiltonian cycle exist?",
    ],
    
    "GNN_MESSAGE_PASSING": [
        "Graph: A--B, B--C, C--D. Simulate message passing from A with initial value 1.",
        "Graph: Node1--Node2, Node2--Node3. Propagate features through the graph.",
        "Graph: 1--2, 2--3, 3--1. Run GNN message passing for 2 iterations.",
    ],
}


def run_tests(verbose=True, log_dir="logs"):
    """Run all algorithm tests"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_results_{timestamp}.log")
    json_file = os.path.join(log_dir, f"test_results_{timestamp}.json")
    
    # Initialize logger
    logger = Logger(log_file)
    
    logger.write("=" * 80 + "\n")
    logger.write("GRAPH REASONING PIPELINE - ALGORITHM TEST SUITE\n")
    logger.write("=" * 80 + "\n")
    logger.write(f"\nTest Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.write(f"Log File: {log_file}\n")
    logger.write(f"JSON File: {json_file}\n")
    logger.write("\nInitializing pipeline...\n")
    
    # Initialize pipeline
    pipeline = GraphReasoningPipeline(verbose=verbose)
    logger.write("✅ Pipeline ready!\n\n")
    
    # Track results
    total_tests = sum(len(queries) for queries in TEST_QUERIES.values())
    passed = 0
    failed = 0
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "passed": 0,
        "failed": 0,
        "success_rate": 0.0,
        "algorithm_results": {}
    }
    
    # Run tests for each algorithm type
    for algo_type, queries in TEST_QUERIES.items():
        logger.write(f"\n{'='*80}\n")
        logger.write(f"Testing: {algo_type}\n")
        logger.write(f"{'='*80}\n\n")
        
        results["algorithm_results"][algo_type] = []
        
        for i, query in enumerate(queries, 1):
            logger.write(f"\n[Test {i}/{len(queries)}] Query: {query[:80]}...\n")
            
            # Add delay to avoid rate limits
            if i > 1 or list(TEST_QUERIES.keys()).index(algo_type) > 0:
                time.sleep(10)  # Wait 10 seconds between requests
            
            try:
                result = pipeline.run(query)
                
                if result.success:
                    logger.write(f"✅ SUCCESS\n")
                    logger.write(f"   Response: {result.natural_language_response}\n")
                    logger.write(f"   Algorithm: {result.algorithm_used}\n")
                    passed += 1
                    results["algorithm_results"][algo_type].append({
                        "query": query,
                        "success": True,
                        "response": result.natural_language_response,
                        "algorithm": result.algorithm_used
                    })
                else:
                    logger.write(f"❌ FAILED: {result.error_message}\n")
                    failed += 1
                    results["algorithm_results"][algo_type].append({
                        "query": query,
                        "success": False,
                        "error": result.error_message
                    })
                    
            except Exception as e:
                logger.write(f"❌ EXCEPTION: {str(e)}\n")
                failed += 1
                results["algorithm_results"][algo_type].append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
    
    # Update summary statistics
    results["passed"] = passed
    results["failed"] = failed
    results["success_rate"] = (passed/total_tests)*100 if total_tests > 0 else 0.0
    
    # Print summary
    logger.write("\n" + "=" * 80 + "\n")
    logger.write("TEST SUMMARY\n")
    logger.write("=" * 80 + "\n")
    logger.write(f"\nTotal Tests: {total_tests}\n")
    logger.write(f"✅ Passed: {passed}\n")
    logger.write(f"❌ Failed: {failed}\n")
    logger.write(f"Success Rate: {results['success_rate']:.1f}%\n\n")
    
    # Print per-algorithm summary
    logger.write("\nPer-Algorithm Results:\n")
    logger.write("-" * 80 + "\n")
    for algo_type, algo_results in results["algorithm_results"].items():
        algo_passed = sum(1 for r in algo_results if r["success"])
        algo_total = len(algo_results)
        logger.write(f"{algo_type:25} {algo_passed}/{algo_total} passed\n")
    
    logger.write("\n" + "=" * 80 + "\n")
    logger.write(f"\n✅ Results saved to:\n")
    logger.write(f"   Log: {log_file}\n")
    logger.write(f"   JSON: {json_file}\n\n")
    
    # Close logger
    logger.close()
    
    # Save JSON results
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def run_single_test(algo_type, test_index=0):
    """Run a single test for debugging"""
    if algo_type not in TEST_QUERIES:
        print(f"❌ Unknown algorithm type: {algo_type}")
        print(f"Available types: {', '.join(TEST_QUERIES.keys())}")
        return
    
    queries = TEST_QUERIES[algo_type]
    if test_index >= len(queries):
        print(f"❌ Test index {test_index} out of range (0-{len(queries)-1})")
        return
    
    query = queries[test_index]
    
    print(f"\nTesting {algo_type} - Test {test_index}")
    print(f"Query: {query}\n")
    
    pipeline = GraphReasoningPipeline(verbose=True)
    result = pipeline.run(query)
    
    print(f"\n{'='*60}")
    if result.success:
        print(f"✅ {result.natural_language_response}")
        print(f"   Algorithm: {result.algorithm_used}")
    else:
        print(f"❌ {result.error_message}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all graph algorithms")
    parser.add_argument(
        "--single",
        type=str,
        help="Run single test (format: ALGORITHM_TYPE:INDEX, e.g., SHORTEST_PATH:0)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Run single test
        parts = args.single.split(":")
        algo_type = parts[0]
        test_index = int(parts[1]) if len(parts) > 1 else 0
        run_single_test(algo_type, test_index)
    else:
        # Run all tests
        run_tests(verbose=not args.quiet)
