"""
Run comprehensive NLGraph benchmark: 3 samples per task per difficulty level
"""

import subprocess
import sys
import os

# Change to code directory
os.chdir(r'c:\Users\user\Documents\GitHub\LLM-understand-graph\code')

difficulties = ['easy', 'medium', 'hard']
tasks = ['shortest_path', 'connectivity', 'cycle', 'flow', 
         'matching', 'hamilton', 'topology', 'GNN']

print("="*80)
print("Running Comprehensive NLGraph Benchmark")
print("3 samples per task per difficulty level")
print("="*80)

for difficulty in difficulties:
    print(f"\n{'='*80}")
    print(f"Running {difficulty.upper()} difficulty level")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        'run_nlgraph_benchmark.py',
        '--difficulty', difficulty,
        '--max-samples', '3',
        '--output-dir', f'logs/nlgraph_results/{difficulty}'
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nWarning: {difficulty} run had issues")

print("\n" + "="*80)
print("Benchmark Complete!")
print("="*80)
print("\nResults saved to:")
print("  - logs/nlgraph_results/easy/")
print("  - logs/nlgraph_results/medium/")
print("  - logs/nlgraph_results/hard/")
