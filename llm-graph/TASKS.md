# Task Tracking - LLM Graph Reasoning Framework

## High Priority Tasks (Start Immediately)

### Models Team
- [ ] [MODELS-001] Implement BaseLLMClient initialization
- [ ] [MODELS-002] Implement base generate() method
- [ ] [MODELS-003] Implement structured output generation
- [ ] [OPENAI-001] Set up OpenAI client
- [ ] [OPENAI-002] Implement OpenAI generation
- [ ] [OLLAMA-001] Set up Ollama client
- [ ] [OLLAMA-002] Implement Ollama generation

### Parser Team
- [ ] [PARSER-001] Initialize parser with LLM client
- [ ] [PARSER-002] Implement parse() method
- [ ] [PARSER-003] Implement node extraction
- [ ] [PARSER-004] Implement edge extraction
- [ ] [PARSER-005] Implement structure validation
- [ ] [PARSER-006] Implement parse_with_repair()

### Chooser Team
- [ ] [CHOOSER-001] Initialize chooser
- [ ] [CHOOSER-002] Implement choose_algorithm()
- [ ] [CHOOSER-003] Implement task classification
- [ ] [CHOOSER-004] Implement parameter extraction
- [ ] [CHOOSER-005] Implement algorithm selection logic
- [ ] [CHOOSER-006] Implement validation

### Verifier Team
- [ ] [VERIFIER-001] Initialize verifier
- [ ] [VERIFIER-002] Implement verify_structure()
- [ ] [VERIFIER-003] Implement verify_solution()
- [ ] [VERIFIER-004] Implement connectivity verification
- [ ] [VERIFIER-005] Implement shortest path verification
- [ ] [VERIFIER-006] Implement flow verification
- [ ] [VERIFIER-007] Implement feedback generation

### Orchestrator Team (Depends on agents)
- [ ] [ORCHESTRATOR-001] Initialize orchestrator
- [ ] [ORCHESTRATOR-002] Implement execute() pipeline
- [ ] [ORCHESTRATOR-003] Implement input separation
- [ ] [ORCHESTRATOR-004] Implement parsing stage
- [ ] [ORCHESTRATOR-005] Implement choosing stage
- [ ] [ORCHESTRATOR-006] Implement algorithm stage
- [ ] [ORCHESTRATOR-007] Implement verification stage
- [ ] [ORCHESTRATOR-008] Implement repair loop
- [ ] [ORCHESTRATOR-009] Build LangGraph pipeline

## Medium Priority Tasks

### Algorithms Team
- [ ] [EXECUTOR-001] Initialize algorithm executor
- [ ] [EXECUTOR-002] Implement execute() method
- [ ] [EXECUTOR-003] Implement precondition validation
- [ ] [CONN-001] Implement connectivity check
- [ ] [CONN-002] Implement all paths finding
- [ ] [SP-001] Implement Dijkstra
- [ ] [SP-002] Implement Bellman-Ford
- [ ] [CYCLE-001] Implement cycle detection
- [ ] [CYCLE-002] Implement topological sort

### Benchmark Team
- [ ] [NLGRAPH-001] Initialize NLGraph loader
- [ ] [NLGRAPH-002] Implement dataset loading
- [ ] [NLGRAPH-003] Implement task filtering
- [ ] [EVAL-001] Initialize evaluator
- [ ] [EVAL-002] Implement evaluate()
- [ ] [EVAL-003] Calculate exact match
- [ ] [EVAL-004] Implement statistical analysis
- [ ] [EVAL-005] Generate reports

## Low Priority Tasks

### Utils Team
- [ ] [UTILS-001] Implement dict_to_networkx
- [ ] [UTILS-002] Implement networkx_to_dict
- [ ] [UTILS-003] Implement graph validation
- [ ] [UTILS-004] Implement LLM serialization
- [ ] [UTILS-005] Set up logging
- [ ] [UTILS-006] Implement config loading

### Experiments Team (Depends on everything)
- [ ] [EXP-001] Implement main experiment runner
- [ ] [EXP-002] Complete run_benchmark.py
- [ ] [EXP-003] Implement ablation experiments

### Testing Team
- [ ] [TEST-PARSER-001] Test simple graph parsing
- [ ] [TEST-CHOOSER-001] Test connectivity classification
- [ ] [TEST-INTEGRATION-001] End-to-end connectivity test
- [ ] Add tests for all modules

## Completed Tasks
- [x] Project structure created
- [x] Skeleton code generated
- [x] TODO tasks defined
- [x] Development guide written

## Notes

- Start with Models team (critical dependency)
- Agent teams can work in parallel once Models are ready
- Algorithms team can work independently
- Integration happens after core agents are complete
- Testing should be ongoing throughout development
