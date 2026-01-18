"""
Agent Orchestrator: Main coordinator for the 4-phase pipeline
"""

from typing import Optional, Any, Dict, Tuple
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import traceback

from .validator import PromptValidator, ValidationResult
from .decomposer import PromptDecomposer, DecompositionResult


class PipelineResult(BaseModel):
    """Result from the complete pipeline execution"""
    success: bool
    natural_language_response: str
    raw_result: Optional[Any] = None
    graph_structure: Optional[Any] = None
    algorithm_used: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentOrchestrator:
    """
    Main orchestrator that coordinates the entire 4-phase pipeline.
    
    Phase 1: Validation & Decomposition
    Phase 2: Parallel Processing (Stream A: Parser, Stream B: Router)
    Phase 3: Execution
    Phase 4: Synthesis
    """
    
    def __init__(
        self,
        llm_client,
        parser,
        chooser,
        executor,
        synthesizer,
        validator: Optional[PromptValidator] = None,
        decomposer: Optional[PromptDecomposer] = None,
        verifier=None
    ):
        """
        Initialize orchestrator with all pipeline components.
        
        Args:
            llm_client: LLM client for agents
            parser: AgentParser for graph parsing
            chooser: AgentChooser for algorithm selection
            executor: AlgorithmExecutor for running algorithms
            synthesizer: AgentSynthesizer for response generation
            validator: Optional PromptValidator
            decomposer: Optional PromptDecomposer
            verifier: Optional AgentVerifier for verification loop
        """
        self.llm_client = llm_client
        self.parser = parser
        self.chooser = chooser
        self.executor = executor
        self.synthesizer = synthesizer
        self.verifier = verifier
        
        # Create validator and decomposer if not provided
        self.validator = validator or PromptValidator(llm_client)
        self.decomposer = decomposer or PromptDecomposer(llm_client)
    
    def execute(self, user_input: str) -> PipelineResult:
        """
        Execute the full 4-phase pipeline on user input.
        
        Args:
            user_input: Natural language input containing graph and task
            
        Returns:
            PipelineResult with natural language response
        """
        try:
            # Phase 1: Validation & Decomposition
            print("\n[Orchestrator] ========== PHASE 1: VALIDATION & DECOMPOSITION ==========")
            validation_result = self._validate_prompt(user_input)
            
            if not validation_result.is_valid:
                print(f"[Orchestrator] ❌ Validation failed: {validation_result.rejection_reason}")
                return PipelineResult(
                    success=False,
                    natural_language_response=f"Invalid query: {validation_result.rejection_reason}",
                    error_message=validation_result.rejection_reason
                )
            
            print(f"[Orchestrator] ✅ Validation passed (confidence: {validation_result.confidence})")
            
            # Decompose prompt
            decomposition_result = self._decompose_prompt(user_input)
            graph_ctx = decomposition_result.graph_context
            task_ctx = decomposition_result.task_context
            print(f"[Orchestrator] Graph context: {graph_ctx[:80]}...")
            print(f"[Orchestrator] Task context: {task_ctx[:80]}...")
            
            # Phase 2: Parallel Processing
            print("\n[Orchestrator] ========== PHASE 2: PARALLEL PROCESSING ==========")
            graph_structure, algorithm_choice = self._execute_parallel_streams(
                graph_ctx, task_ctx
            )
            
            # Convert graph structure to NetworkX graph
            from ..utils.graph_utils import GraphUtils
            
            # Check if graph_structure has a to_dict method (UnifiedGraphFormat)
            if hasattr(graph_structure, 'to_dict'):
                graph_dict = graph_structure.to_dict()
            elif hasattr(graph_structure, 'dict'):
                graph_dict = graph_structure.dict()
            else:
                # Assume it's already a dict-like object
                graph_dict = {
                    'nodes': graph_structure.nodes,
                    'edges': graph_structure.edges,
                    'directed': graph_structure.directed,
                    'weighted': graph_structure.weighted,
                    'weights': graph_structure.weights if graph_structure.weighted else None
                }
            
            nx_graph = GraphUtils.dict_to_networkx(graph_dict)
            print(f"[Orchestrator] NetworkX graph created: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
            
            # Phase 3: Execution
            print("\n[Orchestrator] ========== PHASE 3: ALGORITHM EXECUTION ==========")
            print(f"[Orchestrator] Executing {algorithm_choice.algorithm_name} with params: {algorithm_choice.parameters}")
            raw_result = self.executor.execute(
                algorithm_name=algorithm_choice.algorithm_name,
                graph=nx_graph,
                parameters=algorithm_choice.parameters
            )
            print(f"[Orchestrator] Execution result: {raw_result}")
            
            # Phase 4: Synthesis
            print("\n[Orchestrator] ========== PHASE 4: SYNTHESIS ==========")
            synthesis_result = self.synthesizer.synthesize(
                raw_result=raw_result,
                task_query=task_ctx,
                algorithm_name=algorithm_choice.algorithm_name,
                graph_structure=graph_structure
            )
            print(f"[Orchestrator] Synthesized response: {synthesis_result.natural_language_response}")
            
            print("\n[Orchestrator] ========== PIPELINE COMPLETE ==========")
            return PipelineResult(
                success=True,
                natural_language_response=synthesis_result.natural_language_response,
                raw_result=raw_result,
                graph_structure=graph_structure,
                algorithm_used=algorithm_choice.algorithm_name,
                metadata={
                    'validation_confidence': validation_result.confidence,
                    'decomposition_confidence': decomposition_result.confidence,
                    'algorithm_confidence': algorithm_choice.confidence
                }
            )
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            traceback.print_exc()
            
            return PipelineResult(
                success=False,
                natural_language_response=f"I encountered an error processing your query: {str(e)}",
                error_message=error_msg
            )
    
    def _validate_prompt(self, prompt: str) -> ValidationResult:
        """
        Validate that prompt is a graph reasoning task.
        
        Args:
            prompt: User input
            
        Returns:
            ValidationResult
        """
        return self.validator.validate(prompt)
    
    def _decompose_prompt(self, prompt: str) -> DecompositionResult:
        """
        Split prompt into (graph_context, task_context).
        
        Args:
            prompt: User input
            
        Returns:
            DecompositionResult
        """
        return self.decomposer.decompose(prompt)
    
    def _execute_parallel_streams(
        self, 
        graph_ctx: str, 
        task_ctx: str
    ) -> Tuple[Any, Any]:
        """
        Execute Stream A (Parser) and Stream B (Router) in parallel.
        
        Uses ThreadPoolExecutor for parallel execution.
        
        Args:
            graph_ctx: Graph context from decomposition
            task_ctx: Task context from decomposition
            
        Returns:
            Tuple of (graph_structure, algorithm_choice)
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both streams
            future_a = executor.submit(self._run_stream_a, graph_ctx)
            future_b = executor.submit(self._run_stream_b, task_ctx)
            
            # Wait for both to complete
            graph_structure = future_a.result()
            algorithm_choice = future_b.result()
        
        return graph_structure, algorithm_choice
    
    def _run_stream_a(self, graph_ctx: str) -> Any:
        """
        Stream A: Data Pipeline (Parser → Graph Builder).
        
        Args:
            graph_ctx: Graph context
            
        Returns:
            GraphStructure or UnifiedGraphFormat
        """
        # Parse the graph description
        graph_structure = self.parser.parse(graph_ctx)
        return graph_structure
    
    def _run_stream_b(self, task_ctx: str) -> Any:
        """
        Stream B: Logic Pipeline (Router → Algorithm Selection).
        
        Args:
            task_ctx: Task context
            
        Returns:
            AlgorithmChoice
        """
        # Choose algorithm based on task
        algorithm_choice = self.chooser.choose_algorithm(task_ctx)
        return algorithm_choice
