"""
Main pipeline entry point for the LLM Graph Reasoning Framework

This module provides the complete 4-phase pipeline:
1. Validation & Decomposition
2. Parallel Processing (Parser + Router)
3. Execution
4. Synthesis
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel

from .agents import AgentParser, AgentChooser, AgentSynthesizer
from .algorithms import AlgorithmExecutor
from .agents import AgentOrchestrator, PromptValidator, PromptDecomposer
from .llms.base import BaseLLMClient


# Re-export PipelineResult from orchestrator
from .agents.orchestrator import PipelineResult


class GraphReasoningPipeline:
    """
    Complete graph reasoning pipeline.
    
    This is the main entry point for using the framework.
    
    Usage:
        >>> from models import OpenRouterClient
        >>> llm = OpenRouterClient(model_name="meta-llama/llama-3.3-70b-instruct:free")
        >>> pipeline = GraphReasoningPipeline(llm)
        >>> result = pipeline.run("Graph: A--B--C. Is A connected to C?")
        >>> print(result.natural_language_response)
        "Yes, the nodes are connected."
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        enable_verification: bool = False,
        verbose: bool = False,
        agent_models: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the complete pipeline.
        
        Args:
            llm_client: LLM client for all agents (if None, creates OpenRouter client with Llama 3.3 70B)
            enable_verification: Whether to enable verification loop (experimental)
            verbose: Whether to print debug information
            agent_models: Optional dict to specify different models per agent
                         e.g., {"parser": "llama-3.3-70b", "chooser": "deepseek-r1"}
        """
        self.verbose = verbose
        
        # Initialize LLM client if not provided
        if llm_client is None:
            if self.verbose:
                print("No LLM client provided, using OpenRouter with Llama 3.3 70B...")
            from .llms import OpenRouterClient
            llm_client = OpenRouterClient(model_name="meta-llama/llama-3.3-70b-instruct:free")
        
        self.llm_client = llm_client
        
        # Initialize all components
        if self.verbose:
            print("Initializing pipeline components...")
        
        # Create per-agent clients if different models specified
        if agent_models:
            from .llms import OpenRouterClient
            parser_client = OpenRouterClient(model_name=agent_models.get("parser", "meta-llama/llama-3.3-70b-instruct:free"))
            chooser_client = OpenRouterClient(model_name=agent_models.get("chooser", "meta-llama/llama-3.3-70b-instruct:free"))
            synthesizer_client = OpenRouterClient(model_name=agent_models.get("synthesizer", "meta-llama/llama-3.3-70b-instruct:free"))
            validator_client = OpenRouterClient(model_name=agent_models.get("validator", "meta-llama/llama-3.3-70b-instruct:free"))
            decomposer_client = OpenRouterClient(model_name=agent_models.get("decomposer", "meta-llama/llama-3.3-70b-instruct:free"))
            
            self.parser = AgentParser(parser_client)
            self.chooser = AgentChooser(chooser_client)
            self.synthesizer = AgentSynthesizer(synthesizer_client)
            self.validator = PromptValidator(validator_client)
            self.decomposer = PromptDecomposer(decomposer_client)
            
            if self.verbose:
                print(f"Using custom models per agent:")
                for agent, model in agent_models.items():
                    print(f"  {agent}: {model}")
        else:
            # Use same client for all agents
            self.parser = AgentParser(llm_client)
            self.chooser = AgentChooser(llm_client)
            self.synthesizer = AgentSynthesizer(llm_client)
            self.validator = PromptValidator(llm_client)
            self.decomposer = PromptDecomposer(llm_client)
            
            if self.verbose:
                print(f"Using {llm_client.model_name} for all agents")
        
        self.executor = AlgorithmExecutor()
        
        # Verifier is not implemented yet
        if enable_verification:
            if self.verbose:
                print("Warning: Verification not yet implemented")
        self.verifier = None
        
        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(
            llm_client=llm_client,
            parser=self.parser,
            chooser=self.chooser,
            executor=self.executor,
            synthesizer=self.synthesizer,
            validator=self.validator,
            decomposer=self.decomposer,
            verifier=self.verifier
        )
        
        if self.verbose:
            print("Pipeline initialized successfully!")
    
    def run(self, user_input: str) -> PipelineResult:
        """
        Execute the full 4-phase pipeline.
        
        Args:
            user_input: Natural language input containing graph and task
            
        Returns:
            PipelineResult with natural language response and metadata
            
        Example:
            >>> result = pipeline.run("Graph: A--B, B--C. Find path from A to C.")
            >>> print(result.natural_language_response)
            >>> print(result.algorithm_used)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing query: {user_input}")
            print(f"{'='*60}\n")
        
        result = self.orchestrator.execute(user_input)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Result: {result.natural_language_response}")
            print(f"Success: {result.success}")
            if result.algorithm_used:
                print(f"Algorithm: {result.algorithm_used}")
            print(f"{'='*60}\n")
        
        return result
    
    def run_batch(self, queries: list) -> list:
        """
        Run multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        for i, query in enumerate(queries):
            if self.verbose:
                print(f"\nProcessing query {i+1}/{len(queries)}")
            result = self.run(query)
            results.append(result)
        return results


# Convenience function for quick usage
def quick_query(query: str, model_name: str = "meta-llama/llama-3.3-70b-instruct:free", provider: str = "openrouter"):
    """
    Quick one-off query without creating a pipeline object.
    
    Args:
        query: Natural language query
        model_name: Model name to use (defaults to Llama 3.3 70B on OpenRouter)
        provider: LLM provider ("ollama", "openai", "openrouter")
        
    Returns:
        Natural language response string
        
    Example:
        >>> response = quick_query("Graph: A--B--C. Is A connected to C?")
        >>> print(response)
    """
    # Import appropriate client
    if provider == "openrouter":
        from .llms import OpenRouterClient
        client = OpenRouterClient(model_name=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Create pipeline and run
    pipeline = GraphReasoningPipeline(client)
    result = pipeline.run(query)
    
    return result.natural_language_response


if __name__ == "__main__":
    # Example usage
    print("LLM Graph Reasoning Framework - Pipeline Demo")
    print("=" * 60)
    
    # This is just a demo - actual usage requires proper LLM client setup
    print("\nTo use this pipeline:")
    print("1. Set up your LLM client (Ollama, OpenAI, or OpenRouter)")
    print("2. Create a pipeline: pipeline = GraphReasoningPipeline(llm_client)")
    print("3. Run queries: result = pipeline.run('your query')")
    print("\nSee examples/basic_usage.py for complete examples.")
