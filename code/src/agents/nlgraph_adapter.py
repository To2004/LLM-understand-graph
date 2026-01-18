"""
NLGraph Adapter: Adapts NLGraph benchmark questions for the agent pipeline

This module provides an adapter that converts NLGraph benchmark questions
into a format compatible with the existing agent orchestrator.
"""

from typing import Dict, Any, Optional, Tuple
import re
from pydantic import BaseModel


class NLGraphResult(BaseModel):
    """Result from processing an NLGraph question"""
    success: bool
    natural_language_response: str
    raw_result: Optional[Any] = None
    algorithm_used: Optional[str] = None
    error_message: Optional[str] = None
    matches_expected: Optional[bool] = None
    expected_answer: Optional[str] = None


class NLGraphAdapter:
    """
    Adapter for processing NLGraph benchmark questions through the agent pipeline.
    
    This adapter:
    1. Extracts graph description and task from NLGraph question format
    2. Processes through the agent orchestrator
    3. Optionally validates against expected answers
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the NLGraph adapter.
        
        Args:
            orchestrator: AgentOrchestrator instance
        """
        self.orchestrator = orchestrator
    
    def process_nlgraph_question(
        self, 
        question: str,
        expected_answer: Optional[str] = None
    ) -> NLGraphResult:
        """
        Process an NLGraph question through the agent pipeline.
        
        Args:
            question: NLGraph formatted question
            expected_answer: Optional expected answer for validation
            
        Returns:
            NLGraphResult with pipeline output and optional validation
        """
        try:
            # Extract graph and task contexts
            graph_ctx, task_ctx = self.extract_graph_and_task(question)
            
            # Combine for full input (orchestrator will decompose it)
            full_input = f"{graph_ctx}\n{task_ctx}"
            
            # Run through orchestrator
            result = self.orchestrator.execute(full_input)
            
            # Validate against expected answer if provided
            matches_expected = None
            if expected_answer and result.success:
                matches_expected = self._validate_answer(
                    result.natural_language_response,
                    expected_answer,
                    result.algorithm_used
                )
            
            return NLGraphResult(
                success=result.success,
                natural_language_response=result.natural_language_response,
                raw_result=result.raw_result,
                algorithm_used=result.algorithm_used,
                error_message=result.error_message,
                matches_expected=matches_expected,
                expected_answer=expected_answer
            )
            
        except Exception as e:
            return NLGraphResult(
                success=False,
                natural_language_response=f"Error processing NLGraph question: {str(e)}",
                error_message=str(e)
            )
    
    def extract_graph_and_task(self, question: str) -> Tuple[str, str]:
        """
        Extract graph description and task from NLGraph question.
        
        NLGraph questions have formats like:
        - "In an undirected graph, the nodes are numbered from 0 to 6, and the edges are:..."
        - "Determine if there is a path between two nodes in the graph. Note that (i,j)..."
        
        Args:
            question: NLGraph formatted question
            
        Returns:
            Tuple of (graph_context, task_context)
        """
        # Remove trailing "A:" if present
        question = question.rstrip()
        if question.endswith("A:"):
            question = question[:-2].rstrip()
        
        # Split by "Q:" to separate graph description from question
        if "Q:" in question:
            parts = question.split("Q:", 1)
            graph_ctx = parts[0].strip()
            task_ctx = parts[1].strip()
        else:
            # Some formats don't have explicit Q: marker
            # Try to find the question part
            graph_ctx, task_ctx = self._heuristic_split(question)
        
        return graph_ctx, task_ctx
    
    def _heuristic_split(self, question: str) -> Tuple[str, str]:
        """
        Heuristically split question when no Q: marker exists.
        
        Args:
            question: Question text
            
        Returns:
            Tuple of (graph_context, task_context)
        """
        # Look for common question patterns
        question_patterns = [
            r"(.*?)(Is there a path.*)",
            r"(.*?)(Give the shortest path.*)",
            r"(.*?)(Does .*? have a cycle.*)",
            r"(.*?)(What.*?topological.*)",
            r"(.*?)(Find.*?maximum flow.*)",
            r"(.*?)(Find.*?matching.*)",
            r"(.*?)(Does.*?Hamiltonian.*)",
            r"(.*?)(Simulate.*?message passing.*)",
        ]
        
        for pattern in question_patterns:
            match = re.match(pattern, question, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        # Fallback: treat entire question as combined context
        return question, question
    
    def _validate_answer(
        self, 
        agent_response: str, 
        expected_answer: str,
        algorithm_used: Optional[str] = None
    ) -> bool:
        """
        Validate agent response against expected answer.
        
        This is a simple validation - for more sophisticated comparison,
        you may want to parse both answers and compare structured results.
        
        Args:
            agent_response: Response from agent
            expected_answer: Expected answer from NLGraph
            algorithm_used: Algorithm that was used
            
        Returns:
            True if answers match (approximately), False otherwise
        """
        # Normalize both answers
        agent_norm = self._normalize_answer(agent_response)
        expected_norm = self._normalize_answer(expected_answer)
        
        # For connectivity tasks
        if "yes" in expected_norm or "no" in expected_norm:
            return ("yes" in agent_norm) == ("yes" in expected_norm)
        
        # For path/cycle tasks - extract path if present
        agent_path = self._extract_path(agent_norm)
        expected_path = self._extract_path(expected_norm)
        
        if agent_path and expected_path:
            return agent_path == expected_path
        
        # For numeric answers (flow, matching, etc.)
        agent_num = self._extract_number(agent_norm)
        expected_num = self._extract_number(expected_norm)
        
        if agent_num is not None and expected_num is not None:
            return abs(agent_num - expected_num) < 0.01
        
        # Fallback: simple substring match
        return expected_norm in agent_norm or agent_norm in expected_norm
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        return answer.lower().strip().replace(",", "").replace(".", "")
    
    def _extract_path(self, text: str) -> Optional[list]:
        """Extract path from answer text"""
        # Look for patterns like "1,2,3" or "1 -> 2 -> 3" or "1-2-3"
        patterns = [
            r"path[:\s]+([0-9,\s\-\>]+)",
            r"is[:\s]+([0-9,\s\-\>]+)",
            r"([0-9]+(?:[,\-\>]+[0-9]+)+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                path_str = match.group(1)
                # Extract numbers
                numbers = re.findall(r'\d+', path_str)
                return [int(n) for n in numbers]
        
        return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from answer text"""
        # Look for patterns like "weight of 11" or "flow is 25"
        patterns = [
            r"weight[:\s]+of[:\s]+([0-9.]+)",
            r"flow[:\s]+(?:is|of)[:\s]+([0-9.]+)",
            r"matching[:\s]+(?:is|of)[:\s]+([0-9.]+)",
            r"([0-9.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def run_batch(
        self,
        samples: list,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a batch of NLGraph samples through the pipeline.
        
        Args:
            samples: List of NLGraph samples (dicts with 'question' and 'answer')
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary with results and statistics
        """
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        correct = 0
        total = 0
        
        for i, sample in enumerate(samples):
            question = sample.get('question', '')
            expected = sample.get('answer', '')
            
            print(f"\n[{i+1}/{len(samples)}] Processing sample {sample.get('id', i)}...")
            
            result = self.process_nlgraph_question(question, expected)
            
            results.append({
                'id': sample.get('id', i),
                'task': sample.get('task', 'unknown'),
                'success': result.success,
                'matches_expected': result.matches_expected,
                'agent_response': result.natural_language_response,
                'expected_answer': expected,
                'algorithm_used': result.algorithm_used,
                'error': result.error_message
            })
            
            total += 1
            if result.matches_expected:
                correct += 1
                print(f"✅ Correct")
            elif result.success:
                print(f"⚠️ Completed but answer may not match")
            else:
                print(f"❌ Failed: {result.error_message}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'results': results
        }
