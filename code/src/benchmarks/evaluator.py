"""
Benchmark evaluation and metrics calculation
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from pathlib import Path
from scipy import stats
import numpy as np


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for benchmark results"""
    exact_match_accuracy: float
    tool_calling_success_rate: float
    repair_efficiency: float
    average_iterations: float
    total_samples: int
    successful_samples: int


class BenchmarkEvaluator:
    """
    Evaluates system performance on benchmarks.
    
    Provides comprehensive evaluation metrics including exact match accuracy,
    tool calling success rates, repair efficiency, and statistical significance
    testing for comparing different models or approaches.
    """
    
    def __init__(self):
        """
        Initialize evaluator.
        
        Sets up metrics tracking, result storage, and statistical test
        configurations.
        """
        self.results_history = []
        self.current_results = None
    
    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted outputs
            ground_truths: List of ground truth answers
            metadata: List of metadata dictionaries containing execution info
            
        Returns:
            EvaluationMetrics object with calculated metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truths "
                f"({len(ground_truths)}) must have same length"
            )
        
        # Calculate exact match accuracy
        exact_match_acc = self.calculate_exact_match(predictions, ground_truths)
        
        # Calculate tool calling success rate
        tool_success_rate = self._calculate_tool_success_rate(metadata)
        
        # Calculate repair efficiency
        repair_eff = self._calculate_repair_efficiency(metadata)
        
        # Calculate average iterations
        avg_iterations = self._calculate_average_iterations(metadata)
        
        # Count successful samples
        successful_samples = sum(
            1 for m in metadata if m.get('success', False)
        )
        
        metrics = EvaluationMetrics(
            exact_match_accuracy=exact_match_acc,
            tool_calling_success_rate=tool_success_rate,
            repair_efficiency=repair_eff,
            average_iterations=avg_iterations,
            total_samples=len(predictions),
            successful_samples=successful_samples
        )
        
        self.current_results = metrics
        self.results_history.append(metrics)
        
        return metrics
    
    def calculate_exact_match(
        self,
        predictions: List[Any],
        ground_truths: List[Any]
    ) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted outputs
            ground_truths: List of ground truth answers
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) == 0:
            return 0.0
        
        matches = 0
        for pred, truth in zip(predictions, ground_truths):
            # Normalize for comparison
            pred_normalized = self._normalize_answer(pred)
            truth_normalized = self._normalize_answer(truth)
            
            if pred_normalized == truth_normalized:
                matches += 1
        
        return matches / len(predictions)
    
    def _normalize_answer(self, answer: Any) -> str:
        """
        Normalize answer for comparison.
        
        Args:
            answer: Answer to normalize
            
        Returns:
            Normalized string representation
        """
        if answer is None:
            return ""
        
        # Convert to string and normalize
        answer_str = str(answer).strip().lower()
        
        # Remove common variations
        answer_str = answer_str.replace("the answer is", "").strip()
        answer_str = answer_str.replace(":", "").strip()
        
        return answer_str
    
    def _calculate_tool_success_rate(self, metadata: List[Dict[str, Any]]) -> float:
        """
        Calculate tool calling success rate.
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Success rate as a float between 0 and 1
        """
        if len(metadata) == 0:
            return 0.0
        
        successful_tool_calls = sum(
            1 for m in metadata 
            if m.get('tool_calling_success', False) or m.get('success', False)
        )
        
        return successful_tool_calls / len(metadata)
    
    def _calculate_repair_efficiency(self, metadata: List[Dict[str, Any]]) -> float:
        """
        Calculate repair efficiency (successful repairs / total repair attempts).
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Repair efficiency as a float between 0 and 1
        """
        total_repairs = 0
        successful_repairs = 0
        
        for m in metadata:
            repair_attempts = m.get('repair_attempts', 0)
            if repair_attempts > 0:
                total_repairs += repair_attempts
                if m.get('success', False):
                    successful_repairs += repair_attempts
        
        if total_repairs == 0:
            return 1.0  # No repairs needed = perfect efficiency
        
        return successful_repairs / total_repairs
    
    def _calculate_average_iterations(self, metadata: List[Dict[str, Any]]) -> float:
        """
        Calculate average number of iterations per sample.
        
        Args:
            metadata: List of metadata dictionaries
            
        Returns:
            Average iterations as a float
        """
        if len(metadata) == 0:
            return 0.0
        
        total_iterations = sum(m.get('iterations', 1) for m in metadata)
        return total_iterations / len(metadata)
    
    def statistical_analysis(
        self,
        baseline_results: EvaluationMetrics,
        system_results: EvaluationMetrics
    ) -> Dict[str, Any]:
        """
        Perform statistical significance tests.
        
        Compares baseline and system results using McNemar's test for
        paired comparisons.
        
        Args:
            baseline_results: Baseline evaluation metrics
            system_results: System evaluation metrics
            
        Returns:
            Dictionary with statistical test results and p-values
        """
        analysis = {
            'baseline_accuracy': baseline_results.exact_match_accuracy,
            'system_accuracy': system_results.exact_match_accuracy,
            'improvement': system_results.exact_match_accuracy - baseline_results.exact_match_accuracy,
            'relative_improvement': (
                (system_results.exact_match_accuracy - baseline_results.exact_match_accuracy) 
                / baseline_results.exact_match_accuracy
                if baseline_results.exact_match_accuracy > 0 else 0
            )
        }
        
        # McNemar's test for paired binary outcomes
        # Requires individual sample results, using aggregate approximation
        n = baseline_results.total_samples
        
        # Approximate contingency table
        baseline_correct = int(baseline_results.exact_match_accuracy * n)
        system_correct = int(system_results.exact_match_accuracy * n)
        
        # Estimate discordant pairs (simplified)
        both_correct = min(baseline_correct, system_correct)
        baseline_only = baseline_correct - both_correct
        system_only = system_correct - both_correct
        
        if baseline_only + system_only > 0:
            # McNemar's test statistic
            mcnemar_stat = ((abs(baseline_only - system_only) - 1) ** 2) / (baseline_only + system_only)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            
            analysis['mcnemar_statistic'] = mcnemar_stat
            analysis['p_value'] = p_value
            analysis['significant'] = p_value < 0.05
        else:
            analysis['mcnemar_statistic'] = None
            analysis['p_value'] = None
            analysis['significant'] = False
        
        return analysis
    
    def multi_model_comparison(
        self,
        results_list: List[EvaluationMetrics],
        model_names: List[str]
    ) -> Dict[str, Any]:
        """
        Perform ANOVA for multi-model comparison.
        
        Args:
            results_list: List of evaluation metrics for different models
            model_names: List of model names
            
        Returns:
            Dictionary with ANOVA results
        """
        if len(results_list) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Extract accuracies
        accuracies = [r.exact_match_accuracy for r in results_list]
        
        comparison = {
            'models': model_names,
            'accuracies': accuracies,
            'best_model': model_names[np.argmax(accuracies)],
            'best_accuracy': max(accuracies),
            'worst_model': model_names[np.argmin(accuracies)],
            'worst_accuracy': min(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
        
        # Simple ranking
        sorted_indices = np.argsort(accuracies)[::-1]
        comparison['ranking'] = [model_names[i] for i in sorted_indices]
        
        return comparison
    
    def generate_report(
        self,
        results: EvaluationMetrics,
        output_path: str,
        format: str = 'json'
    ):
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation metrics to report
            output_path: Path to save the report
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._generate_json_report(results, output_path)
        elif format == 'csv':
            self._generate_csv_report(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_json_report(self, results: EvaluationMetrics, output_path: Path):
        """Generate JSON report."""
        report = {
            'metrics': results.dict(),
            'summary': {
                'accuracy_percentage': f"{results.exact_match_accuracy * 100:.2f}%",
                'success_rate': f"{results.tool_calling_success_rate * 100:.2f}%",
                'repair_efficiency': f"{results.repair_efficiency * 100:.2f}%",
                'avg_iterations': f"{results.average_iterations:.2f}"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved to {output_path}")
    
    def _generate_csv_report(self, results: EvaluationMetrics, output_path: Path):
        """Generate CSV report."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Exact Match Accuracy', f"{results.exact_match_accuracy:.4f}"])
            writer.writerow(['Tool Calling Success Rate', f"{results.tool_calling_success_rate:.4f}"])
            writer.writerow(['Repair Efficiency', f"{results.repair_efficiency:.4f}"])
            writer.writerow(['Average Iterations', f"{results.average_iterations:.2f}"])
            writer.writerow(['Total Samples', results.total_samples])
            writer.writerow(['Successful Samples', results.successful_samples])
        
        print(f"CSV report saved to {output_path}")
    
    def generate_latex_table(self, results: EvaluationMetrics) -> str:
        """
        Generate LaTeX table for paper.
        
        Args:
            results: Evaluation metrics
            
        Returns:
            LaTeX table string
        """
        latex = r"""\begin{table}[h]
\centering
\begin{tabular}{|l|r|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Exact Match Accuracy & """ + f"{results.exact_match_accuracy:.2%}" + r""" \\
Tool Calling Success Rate & """ + f"{results.tool_calling_success_rate:.2%}" + r""" \\
Repair Efficiency & """ + f"{results.repair_efficiency:.2%}" + r""" \\
Average Iterations & """ + f"{results.average_iterations:.2f}" + r""" \\
\hline
Total Samples & """ + f"{results.total_samples}" + r""" \\
Successful Samples & """ + f"{results.successful_samples}" + r""" \\
\hline
\end{tabular}
\caption{Benchmark Evaluation Results}
\label{tab:results}
\end{table}"""
        
        return latex

