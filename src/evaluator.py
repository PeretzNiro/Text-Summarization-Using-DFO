"""
Text summarization evaluation module using ROUGE metrics.

This module provides:
- ROUGE score calculation (ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-L)
- Summary evaluation with multiple metrics
- Batch evaluation capabilities
- Score aggregation and statistics
"""

from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class EvaluationConfig:
    """Configuration for ROUGE evaluation metrics.
    
    Attributes:
        use_stemming: Whether to apply stemming to words
        rouge_types: List of ROUGE metrics to compute
    """
    use_stemming: bool = True
    rouge_types: List[str] = None
    
    def __post_init__(self):
        """Initialize default ROUGE types if none specified."""
        if self.rouge_types is None:
            self.rouge_types = ['rouge1', 'rouge2', 'rouge3', 'rougeL']

class SummaryEvaluator:
    """Evaluator for text summarization using ROUGE metrics.
    
    Features:
    - Multiple ROUGE metric computation
    - Batch evaluation support
    - Score aggregation and statistics
    - Progress tracking
    """
    
    def __init__(self, config: EvaluationConfig = EvaluationConfig()):
        """Initialize the evaluator.
        
        Args:
            config: Evaluation configuration settings
        """
        self.config = config
        self.scorer = rouge_scorer.RougeScorer(
            self.config.rouge_types,
            use_stemmer=self.config.use_stemming
        )
        
    def evaluate_summary(self, 
                        reference: str, 
                        candidate: str) -> Dict[str, Dict[str, float]]:
        """Evaluate a single summary using ROUGE metrics.
        
        Args:
            reference: Reference (gold) summary text
            candidate: Generated summary to evaluate
            
        Returns:
            Dictionary containing ROUGE scores:
            - Keys: ROUGE metric types
            - Values: Dict with precision, recall, and F1 scores
            
        Example:
            {
                'rouge1': {'precision': 0.5, 'recall': 0.6, 'f1': 0.55},
                'rouge2': {'precision': 0.4, 'recall': 0.5, 'f1': 0.45},
                ...
            }
        """
        try:
            scores = self.scorer.score(reference, candidate)
            
            # Convert RougeScore objects to dictionary
            results = {}
            for rouge_type, score in scores.items():
                results[rouge_type] = {
                    'precision': score.precision,
                    'recall': score.recall,
                    'f1': score.fmeasure
                }
            
            return results
            
        except Exception as e:
            raise

    def evaluate_batch(self, 
                      references: List[str], 
                      candidates: List[str]
                      ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Dict[str, float]]]]:
        """Evaluate a batch of summaries."""
        try:
            individual_scores = [
                self.evaluate_summary(ref, cand)
                for ref, cand in zip(references, candidates)
            ]
            
            # Calculate average scores
            avg_scores = defaultdict(lambda: defaultdict(float))
            n_samples = len(individual_scores)
            
            for scores in individual_scores:
                for rouge_type, metrics in scores.items():
                    for metric_name, value in metrics.items():
                        avg_scores[rouge_type][metric_name] += value / n_samples
            
            return dict(avg_scores), individual_scores
            
        except Exception as e:
            raise

    def evaluate_corpus(self,
                       documents: List[Tuple[str, str]],
                       batch_size: int = 32) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate entire corpus with statistical analysis.
        
        Args:
            documents: List of (reference, candidate) summary pairs
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary containing average scores and statistics
        """
        try:
            all_scores = []
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                references, candidates = zip(*batch)
                _, batch_scores = self.evaluate_batch(references, candidates)
                all_scores.extend(batch_scores)
            
            # Calculate statistics
            stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            for rouge_type in self.config.rouge_types:
                for metric in ['precision', 'recall', 'f1']:
                    values = [
                        scores[rouge_type][metric]
                        for scores in all_scores
                    ]
                    
                    stats[rouge_type][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            
            return dict(stats)
            
        except Exception as e:
            raise

    def calculate_statistics(self, 
                           scores: List[Dict[str, Dict[str, float]]],
                           confidence_level: float = 0.95) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate detailed statistics for a set of scores.
        
        Args:
            scores: List of score dictionaries
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary of statistical measures
        """
        try:
            stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            for rouge_type in self.config.rouge_types:
                for metric in ['precision', 'recall', 'f1']:
                    values = [
                        scores[i][rouge_type][metric]
                        for i in range(len(scores))
                    ]
                    
                    # Basic statistics
                    stats[rouge_type][metric].update({
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    })
                    
                    # Confidence intervals
                    if len(values) > 1:
                        from scipy import stats as scipy_stats
                        ci = scipy_stats.t.interval(
                            confidence_level,
                            len(values) - 1,
                            loc=np.mean(values),
                            scale=scipy_stats.sem(values)
                        )
                        stats[rouge_type][metric].update({
                            'ci_lower': ci[0],
                            'ci_upper': ci[1]
                        })
            
            return dict(stats)
            
        except Exception as e:
            raise

    def generate_report(self,
                       scores: Dict[str, Dict[str, Dict[str, float]]],
                       output_format: str = 'text') -> str:
        """Generate evaluation report in specified format.
        
        Args:
            scores: Evaluation scores dictionary
            output_format: Output format ('text' or 'markdown')
            
        Returns:
            Formatted report string
        """
        try:
            if output_format == 'text':
                return self._generate_text_report(scores)
            elif output_format == 'markdown':
                return self._generate_markdown_report(scores)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            raise

    def _generate_text_report(self, scores: Dict) -> str:
        """Generate plain text report."""
        lines = ['Evaluation Report', '================']
        
        for rouge_type in sorted(scores.keys()):
            lines.append(f"\n{rouge_type.upper()}")
            lines.append('-' * len(rouge_type))
            
            for metric, stats in scores[rouge_type].items():
                lines.append(f"\n{metric}:")
                for stat_name, value in stats.items():
                    lines.append(f"  {stat_name}: {value:.4f}")
        
        return '\n'.join(lines)

    def _generate_markdown_report(self, scores: Dict) -> str:
        """Generate markdown report."""
        lines = ['# Evaluation Report']
        
        for rouge_type in sorted(scores.keys()):
            lines.append(f"\n## {rouge_type.upper()}")
            
            # Create table header
            lines.append("\n| Metric | Value |")
            lines.append("|--------|-------|")
            
            for metric, stats in scores[rouge_type].items():
                for stat_name, value in stats.items():
                    lines.append(f"| {metric}/{stat_name} | {value:.4f} |")
        
        return '\n'.join(lines)