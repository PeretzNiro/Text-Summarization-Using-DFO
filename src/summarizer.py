"""
Core text summarization implementation using DFO optimization.

This module provides the main summarization interface, combining:
- Document preprocessing and feature extraction
- DFO-based optimization for sentence selection
- Summary generation and ROUGE evaluation
- Caching mechanisms for improved performance
"""

from typing import List, Dict, Tuple, Optional, Type
from dataclasses import dataclass
from pathlib import Path
from .preprocessor import Document, DocumentBatchProcessor
from .dfo import DFOOptimizer, DFOParams, SummaryGenerator
from .dfo_elite_local import DFOOptimizerEliteLocal, DFOParamsEliteLocal
from .evaluator import SummaryEvaluator
import numpy as np
import logging

# Map of algorithm names to their optimizer classes
DFO_ALGORITHMS = {
    'dfo': DFOOptimizer,
    'dfo_elite_local': DFOOptimizerEliteLocal
}

@dataclass
class SummarizerConfig:
    """Configuration settings for the summarization pipeline.
    
    Attributes:
        algorithm: Name of the DFO algorithm variant to use
        dfo_params: Parameters for the DFO optimization algorithm
        batch_size: Number of documents to process in parallel
        n_jobs: Number of parallel jobs (-1 for all available cores)
        cache_features: Whether to cache document features
        cache_dir: Directory for storing cached features
    """
    algorithm: str = 'dfo_elite_local_momentum'
    dfo_params: DFOParams = DFOParams()
    batch_size: int = 32
    n_jobs: int = -1
    cache_features: bool = True
    cache_dir: Optional[str] = None
    
    def validate(self) -> Optional[str]:
        """Validate configuration settings.
        
        Returns:
            str: Error message if validation fails, None otherwise
        """
        if self.algorithm not in DFO_ALGORITHMS:
            return f"Invalid algorithm: {self.algorithm}. Available algorithms: {list(DFO_ALGORITHMS.keys())}"
        return None

class TextSummarizer:
    """Main summarizer class combining preprocessing, DFO, and evaluation.
    
    This class orchestrates the complete summarization pipeline:
    1. Document preprocessing and feature extraction
    2. DFO-based optimization for sentence selection
    3. Summary generation and ROUGE evaluation
    4. Result caching and performance optimization
    """
    
    def __init__(self, 
                 config: SummarizerConfig = SummarizerConfig(),
                 processor: Optional[DocumentBatchProcessor] = None):
        """Initialize the summarization pipeline.
        
        Args:
            config: Configuration settings for the summarizer
            processor: Optional pre-configured document processor
            
        Raises:
            ValueError: If configuration validation fails
        """
        # Validate configuration
        error = config.validate()
        if error:
            raise ValueError(error)
            
        self.config = config
        
        # Initialize document processor if not provided
        self.processor = processor or DocumentBatchProcessor(
            batch_size=config.batch_size,
            n_jobs=config.n_jobs
        )
        
        # Initialize the specified DFO optimizer
        optimizer_class = DFO_ALGORITHMS[config.algorithm]
        self.optimizer = optimizer_class(params=config.dfo_params)
        
        # Initialize summary generator with the chosen optimizer
        self.generator = SummaryGenerator(self.optimizer)
        
        # Initialize evaluator
        self.evaluator = SummaryEvaluator()
        
        logging.info(f"Initialized TextSummarizer with {config.algorithm} algorithm")

    def _clear_rouge_cache(self):
        """Clear the ROUGE score cache to free memory."""
        if hasattr(self, 'rouge_cache'):
            self.rouge_cache.clear()
        if hasattr(self, 'current_doc_id'):
            self.current_doc_id = None
        
    def summarize_document(self, document: Document) -> Tuple[str, Dict[str, float], Dict]:
        """Generate and evaluate a summary for a single document.
        
        This method:
        1. Preprocesses the document if needed
        2. Extracts document features
        3. Optimizes sentence selection using DFO
        4. Generates and evaluates the summary
        
        Args:
            document: Preprocessed or raw document to summarize
            
        Returns:
            Tuple containing:
            - summary: Generated summary text
            - scores: Dictionary of evaluation metrics
            - history: Optimization convergence history
            
        Raises:
            ValueError: If document preprocessing fails
        """
        try:
            # Ensure document is preprocessed
            if not document.features:
                self.processor.process_single_document(document)

            # Create ROUGE calculator for this document
            def rouge_calculator(selected_indices: List[int]) -> float:
                summary = ' '.join([document.sentences[i] for i in selected_indices])
                scores = self.evaluator.evaluate_summary(document.highlights, summary)
                f1_scores = [metric_scores['f1'] for metric_scores in scores.values()]
                return np.mean(f1_scores)

            # Generate summary
            summary, selected_indices, conv_history = self.generator.generate_summary(
                document.get_feature_vector(),
                document.sentences,
                rouge_calculator
            )

            # Evaluate summary
            scores = self.evaluator.evaluate_summary(document.highlights, summary)
            return summary, scores, conv_history

        except Exception as e:
            logging.error(f"Error summarizing document {document.id}: {str(e)}")
            return "", {}, {}


    def summarize_batch(self, documents: List[Document]) -> List[Tuple[str, Dict[str, float], Dict]]:
        """Generate summaries for a batch of documents."""
        try:
            processed_docs = self.processor.process_batch(documents)
            results = []
            
            for doc in processed_docs:
                try:
                    # Clear cache if processing a new document
                    if hasattr(self, 'current_doc_id') and self.current_doc_id != doc.id:
                        self._clear_rouge_cache()
                        self.current_doc_id = doc.id
                    
                    if not doc.features:
                        self.processor.process_single_document(doc)

                    # Create ROUGE calculator with caching
                    def rouge_calculator(selected_indices: List[int]) -> float:
                        if not hasattr(self, 'rouge_cache'):
                            self.rouge_cache = {}
                            
                        cache_key = tuple(sorted(selected_indices))
                        if cache_key in self.rouge_cache:
                            return self.rouge_cache[cache_key]
                        
                        summary = ' '.join([doc.sentences[i] for i in selected_indices])
                        scores = self.evaluator.evaluate_summary(doc.highlights, summary)
                        f1_scores = [metric_scores['f1'] for metric_scores in scores.values()]
                        mean_score = np.mean(f1_scores)
                        
                        self.rouge_cache[cache_key] = mean_score
                        return mean_score

                    # Generate summary using cached ROUGE calculator
                    summary, selected_indices, conv_history = self.generator.generate_summary(
                        doc.get_feature_vector(),
                        doc.sentences,
                        rouge_calculator
                    )
                    
                    # Evaluate final summary
                    scores = self.evaluator.evaluate_summary(doc.highlights, summary)
                    results.append((summary, scores, conv_history))
                    
                except Exception as e:
                    logging.error(f"Error processing document {doc.id}: {str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            return []