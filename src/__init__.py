"""
Text Summarization Package using DFO optimization.
This package contains modules for preprocessing, summarization, and evaluation.
"""

# Import main components for easier access
from .preprocessor import CNNDMDataLoader, DocumentBatchProcessor, Document
from .summarizer import TextSummarizer, SummarizerConfig
from .evaluator import SummaryEvaluator, EvaluationConfig
from .utils import setup_logging, Timer, Cache, ExperimentTracker

# Import all DFO variants with consistent naming
from .dfo import DFOOptimizer, DFOParams, SummaryGenerator
from .dfo_elite_local import DFOOptimizerEliteLocal, DFOParamsEliteLocal
# Version info
__version__ = '0.1.0'

# Export all DFO variants and their parameters
__all__ = [
    # Core components
    'CNNDMDataLoader', 'DocumentBatchProcessor', 'Document',
    'TextSummarizer', 'SummarizerConfig',
    'SummaryEvaluator', 'EvaluationConfig',
    'setup_logging', 'Timer', 'Cache', 'ExperimentTracker',
    'SummaryGenerator',
    
    # DFO variants
    'DFOOptimizer', 'DFOParams',
    'DFOOptimizerEliteLocal', 'DFOParamsEliteLocal'
]