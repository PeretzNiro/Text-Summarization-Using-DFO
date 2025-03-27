"""
Main entry point for the text summarization system.
This module orchestrates the document processing pipeline, including dataset preparation,
batch processing, and results evaluation. It implements a configurable summarization
workflow using DFO-based optimization techniques.
"""

import argparse
import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import pickle
from datasets import load_dataset, config
from tqdm import tqdm
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import (
    CNNDMDataLoader,
    DocumentBatchProcessor,
    TextSummarizer,
    SummarizerConfig,
    setup_logging,
    # DFO variants
    DFOParams,
    DFOOptimizer,
    DFOParamsEliteLocal,
    DFOOptimizerEliteLocal,
    Document
)

# Add timeout configuration here
config.HF_DATASETS_TIMEOUT = 60
config.HF_DATASETS_HTTP_MAX_RETRIES = 5

# ============================================================================
# Configuration Section
# ============================================================================

# DFO algorithm variants with their parameter classes
DFO_VARIANTS = {
    'dfo': (DFOParams, DFOOptimizer),
    'dfo_elite_local': (DFOParamsEliteLocal, DFOOptimizerEliteLocal)
}

# Available DFO algorithms with descriptions
AVAILABLE_ALGORITHMS = {
    'dfo': 'Basic DFO algorithm',
    'dfo_elite_local': 'DFO with elite strategy and local search'
}

# User Configuration
class Config:
    # Algorithm selection
    algorithm_to_run: str = 'dfo'  # Default algorithm
    
    # Logging configuration
    enable_paper_summaries: bool = False  # Toggle for paper summary logging
    
    # Dataset parameters
    num_papers: int = 20  # Number of papers to process
    batch_size: int = 10   # Batch size for processing
    
    # DFO parameters will be loaded from the respective algorithm's default values
    dfo_params = None  # Will be set based on selected algorithm
    
    @classmethod
    def validate(cls) -> Optional[str]:
        """Validate configuration settings.
        
        Returns:
            str: Error message if validation fails, None otherwise
        """
        if cls.algorithm_to_run not in AVAILABLE_ALGORITHMS:
            return f"Invalid algorithm: {cls.algorithm_to_run}. Available algorithms: {list(AVAILABLE_ALGORITHMS.keys())}"
        
        if cls.num_papers < 1:
            return f"Number of papers must be positive, got: {cls.num_papers}"
            
        if cls.batch_size < 1:
            return f"Batch size must be positive, got: {cls.batch_size}"
            
        if cls.batch_size > cls.num_papers:
            return f"Batch size ({cls.batch_size}) cannot be larger than number of papers ({cls.num_papers})"
        
        # Load default parameters for the selected algorithm
        params_class = DFO_VARIANTS[cls.algorithm_to_run][0]
        cls.dfo_params = params_class()
        cls.dfo_params.enable_paper_summaries = cls.enable_paper_summaries
        
        return None

def download_and_prepare_dataset(
    output_dir: str,
    dataset_name: str = "abisee/cnn_dailymail",
    dataset_version: str = "3.0.0",
    dataset_split: str = "test",
    num_articles: int = Config.num_papers
) -> str:
    """Download and prepare the dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
        dataset_name: Name of the dataset to download
        dataset_version: Version of the dataset
        dataset_split: Dataset split to use (train, validation, test)
        num_articles: Number of articles to process
        
    Returns:
        Path to the processed dataset file
    """
    logging.info(f"Downloading {dataset_name} ({dataset_version}) - {dataset_split} split")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        dataset = load_dataset(dataset_name, dataset_version, split=dataset_split)
        
        # Take the first num_articles
        articles = list(dataset.select(range(num_articles)))
        
        # Save processed dataset
        output_file = output_dir / f"{dataset_name.split('/')[-1]}_{dataset_split}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(articles, f)
            
        logging.info(f"Dataset saved to {output_file}")
        return str(output_file)
        
    except Exception as e:
        logging.error(f"Error downloading dataset: {str(e)}")
        raise

def process_dataset(
    data_path: str,
    batch_size: int = 32,
    n_jobs: int = -1
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Process the dataset and generate summaries.
    
    Args:
        data_path: Path to the preprocessed dataset
        batch_size: Number of documents to process in parallel
        n_jobs: Number of parallel jobs (-1 for all available cores)
        
    Returns:
        Tuple containing:
        - results: List of dictionaries containing evaluation metrics
        - convergence_history: Optimization history
    """
    logging.info(f"Processing dataset from {data_path}")
    
    try:
        # Load preprocessed dataset
        with open(Path(data_path), 'rb') as f:
            raw_documents = pickle.load(f)
            
        logging.info(f"Loaded {len(raw_documents)} documents")
        
        # Convert dictionaries to Document objects
        documents = []
        for idx, doc_dict in enumerate(raw_documents):
            doc = Document(
                article_id=f"doc_{idx}",
                article=doc_dict['article'],
                highlights=doc_dict['highlights']
            )
            documents.append(doc)
        
        # Create cache directory if it doesn't exist
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize summarizer configuration
        summarizer_config = SummarizerConfig(
            algorithm=Config.algorithm_to_run,
            dfo_params=Config.dfo_params,
            batch_size=batch_size,
            n_jobs=n_jobs,
            cache_features=True,
            cache_dir=str(cache_dir)
        )
        
        # Initialize summarizer with configuration
        summarizer = TextSummarizer(config=summarizer_config)
        
        # Process documents in batches
        results = []
        convergence_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': []
        }
        
        for batch_start in tqdm(range(0, len(documents), batch_size)):
            batch = documents[batch_start:batch_start + batch_size]
            batch_results = summarizer.summarize_batch(batch)
            
            # Extract summaries, scores, and history from batch results
            for summary, scores, history in batch_results:
                results.append({
                    'summary': summary,
                    'scores': scores
                })
                
                # Aggregate convergence history
                for key in convergence_history:
                    if key in history:
                        convergence_history[key].extend(history[key])
        
        return results, convergence_history
        
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        raise

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy data types.
    
    This encoder extends the default JSON encoder to properly serialize NumPy
    integers, floating-point numbers, and arrays into JSON-compatible formats.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_results_to_json(
    results: List[Dict[str, Any]],
    convergence_history: Dict[str, List[float]],
    output_dir: str
) -> None:
    """Save results and convergence history to JSON files.
    
    Args:
        results: List of dictionaries containing evaluation metrics
        convergence_history: Dictionary of optimization history metrics
        output_dir: Directory to save results
    """
    try:
        # Prepare summary data
        summary_data = {
            'config': {
                'algorithm': Config.algorithm_to_run,
                'batch_size': Config.batch_size,
                'num_documents': len(results)
            },
            'metrics': {
                'rouge1': {'precision': [], 'recall': [], 'f1': []},
                'rouge2': {'precision': [], 'recall': [], 'f1': []},
                'rouge3': {'precision': [], 'recall': [], 'f1': []},
                'rougeL': {'precision': [], 'recall': [], 'f1': []}
            },
            'convergence_history': convergence_history
        }
        
        # Aggregate ROUGE scores
        for result in results:
            if result and 'scores' in result:
                scores = result['scores']
                for rouge_type in ['rouge1', 'rouge2', 'rouge3', 'rougeL']:
                    if rouge_type in scores:
                        for metric in ['precision', 'recall', 'f1']:
                            if metric in scores[rouge_type]:
                                summary_data['metrics'][rouge_type][metric].append(
                                    scores[rouge_type][metric]
                                )
        
        # Calculate averages
        for rouge_type in summary_data['metrics']:
            for metric in summary_data['metrics'][rouge_type]:
                values = summary_data['metrics'][rouge_type][metric]
                if values:
                    summary_data['metrics'][rouge_type][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
                else:
                    summary_data['metrics'][rouge_type][metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
        
        # Save results
        output_file = Path(output_dir) / 'results.json'
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
            
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def display_results_summary(results: List[Dict[str, Any]], convergence_history: Dict[str, List[float]], execution_time: float):
    """Display a formatted summary of the results."""
    
    # Clear any remaining progress bars
    print("\033[2J\033[H")  # Clear screen and move cursor to top
    
    # Calculate hours, minutes, seconds from execution time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create a visual separator
    separator = "=" * 80
    subseparator = "-" * 80
    
    print("\n" + separator)
    print(" SUMMARIZATION RESULTS SUMMARY")
    print(separator + "\n")
    
    # Time Information
    print(" Execution Time:")
    print(f"   Total Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(subseparator)
    
    # Check if we have any results
    if not results:
        print("\nNo results available. All documents failed to process.")
        print(subseparator + "\n")
        return
        
    # ROUGE Scores
    print("\n ROUGE Scores:")
    rouge_types = ['rouge1', 'rouge2', 'rouge3', 'rougeL']
    metrics = ['precision', 'recall', 'f1']
    
    for rouge_type in rouge_types:
        print(f"\n {rouge_type.upper()}:")
        for metric in metrics:
            scores = [
                result['scores'][rouge_type][metric]
                for result in results
                if 'scores' in result 
                and rouge_type in result['scores']
                and metric in result['scores'][rouge_type]
            ]
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"   {metric.title():9}: {mean_score:.4f} ± {std_score:.4f}")
    print(subseparator)
    
    # Optimization Performance
    if convergence_history and all(key in convergence_history for key in ['best_fitness', 'avg_fitness']):
        print("\n Optimization Performance:")
        final_best = convergence_history['best_fitness'][-1]
        final_avg = convergence_history['avg_fitness'][-1]
        iterations = len(convergence_history['best_fitness'])
        
        print(f"   Final Best Fitness: {final_best:.4f}")
        print(f"   Final Avg Fitness:  {final_avg:.4f}")
        print(f"   Total Iterations:   {iterations}")
    print(subseparator)
    
    # Overall Metrics
    print("\n Overall Performance:")
    metrics = ['precision', 'recall', 'f1']
    for metric in metrics:
        scores = []
        for rt in rouge_types:
            rt_scores = [
                result['scores'][rt][metric]
                for result in results
                if 'scores' in result 
                and rt in result['scores']
                and metric in result['scores'][rt]
            ]
            if rt_scores:
                scores.append(np.mean(rt_scores))
                
        if scores:
            avg_score = np.mean(scores)
            print(f"   Average {metric.title():9}: {avg_score:.4f}")
            
    print(subseparator + "\n")

def main():
    """Main execution function."""
    # Validate configuration
    error = Config.validate()
    if error:
        logging.error(f"Configuration error: {error}")
        sys.exit(1)
        
    # Setup logging
    setup_logging()
    logging.info(f"Starting summarization with algorithm: {Config.algorithm_to_run}")
    logging.info(f"Processing {Config.num_papers} papers with batch size {Config.batch_size}")
    
    start_time = time.time()
    
    try:
        # Prepare output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and prepare dataset
        data_path = download_and_prepare_dataset(
            output_dir=str(output_dir),
            num_articles=Config.num_papers
        )
        
        # Process dataset
        results, convergence_history = process_dataset(
            data_path=data_path,
            batch_size=Config.batch_size,
            n_jobs=-1  # Use all available cores
        )
        
        # Save results
        save_results_to_json(
            results=results,
            convergence_history=convergence_history,
            output_dir=str(output_dir)
        )
        
        # Display results
        execution_time = time.time() - start_time
        display_results_summary(results, convergence_history, execution_time)
        
    except Exception as e:
        logging.error(f"An error occurred during execution: {str(e)}")
        raise
        
    logging.info("Execution completed successfully")

if __name__ == "__main__":
    main()