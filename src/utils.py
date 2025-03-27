"""
Utility functions and helper classes for the text summarization system.

This module provides:
- Performance monitoring and timing utilities
- Caching mechanisms for intermediate results
- Batch processing helpers
- Logging configuration
"""

from pathlib import Path
from functools import wraps
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging
import time
import json
import pickle
import os
import shutil

class Timer:
    """Context manager for measuring code execution time.
    
    Example:
        with Timer("Processing documents"):
            process_documents()
    """
    
    def __init__(self, description: str = ""):
        """Initialize timer with optional description."""
        self.description = description
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        """Start timing when entering context."""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Calculate duration and print results when exiting context."""
        self.end = time.time()
        self.duration = self.end - self.start
        if self.description:
            print(f"{self.description} took {self.duration:.2f} seconds")

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """Configure logging with file and console handlers.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        format: Log message format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    formatter = logging.Formatter(format)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    
    Returns:
        Path object for the directory
    
    Raises:
        OSError: If directory creation fails
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class Cache:
    """Persistent cache for storing and retrieving objects.
    
    This class provides:
    - Pickle-based object serialization
    - Automatic directory management
    - Cache clearing functionality
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """Initialize cache with specified directory.
        
        Args:
            cache_dir: Directory for storing cached objects
        """
        self.cache_dir = ensure_directory(cache_dir)

    def save(self, obj: Any, filename: str):
        """Save object to cache using pickle serialization.
        
        Args:
            obj: Object to cache
            filename: Name for the cached file
        """
        filepath = self.cache_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, filename: str) -> Any:
        """Load object from cache.
        
        Args:
            filename: Name of the cached file
        
        Returns:
            Cached object
        
        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        filepath = self.cache_dir / filename
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def clear(self):
        """Remove all cached files."""
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True)

class ExperimentTracker:
    """Track and save experimental results."""
    
    def __init__(self, save_dir: Union[str, Path]):
        self.save_dir = ensure_directory(save_dir)
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.current_experiment = {}

    def start_experiment(self, config: Dict) -> None:
        """Start new experiment with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.current_experiment = {
            'config': config,
            'start_time': datetime.now().isoformat(),
            'metrics': {}
        }

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][name] = []
        
        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step
            
        self.current_experiment['metrics'][name].append(metric_entry)

    def end_experiment(self) -> None:
        """End current experiment and save results."""
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.results.append(self.current_experiment)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_json(
            self.current_experiment,
            self.save_dir / f'experiment_{timestamp}.json'
        )

def memory_usage_info() -> Dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
        'percent': process.memory_percent()
    }

def timer_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class BatchGenerator:
    """Iterator for processing items in batches.
    
    This class helps with:
    - Memory-efficient batch processing
    - Parallel processing support
    - Progress tracking
    """

    def __init__(self, items: List, batch_size: int):
        """Initialize batch generator.
        
        Args:
            items: List of items to process
            batch_size: Number of items per batch
        """
        self.items = items
        self.batch_size = batch_size

    def __iter__(self):
        """Yield batches of items."""
        for i in range(0, len(self.items), self.batch_size):
            yield self.items[i:i + self.batch_size]

def save_json(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object as JSON.
    
    Args:
        obj: Object to save
        filepath: Output filepath
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(filepath: Union[str, Path]) -> Any:
    """Load object from JSON.
    
    Args:
        filepath: Input filepath
    
    Returns:
        Loaded object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)