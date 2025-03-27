"""
Derivative-Free Optimization (DFO) implementation for text summarization.

This module implements a modified DFO algorithm specifically designed for extractive text summarization.
The algorithm optimizes sentence selection weights using a population-based approach with local and
global search strategies.
"""

import numpy as np
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict

@dataclass
class DFOParams:
    """Configuration parameters for the DFO algorithm.
    
    Attributes:
        population_size: Number of search agents (flies) in the population
        max_iterations: Maximum number of optimization iterations
        delta: Probability of random position reset (exploration factor)
        alpha: Weight for sentence relevance in fitness calculation
        beta: Weight for ROUGE score in fitness calculation
        gamma: Weight for length penalty in fitness calculation
        target_length_ratio: Target summary length as ratio of original document
        enable_paper_summaries: Enable detailed logging for summary generation
    """
    population_size: int = 90
    max_iterations: int = 100
    delta: float = 0.1
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 0.5
    target_length_ratio: float = 0.08
    enable_paper_summaries: bool = False

class DFOOptimizer:
    """DFO optimizer for text summarization.
    
    This class implements the core DFO algorithm with modifications for text summarization:
    - Vectorized operations for improved performance
    - Progress tracking and convergence history
    - Multi-objective fitness function combining relevance, ROUGE scores, and length
    """

    def __init__(self, 
                 params: DFOParams = DFOParams(),
                 seed: Optional[int] = None):
        """Initialize the DFO optimizer.
        
        Args:
            params: Configuration parameters for the algorithm
            seed: Random seed for reproducibility
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        self.convergence_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': []
        }
        
        if seed is not None:
            np.random.seed(seed)

    def optimize(self, document_features: np.ndarray, rouge_calculator: Callable) -> Tuple[np.ndarray, float, Dict]:
        """Execute the DFO optimization process.
        
        Args:
            document_features: Matrix of sentence features (n_sentences × n_features)
            rouge_calculator: Function to calculate ROUGE scores for a set of selected sentences
            
        Returns:
            Tuple containing:
            - best_solution: Optimal sentence selection weights
            - best_fitness: Fitness score of the best solution
            - convergence_history: Dictionary tracking optimization metrics
        """
        n_features = document_features.shape[1]
        
        # Vectorized population initialization
        population = np.random.uniform(0, 1, (self.params.population_size, n_features))
        
        best_solution = None
        best_fitness = float('-inf')

        # Track convergence history
        self.convergence_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': []
        }

        # Add progress bar for iterations
        pbar = tqdm(range(self.params.max_iterations), 
                    desc="DFO Optimization", 
                    unit="iter",
                    position=2,
                    leave=False)
        
        for iteration in range(self.params.max_iterations):
            # Vectorized fitness calculation
            fitness_scores = np.array([self._calculate_fitness(w, document_features, rouge_calculator) 
                                     for w in population])
            
            # Update best solution
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]


            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()
                
                
                # Calculate and log summary for best solution
                sentence_scores = document_features.dot(best_solution)
                n_sentences = len(sentence_scores)
                target_length = int(n_sentences * self.params.target_length_ratio)
                selected_indices = np.argsort(sentence_scores)[-target_length:]
                selected_indices.sort()

            # Track convergence metrics
            self.convergence_history['best_fitness'].append(float(best_fitness))
            self.convergence_history['avg_fitness'].append(float(np.mean(fitness_scores)))
            self.convergence_history['population_diversity'].append(
                float(np.mean(np.std(population, axis=0)))
            )    

            # Vectorized position updates
            mask = np.random.random(self.params.population_size) < self.params.delta
            mask[current_best_idx] = False  # Don't update best fly
            
            # Calculate distances matrix in one go
            distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            np.fill_diagonal(distances, np.inf)
            nearest_indices = np.argmin(distances, axis=1)
            
            # Update positions using vectorized operations
            random_positions = np.random.uniform(0, 1, (self.params.population_size, n_features))
            updates = population + np.random.random((self.params.population_size, 1)) * (
                population[current_best_idx] - population
            ) + np.random.random((self.params.population_size, 1)) * (
                population[nearest_indices] - population
            )
            
            population = np.where(mask[:, np.newaxis], random_positions, updates)
            population = np.clip(population, 0, 1)  # Ensure bounds


            # Update progress bar
            pbar.set_postfix({
                'Best Fitness': f'{best_fitness:.4f}',
                'Avg Fitness': f'{np.mean(fitness_scores):.4f}',
                'Diversity': f'{self.convergence_history["population_diversity"][-1]:.4f}'
            })
            pbar.update(1)
        
        pbar.close()
        return best_solution, best_fitness, self.convergence_history

    def _calculate_fitness(self,
                         weights: np.ndarray,
                         document_features: np.ndarray,
                         rouge_calculator: Callable) -> float:
        """Calculate the multi-objective fitness score for a solution.
        
        The fitness function combines three components:
        1. Relevance: Average feature importance of selected sentences
        2. ROUGE: Similarity between generated and reference summaries
        3. Length penalty: Penalization for deviation from target length
        
        Args:
            weights: Feature importance weights
            document_features: Matrix of sentence features
            rouge_calculator: Function to calculate ROUGE scores
            
        Returns:
            Combined fitness score
        """
        # Calculate sentence scores
        sentence_scores = document_features.dot(weights)
        
        # Get summary based on top sentences
        n_sentences = len(sentence_scores)
        target_length = int(n_sentences * self.params.target_length_ratio)
        selected_indices = np.argsort(sentence_scores)[-target_length:]
        selected_indices.sort()  # Maintain original order
        
        # Calculate components of fitness
        relevance_score = np.mean(sentence_scores[selected_indices])
        rouge_score = rouge_calculator(selected_indices)
        
        # Calculate length penalty (add safeguard)
        actual_length = len(selected_indices)
        length_diff = abs(actual_length - target_length)
        length_penalty = np.exp(-length_diff / max(target_length, 1))
        
        # Combine components using weights
        fitness = (
            self.params.alpha * relevance_score +
            self.params.beta * rouge_score -
            self.params.gamma * (1 - length_penalty)
        )
        
        return fitness

class SummaryGenerator:
    """Text summary generator using DFO optimization.
    
    This class handles the generation of extractive summaries by:
    1. Optimizing sentence selection weights using DFO
    2. Scoring sentences using the optimal weights
    3. Selecting top sentences while maintaining original order
    """

    def __init__(self, dfo_optimizer: DFOOptimizer):
        """Initialize the summary generator.
        
        Args:
            dfo_optimizer: Configured DFO optimizer instance
        """
        self.optimizer = dfo_optimizer
        self.logger = logging.getLogger(__name__)


    def generate_summary(self,
                    document_features: np.ndarray,
                    sentences: List[str],
                    rouge_calculator: Callable) -> Tuple[str, List[int], Dict]:
        """Generate an extractive summary using DFO optimization.
        
        Args:
            document_features: Matrix of sentence features
            sentences: List of original document sentences
            rouge_calculator: Function to calculate ROUGE scores
            
        Returns:
            Tuple containing:
            - summary: Generated summary text
            - selected_indices: Indices of selected sentences
            - conv_history: Optimization convergence history
        """
        
        # Get optimal weights and convergence history
        weights, fitness, conv_history = self.optimizer.optimize(
            document_features,
            rouge_calculator
        )
        
        # Score sentences
        sentence_scores = document_features.dot(weights)
        
        # Select top sentences
        n_sentences = len(sentences)
        target_length = int(n_sentences * self.optimizer.params.target_length_ratio)
        selected_indices = np.argsort(sentence_scores)[-target_length:]
        selected_indices.sort()  # Maintain original order
        
        # Generate summary
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)

        # Add detailed logging if enabled
        if hasattr(self.optimizer.params, 'enable_paper_summaries') and self.optimizer.params.enable_paper_summaries:
            self.logger.info("\nFinal Summary Generation:")
            self.logger.info(f"Summary Length: {len(selected_indices)} sentences")
            self.logger.info(f"Summary: {summary}\n")
        
        return summary, selected_indices, conv_history