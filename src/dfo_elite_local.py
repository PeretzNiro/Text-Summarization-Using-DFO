"""
Implementation of Dispersive Flies Optimization (DFO) with elitism and local search.

This module combines two enhancement strategies for the basic DFO algorithm:
1. Elitism: Preserves a percentage of the best solutions across iterations
2. Local Search: Periodically applies hill climbing to refine elite solutions

The combination provides both global exploration through DFO's dispersive behavior
and local exploitation through hill climbing, while maintaining good solutions
through elitism. This hybrid approach is particularly effective for complex
optimization landscapes.
"""

import numpy as np
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict, Union

@dataclass
class DFOParamsEliteLocal:
    """Parameters for DFO algorithm with elitism and local search.
    
    Attributes:
        population_size: Number of solutions in the population
        max_iterations: Maximum number of optimization iterations
        delta: Probability of random position reset
        alpha: Weight for relevance score in fitness calculation
        beta: Weight for ROUGE score in fitness calculation
        gamma: Weight for length penalty in fitness calculation
        target_length_ratio: Target summary length as ratio of document length
        elite_percentage: Fraction of best solutions to preserve
        local_search_frequency: How often to apply local search (iterations)
        top_k_solutions: Number of best solutions for local search
        hill_climbing_iterations: Number of improvement attempts per solution
        step_size: Size of perturbations in hill climbing
    """
    population_size: int = 90
    max_iterations: int = 100
    delta: float = 0.1
    alpha: float = 0.3
    beta: float = 0.7
    gamma: float = 0.5
    target_length_ratio: float = 0.08
    elite_percentage: float = 0.15
    local_search_frequency: int = 20
    top_k_solutions: int = 2
    hill_climbing_iterations: int = 40
    step_size: float = 0.03

class DFOOptimizerEliteLocal:
    """DFO with elitism and local search for text summarization."""
    
    def __init__(self, 
                 params: DFOParamsEliteLocal = DFOParamsEliteLocal(),
                 seed: Optional[int] = None):
        self.params = params
        self.logger = logging.getLogger(__name__)
        self.convergence_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': [],
            'elite_fitness': [],
            'local_search_improvements': []
        }
        
        if seed is not None:
            np.random.seed(seed)

    def _hill_climbing(self, 
                      solution: np.ndarray,
                      fitness: float,
                      document_features: np.ndarray,
                      rouge_calculator: Callable) -> Tuple[np.ndarray, float]:
        """Apply hill climbing to improve a solution."""
        current_solution = solution.copy()
        current_fitness = fitness
        
        for _ in range(self.params.hill_climbing_iterations):
            # Generate small random perturbations
            perturbation = np.random.normal(0, self.params.step_size, size=solution.shape)
            candidate = np.clip(current_solution + perturbation, 0, 1)
            
            # Evaluate candidate
            candidate_fitness = self._calculate_fitness(
                candidate,
                document_features,
                rouge_calculator
            )
            
            # Accept if better
            if candidate_fitness > current_fitness:
                current_solution = candidate
                current_fitness = candidate_fitness
        
        return current_solution, current_fitness

    def _apply_local_search(self,
                           population: np.ndarray,
                           fitness_scores: np.ndarray,
                           document_features: np.ndarray,
                           rouge_calculator: Callable) -> Tuple[np.ndarray, np.ndarray, float]:
        """Apply local search to top k solutions."""
        # Find top k solutions
        top_k_indices = np.argsort(fitness_scores)[-self.params.top_k_solutions:]
        initial_best_fitness = np.max(fitness_scores)
        
        # Apply hill climbing to each top solution
        for idx in top_k_indices:
            improved_solution, improved_fitness = self._hill_climbing(
                population[idx],
                fitness_scores[idx],
                document_features,
                rouge_calculator
            )
            
            # Update population if improved
            if improved_fitness > fitness_scores[idx]:
                population[idx] = improved_solution
                fitness_scores[idx] = improved_fitness
        
        # Calculate improvement
        final_best_fitness = np.max(fitness_scores)
        improvement = final_best_fitness - initial_best_fitness
        
        return population, fitness_scores, improvement

    def optimize(self, document_features: np.ndarray, rouge_calculator: Callable) -> Tuple[np.ndarray, float, Dict]:
        """Run DFO optimization with elitism and local search."""
        n_features = document_features.shape[1]
        
        # Initialize population
        population = np.random.uniform(0, 1, (self.params.population_size, n_features))
        
        # Initialize tracking variables
        best_solution = None
        best_fitness = float('-inf')
        
        # Calculate number of elite solutions to preserve
        n_elites = max(1, int(self.params.population_size * self.params.elite_percentage))
        
        # Reset convergence history
        self.convergence_history = {
            'best_fitness': [],
            'avg_fitness': [],
            'population_diversity': [],
            'elite_fitness': [],
            'local_search_improvements': []
        }

        # Progress bar
        pbar = tqdm(range(self.params.max_iterations), 
                   desc=f"DFO-EliteLocal (e={int(self.params.elite_percentage*100)}%, f={self.params.local_search_frequency})", 
                   unit="iter",
                   position=2,
                   leave=False)
        
        for iteration in range(self.params.max_iterations):
            # Calculate fitness for all solutions
            fitness_scores = np.array([
                self._calculate_fitness(w, document_features, rouge_calculator) 
                for w in population
            ])
            
            # Apply local search if needed
            local_search_improvement = 0.0
            if iteration > 0 and iteration % self.params.local_search_frequency == 0:
                population, fitness_scores, local_search_improvement = self._apply_local_search(
                    population,
                    fitness_scores,
                    document_features,
                    rouge_calculator
                )
            
            # Find and update best solution
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_idx].copy()
            
            # Select elite solutions
            elite_indices = np.argsort(fitness_scores)[-n_elites:]
            elite_solutions = population[elite_indices].copy()
            elite_fitness_scores = fitness_scores[elite_indices]
            
            # Update convergence history
            self.convergence_history['best_fitness'].append(float(best_fitness))
            self.convergence_history['avg_fitness'].append(float(np.mean(fitness_scores)))
            self.convergence_history['population_diversity'].append(
                float(np.mean(np.std(population, axis=0)))
            )
            self.convergence_history['elite_fitness'].append(
                float(np.mean(elite_fitness_scores))
            )
            self.convergence_history['local_search_improvements'].append(
                float(local_search_improvement)
            )
            
            # Calculate distances for nearest neighbor
            distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
            np.fill_diagonal(distances, np.inf)  # Exclude self
            nearest_indices = np.argmin(distances, axis=1)
            
            # Create update mask (excluding elite solutions)
            mask = np.random.random(self.params.population_size) < self.params.delta
            mask[elite_indices] = False  # Protect elite solutions
            
            # Update positions of non-elite solutions
            random_positions = np.random.uniform(0, 1, (self.params.population_size, n_features))
            updates = population + np.random.random((self.params.population_size, 1)) * (
                population[current_best_idx] - population
            ) + np.random.random((self.params.population_size, 1)) * (
                population[nearest_indices] - population
            )
            
            population = np.where(mask[:, np.newaxis], random_positions, updates)
            population = np.clip(population, 0, 1)  # Ensure bounds
            
            # Restore elite solutions
            population[elite_indices] = elite_solutions

            # Update progress bar
            pbar.set_postfix({
                'Best Fitness': f'{best_fitness:.4f}',
                'Avg Fitness': f'{np.mean(fitness_scores):.4f}',
                'Elite Fitness': f'{np.mean(elite_fitness_scores):.4f}',
                'LS Improvement': f'{local_search_improvement:.4f}'
            })
            pbar.update(1)
        
        pbar.close()
        return best_solution, best_fitness, self.convergence_history

    def _calculate_fitness(self,
                         weights: np.ndarray,
                         document_features: np.ndarray,
                         rouge_calculator: Callable) -> float:
        """Calculate fitness score for a solution."""
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
        
        # Calculate length penalty
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
    """Generate summaries using DFO-optimized weights with elitism and local search."""
    
    def __init__(self, dfo_optimizer: DFOOptimizerEliteLocal):
        self.optimizer = dfo_optimizer
        self.logger = logging.getLogger(__name__)

    def generate_summary(self,
                        document_features: np.ndarray,
                        sentences: List[str],
                        rouge_calculator: Callable) -> Tuple[str, List[int], Dict]:
        """Generate summary using DFO optimization with elitism and local search."""
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
        
        # Add detailed logging
        self.logger.info("\nFinal Summary Generation:")
        self.logger.info(f"Elite Percentage: {self.optimizer.params.elite_percentage*100}%")
        self.logger.info(f"Local Search Frequency: {self.optimizer.params.local_search_frequency}")
        self.logger.info(f"Summary Length: {len(selected_indices)} sentences")
        self.logger.info(f"Summary: {summary}\n")
        
        return summary, selected_indices, conv_history