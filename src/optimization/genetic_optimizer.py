"""
Genetic Algorithm Optimizer

Implements genetic algorithm for parameter optimization using evolutionary
strategies to efficiently explore large parameter spaces.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import asyncio
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
import random

from .base_optimizer import BaseOptimizer, OptimizationConfig, OptimizationResult
from .parameter_space import ParameterSpace
from .objectives import OptimizationObjective

logger = structlog.get_logger()

class Individual:
    """Represents an individual in the genetic algorithm population"""
    
    def __init__(self, parameters: Dict[str, Any], fitness: float = float('-inf')):
        self.parameters = parameters
        self.fitness = fitness
        self.age = 0
        self.result: Optional[OptimizationResult] = None
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, age={self.age})"

class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for parameter optimization.
    
    Features:
    - Tournament selection
    - Adaptive crossover and mutation
    - Elitism preservation
    - Diversity maintenance
    - Population aging
    - Multi-point crossover
    - Self-adaptive parameters
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective: OptimizationObjective,
        config: OptimizationConfig = None,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        tournament_size: int = 3,
        adaptive_rates: bool = True,
        diversity_threshold: float = 0.1,
        max_age: int = 20
    ):
        """
        Initialize genetic algorithm optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objective: Optimization objective function
            config: Optimization configuration
            population_size: Size of the population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_ratio: Ratio of elite individuals to preserve
            tournament_size: Size of tournament selection
            adaptive_rates: Whether to adapt crossover/mutation rates
            diversity_threshold: Minimum diversity threshold
            max_age: Maximum age for individuals
        """
        super().__init__(parameter_space, objective, config)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        self.adaptive_rates = adaptive_rates
        self.diversity_threshold = diversity_threshold
        self.max_age = max_age
        
        # GA state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.current_crossover_rate = crossover_rate
        self.current_mutation_rate = mutation_rate
        
    async def _optimize_impl(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Implementation of genetic algorithm optimization"""
        
        logger.info("Starting genetic algorithm optimization",
                   population_size=self.population_size,
                   generations=self.generations,
                   crossover_rate=self.crossover_rate,
                   mutation_rate=self.mutation_rate)
        
        # Initialize population
        await self._initialize_population(strategy_factory, historical_data, start_date, end_date)
        
        best_individual = max(self.population, key=lambda x: x.fitness)
        
        # Evolution loop
        for generation in range(self.generations):
            if self.state.is_cancelled:
                logger.info("Genetic algorithm cancelled by user")
                break
            
            self.generation = generation
            
            logger.info("Processing generation",
                       generation=generation + 1,
                       best_fitness=best_individual.fitness,
                       avg_fitness=np.mean([ind.fitness for ind in self.population]))
            
            # Create new generation
            new_population = await self._create_new_generation(
                strategy_factory, historical_data, start_date, end_date
            )
            
            # Replace population
            self.population = new_population
            
            # Update best individual
            generation_best = max(self.population, key=lambda x: x.fitness)
            if generation_best.fitness > best_individual.fitness:
                best_individual = generation_best
                logger.info("New best individual found",
                           generation=generation + 1,
                           fitness=generation_best.fitness)
            
            # Track statistics
            self.best_fitness_history.append(best_individual.fitness)
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # Adaptive parameter adjustment
            if self.adaptive_rates:
                self._adapt_parameters(diversity, generation)
            
            # Update progress
            self.state.iteration = (generation + 1) * self.population_size
            self._update_progress()
            
            # Check convergence
            if self._check_convergence():
                logger.info("Convergence detected, stopping evolution",
                           generation=generation + 1)
                break
        
        logger.info("Genetic algorithm completed",
                   final_generation=self.generation + 1,
                   best_fitness=best_individual.fitness,
                   total_evaluations=len(self.all_results))
        
        return best_individual.result
    
    async def _initialize_population(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ):
        """Initialize the population with random individuals"""
        
        logger.info("Initializing population", size=self.population_size)
        
        # Create initial population
        initialization_tasks = []
        
        for i in range(self.population_size):
            parameters = self.parameter_space.sample_parameters()
            task = self._evaluate_individual(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            initialization_tasks.append(task)
        
        # Evaluate initial population in parallel
        individuals = await asyncio.gather(*initialization_tasks)
        self.population = [ind for ind in individuals if ind is not None]
        
        # Ensure we have a valid population
        while len(self.population) < self.population_size // 2:
            parameters = self.parameter_space.sample_parameters()
            individual = await self._evaluate_individual(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            if individual:
                self.population.append(individual)
        
        logger.info("Population initialized",
                   valid_individuals=len(self.population),
                   avg_fitness=np.mean([ind.fitness for ind in self.population]))
    
    async def _create_new_generation(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[Individual]:
        """Create new generation through selection, crossover, and mutation"""
        
        # Age population
        for individual in self.population:
            individual.age += 1
        
        # Remove old individuals
        self.population = [ind for ind in self.population if ind.age <= self.max_age]
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Calculate number of elites
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        
        # Preserve elites
        new_population = self.population[:num_elites].copy()
        for elite in new_population:
            elite.age = 0  # Reset age for elites
        
        # Generate offspring
        offspring_tasks = []
        offspring_needed = self.population_size - len(new_population)
        
        for _ in range(offspring_needed):
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.current_crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child_params = parent1.parameters.copy()
            
            # Mutation
            if random.random() < self.current_mutation_rate:
                child_params = self._mutate(child_params)
            
            # Evaluate offspring
            task = self._evaluate_individual(
                child_params, strategy_factory, historical_data, start_date, end_date
            )
            offspring_tasks.append(task)
        
        # Evaluate offspring in parallel
        offspring = await asyncio.gather(*offspring_tasks)
        valid_offspring = [child for child in offspring if child is not None]
        
        # Add valid offspring to new population
        new_population.extend(valid_offspring)
        
        # If we don't have enough individuals, fill with random ones
        while len(new_population) < self.population_size:
            parameters = self.parameter_space.sample_parameters()
            individual = await self._evaluate_individual(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            if individual:
                new_population.append(individual)
        
        # Ensure population size
        if len(new_population) > self.population_size:
            new_population.sort(key=lambda x: x.fitness, reverse=True)
            new_population = new_population[:self.population_size]
        
        return new_population
    
    async def _evaluate_individual(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Individual]:
        """Evaluate a single individual"""
        
        try:
            result = await self.evaluate_parameters(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            
            if result.is_valid:
                individual = Individual(parameters, result.objective_value)
                individual.result = result
                return individual
            
        except Exception as e:
            logger.warning("Individual evaluation failed", error=str(e))
        
        return None
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-point crossover"""
        child_params = {}
        
        for param_name in parent1_params.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1_params[param_name]
            else:
                child_params[param_name] = parent2_params[param_name]
        
        # Ensure constraints are satisfied
        if not self.parameter_space.validate_parameters(child_params):
            # If invalid, return one of the parents
            return parent1_params.copy()
        
        return child_params
    
    def _mutate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Gaussian mutation with adaptive strength"""
        mutated_params = parameters.copy()
        
        for param_name, param_range in self.parameter_space.parameters.items():
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                current_value = mutated_params[param_name]
                
                if hasattr(param_range, 'min_value') and hasattr(param_range, 'max_value'):
                    # Numeric parameter
                    if isinstance(current_value, int):
                        # Integer mutation
                        range_size = param_range.max_value - param_range.min_value
                        mutation_strength = max(1, int(range_size * 0.1))
                        delta = random.randint(-mutation_strength, mutation_strength)
                        new_value = max(param_range.min_value, 
                                      min(param_range.max_value, current_value + delta))
                        mutated_params[param_name] = new_value
                    else:
                        # Float mutation
                        range_size = param_range.max_value - param_range.min_value
                        mutation_strength = range_size * 0.1
                        delta = random.gauss(0, mutation_strength)
                        new_value = max(param_range.min_value,
                                      min(param_range.max_value, current_value + delta))
                        mutated_params[param_name] = new_value
                else:
                    # Discrete parameter (choice or boolean)
                    new_value = param_range.sample()
                    mutated_params[param_name] = new_value
        
        # Validate mutated parameters
        if not self.parameter_space.validate_parameters(mutated_params):
            return parameters  # Return original if mutation creates invalid parameters
        
        return mutated_params
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 1.0
        
        # Calculate pairwise parameter distances
        distances = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._parameter_distance(
                    self.population[i].parameters,
                    self.population[j].parameters
                )
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _parameter_distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate normalized distance between parameter sets"""
        distances = []
        
        for param_name in params1.keys():
            val1 = params1[param_name]
            val2 = params2[param_name]
            
            param_range = self.parameter_space.parameters[param_name]
            
            if hasattr(param_range, 'min_value') and hasattr(param_range, 'max_value'):
                # Normalize numeric parameters
                range_size = param_range.max_value - param_range.min_value
                if range_size > 0:
                    normalized_distance = abs(val1 - val2) / range_size
                else:
                    normalized_distance = 0.0
            else:
                # Binary distance for discrete parameters
                normalized_distance = 1.0 if val1 != val2 else 0.0
            
            distances.append(normalized_distance)
        
        return np.mean(distances)
    
    def _adapt_parameters(self, diversity: float, generation: int):
        """Adapt crossover and mutation rates based on population diversity"""
        
        # Increase mutation rate if diversity is low
        if diversity < self.diversity_threshold:
            self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.1)
            self.current_crossover_rate = max(0.3, self.current_crossover_rate * 0.9)
        else:
            # Decrease mutation rate if diversity is high
            self.current_mutation_rate = max(0.01, self.current_mutation_rate * 0.95)
            self.current_crossover_rate = min(0.9, self.current_crossover_rate * 1.05)
        
        # Adaptation based on generation progress
        progress = generation / self.generations
        if progress > 0.7:  # Late in evolution, focus on exploitation
            self.current_mutation_rate *= 0.8
            self.current_crossover_rate *= 1.1
    
    def _check_convergence(self) -> bool:
        """Check if the population has converged"""
        if len(self.best_fitness_history) < 20:
            return False
        
        # Check if best fitness hasn't improved in last 20 generations
        recent_best = self.best_fitness_history[-20:]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < 1e-6
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get detailed evolution statistics"""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population]
        ages = [ind.age for ind in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'current_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'avg_age': np.mean(ages),
            'max_age': max(ages),
            'current_crossover_rate': self.current_crossover_rate,
            'current_mutation_rate': self.current_mutation_rate,
            'convergence_trend': self.best_fitness_history[-10:] if len(self.best_fitness_history) >= 10 else [],
            'diversity_trend': self.diversity_history[-10:] if len(self.diversity_history) >= 10 else []
        }
    
    def get_population_analysis(self) -> Dict[str, Any]:
        """Analyze current population characteristics"""
        if not self.population:
            return {}
        
        # Analyze parameter distributions
        param_stats = {}
        
        for param_name in self.parameter_space.parameters.keys():
            values = [ind.parameters[param_name] for ind in self.population]
            
            if isinstance(values[0], (int, float)):
                param_stats[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'unique_values': len(set(values))
                }
            else:
                # Discrete parameters
                unique_values = list(set(values))
                value_counts = {val: values.count(val) for val in unique_values}
                param_stats[param_name] = {
                    'unique_values': len(unique_values),
                    'most_common': max(value_counts, key=value_counts.get),
                    'distribution': value_counts
                }
        
        return {
            'parameter_statistics': param_stats,
            'fitness_distribution': {
                'mean': np.mean([ind.fitness for ind in self.population]),
                'std': np.std([ind.fitness for ind in self.population]),
                'quartiles': np.percentile([ind.fitness for ind in self.population], [25, 50, 75])
            },
            'age_distribution': {
                'mean': np.mean([ind.age for ind in self.population]),
                'max': max([ind.age for ind in self.population]),
                'distribution': np.histogram([ind.age for ind in self.population], bins=5)[0].tolist()
            }
        }