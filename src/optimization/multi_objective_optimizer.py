"""
Multi-Objective Optimization

Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
multi-objective parameter optimization with Pareto front analysis.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import asyncio
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
import random
from dataclasses import dataclass

from .base_optimizer import BaseOptimizer, OptimizationConfig, OptimizationResult
from .parameter_space import ParameterSpace
from .objectives import OptimizationObjective

logger = structlog.get_logger()

@dataclass
class MultiObjectiveResult:
    """Result from multi-objective optimization"""
    parameters: Dict[str, Any]
    objective_values: List[float]  # Multiple objective values
    metrics: Dict[str, float]
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0
    is_valid: bool = True
    
    def dominates(self, other: 'MultiObjectiveResult') -> bool:
        """Check if this solution dominates another (all objectives better or equal, at least one strictly better)"""
        if not self.is_valid or not other.is_valid:
            return False
        
        all_better_or_equal = all(a >= b for a, b in zip(self.objective_values, other.objective_values))
        at_least_one_better = any(a > b for a, b in zip(self.objective_values, other.objective_values))
        
        return all_better_or_equal and at_least_one_better

class MultiObjectiveOptimizer(BaseOptimizer):
    """
    Multi-Objective Optimizer using NSGA-II algorithm.
    
    Features:
    - Pareto front identification
    - Non-dominated sorting
    - Crowding distance calculation
    - Elite preservation across fronts
    - Diversity maintenance
    - Trade-off analysis
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objectives: List[OptimizationObjective],
        config: OptimizationConfig = None,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        tournament_size: int = 2
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objectives: List of optimization objectives
            config: Optimization configuration
            population_size: Size of the population
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            tournament_size: Tournament selection size
        """
        # Use first objective for base class compatibility
        super().__init__(parameter_space, objectives[0], config)
        self.objectives = objectives
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Multi-objective state
        self.population: List[MultiObjectiveResult] = []
        self.pareto_fronts: List[List[MultiObjectiveResult]] = []
        self.generation = 0
        self.hypervolume_history: List[float] = []
        self.pareto_front_size_history: List[int] = []
    
    async def _optimize_impl(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Implementation of NSGA-II multi-objective optimization"""
        
        logger.info("Starting NSGA-II multi-objective optimization",
                   population_size=self.population_size,
                   generations=self.generations,
                   num_objectives=len(self.objectives))
        
        # Initialize population
        await self._initialize_population(strategy_factory, historical_data, start_date, end_date)
        
        # Evolution loop
        for generation in range(self.generations):
            if self.state.is_cancelled:
                logger.info("Multi-objective optimization cancelled by user")
                break
            
            self.generation = generation
            
            # Perform non-dominated sorting and crowding distance
            self._non_dominated_sort()
            self._calculate_crowding_distance()
            
            # Log progress
            pareto_front_size = len(self.pareto_fronts[0]) if self.pareto_fronts else 0
            logger.info("Processing generation",
                       generation=generation + 1,
                       pareto_front_size=pareto_front_size,
                       total_fronts=len(self.pareto_fronts))
            
            # Track statistics
            self.pareto_front_size_history.append(pareto_front_size)
            hypervolume = self._calculate_hypervolume()
            self.hypervolume_history.append(hypervolume)
            
            # Create offspring
            offspring = await self._create_offspring(strategy_factory, historical_data, start_date, end_date)
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring
            
            # Environmental selection
            self.population = self._environmental_selection(combined_population)
            
            # Update progress
            self.state.iteration = (generation + 1) * self.population_size
            self._update_progress()
            
            # Check convergence
            if self._check_convergence():
                logger.info("Convergence detected", generation=generation + 1)
                break
        
        # Final sorting to get best results
        self._non_dominated_sort()
        
        # Return best solution from first Pareto front
        if self.pareto_fronts and self.pareto_fronts[0]:
            best_solution = self.pareto_fronts[0][0]
            
            # Convert to OptimizationResult format
            return OptimizationResult(
                parameters=best_solution.parameters,
                objective_value=best_solution.objective_values[0],  # Use first objective
                metrics=best_solution.metrics,
                is_valid=best_solution.is_valid
            )
        
        raise RuntimeError("No valid solutions found during multi-objective optimization")
    
    async def _initialize_population(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ):
        """Initialize population with random individuals"""
        
        logger.info("Initializing multi-objective population", size=self.population_size)
        
        initialization_tasks = []
        
        for i in range(self.population_size):
            parameters = self.parameter_space.sample_parameters()
            task = self._evaluate_individual(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            initialization_tasks.append(task)
        
        # Evaluate in parallel
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        self.population = []
        for result in results:
            if isinstance(result, MultiObjectiveResult) and result.is_valid:
                self.population.append(result)
        
        # Ensure minimum population size
        while len(self.population) < self.population_size // 2:
            parameters = self.parameter_space.sample_parameters()
            individual = await self._evaluate_individual(
                parameters, strategy_factory, historical_data, start_date, end_date
            )
            if individual and individual.is_valid:
                self.population.append(individual)
        
        logger.info("Population initialized",
                   valid_individuals=len(self.population))
    
    async def _evaluate_individual(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[MultiObjectiveResult]:
        """Evaluate individual across all objectives"""
        
        try:
            objective_values = []
            all_metrics = {}
            
            # Evaluate each objective
            for i, objective in enumerate(self.objectives):
                result = await objective.evaluate(
                    parameters=parameters,
                    strategy_factory=strategy_factory,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not result.is_valid:
                    return MultiObjectiveResult(
                        parameters=parameters,
                        objective_values=[],
                        metrics={},
                        is_valid=False
                    )
                
                objective_values.append(result.objective_value)
                
                # Prefix metrics with objective name
                for key, value in result.metrics.items():
                    all_metrics[f"obj_{i}_{key}"] = value
            
            return MultiObjectiveResult(
                parameters=parameters,
                objective_values=objective_values,
                metrics=all_metrics,
                is_valid=True
            )
            
        except Exception as e:
            logger.warning("Multi-objective evaluation failed", error=str(e))
            return MultiObjectiveResult(
                parameters=parameters,
                objective_values=[],
                metrics={},
                is_valid=False
            )
    
    def _non_dominated_sort(self):
        """Perform non-dominated sorting to create Pareto fronts"""
        
        if not self.population:
            self.pareto_fronts = []
            return
        
        # Initialize domination counts and dominated solutions
        domination_count = [0] * len(self.population)
        dominated_solutions = [[] for _ in range(len(self.population))]
        
        # Calculate domination relationships
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i != j:
                    if self.population[i].dominates(self.population[j]):
                        dominated_solutions[i].append(j)
                    elif self.population[j].dominates(self.population[i]):
                        domination_count[i] += 1
        
        # Find first front (non-dominated solutions)
        current_front = []
        for i in range(len(self.population)):
            if domination_count[i] == 0:
                self.population[i].rank = 0
                current_front.append(i)
        
        self.pareto_fronts = []
        front_number = 0
        
        while current_front:
            self.pareto_fronts.append([self.population[i] for i in current_front])
            next_front = []
            
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.population[j].rank = front_number + 1
                        next_front.append(j)
            
            current_front = next_front
            front_number += 1
    
    def _calculate_crowding_distance(self):
        """Calculate crowding distance for diversity maintenance"""
        
        for front in self.pareto_fronts:
            if len(front) <= 2:
                # For small fronts, set maximum distance
                for solution in front:
                    solution.crowding_distance = float('inf')
                continue
            
            # Initialize crowding distance
            for solution in front:
                solution.crowding_distance = 0.0
            
            # Calculate distance for each objective
            for obj_idx in range(len(self.objectives)):
                # Sort front by objective value
                front.sort(key=lambda x: x.objective_values[obj_idx])
                
                # Set boundary solutions to infinite distance
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                
                # Calculate objective range
                obj_range = front[-1].objective_values[obj_idx] - front[0].objective_values[obj_idx]
                
                if obj_range > 0:
                    # Calculate crowding distance for interior solutions
                    for i in range(1, len(front) - 1):
                        distance = (front[i + 1].objective_values[obj_idx] - 
                                  front[i - 1].objective_values[obj_idx]) / obj_range
                        front[i].crowding_distance += distance
    
    async def _create_offspring(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[MultiObjectiveResult]:
        """Create offspring through selection, crossover, and mutation"""
        
        offspring_tasks = []
        
        for _ in range(self.population_size):
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child_params = parent1.parameters.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_params = self._mutate(child_params)
            
            # Evaluate offspring
            task = self._evaluate_individual(
                child_params, strategy_factory, historical_data, start_date, end_date
            )
            offspring_tasks.append(task)
        
        # Evaluate offspring in parallel
        offspring_results = await asyncio.gather(*offspring_tasks, return_exceptions=True)
        
        offspring = []
        for result in offspring_results:
            if isinstance(result, MultiObjectiveResult) and result.is_valid:
                offspring.append(result)
        
        return offspring
    
    def _tournament_selection(self) -> MultiObjectiveResult:
        """Tournament selection based on Pareto rank and crowding distance"""
        
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        
        # Sort by rank (lower is better), then by crowding distance (higher is better)
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        return tournament[0]
    
    def _environmental_selection(self, combined_population: List[MultiObjectiveResult]) -> List[MultiObjectiveResult]:
        """Environmental selection to maintain population size"""
        
        if len(combined_population) <= self.population_size:
            return combined_population
        
        # Sort combined population
        temp_population = combined_population
        self.population = temp_population  # Temporarily assign for sorting
        self._non_dominated_sort()
        self._calculate_crowding_distance()
        
        new_population = []
        
        # Add complete fronts
        for front in self.pareto_fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Partially fill from this front using crowding distance
                remaining_slots = self.population_size - len(new_population)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front[:remaining_slots])
                break
        
        return new_population
    
    def _crossover(self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulated binary crossover for real parameters, uniform for discrete"""
        
        child_params = {}
        
        for param_name in parent1_params.keys():
            param_range = self.parameter_space.parameters[param_name]
            
            if hasattr(param_range, 'min_value') and hasattr(param_range, 'max_value'):
                # Numeric parameter - simulated binary crossover
                val1 = parent1_params[param_name]
                val2 = parent2_params[param_name]
                
                if random.random() < 0.5:
                    child_params[param_name] = val1
                else:
                    child_params[param_name] = val2
            else:
                # Discrete parameter - uniform crossover
                if random.random() < 0.5:
                    child_params[param_name] = parent1_params[param_name]
                else:
                    child_params[param_name] = parent2_params[param_name]
        
        return child_params
    
    def _mutate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Polynomial mutation for real parameters, uniform for discrete"""
        
        mutated_params = parameters.copy()
        
        for param_name, param_range in self.parameter_space.parameters.items():
            if random.random() < 0.2:  # 20% chance to mutate each parameter
                
                if hasattr(param_range, 'min_value') and hasattr(param_range, 'max_value'):
                    # Numeric parameter
                    current_value = mutated_params[param_name]
                    range_size = param_range.max_value - param_range.min_value
                    
                    if isinstance(current_value, int):
                        mutation_strength = max(1, int(range_size * 0.1))
                        delta = random.randint(-mutation_strength, mutation_strength)
                        new_value = max(param_range.min_value,
                                      min(param_range.max_value, current_value + delta))
                    else:
                        mutation_strength = range_size * 0.1
                        delta = random.gauss(0, mutation_strength)
                        new_value = max(param_range.min_value,
                                      min(param_range.max_value, current_value + delta))
                    
                    mutated_params[param_name] = new_value
                else:
                    # Discrete parameter
                    mutated_params[param_name] = param_range.sample()
        
        return mutated_params
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator for convergence assessment"""
        
        if not self.pareto_fronts or not self.pareto_fronts[0]:
            return 0.0
        
        # Use first front for hypervolume calculation
        front = self.pareto_fronts[0]
        
        if len(self.objectives) == 2:
            # 2D hypervolume calculation
            return self._hypervolume_2d(front)
        else:
            # Approximate hypervolume for higher dimensions
            return self._hypervolume_approximate(front)
    
    def _hypervolume_2d(self, front: List[MultiObjectiveResult]) -> float:
        """Calculate 2D hypervolume"""
        
        if not front:
            return 0.0
        
        # Sort by first objective
        sorted_front = sorted(front, key=lambda x: x.objective_values[0], reverse=True)
        
        # Reference point (assuming maximization)
        ref_point = [0.0, 0.0]
        
        hypervolume = 0.0
        prev_y = ref_point[1]
        
        for solution in sorted_front:
            x = solution.objective_values[0]
            y = solution.objective_values[1]
            
            if x > ref_point[0] and y > prev_y:
                hypervolume += (x - ref_point[0]) * (y - prev_y)
                prev_y = y
        
        return hypervolume
    
    def _hypervolume_approximate(self, front: List[MultiObjectiveResult]) -> float:
        """Approximate hypervolume for multi-dimensional objectives"""
        
        if not front:
            return 0.0
        
        # Simple approximation: sum of normalized objective values
        total_volume = 0.0
        
        for solution in front:
            # Normalize objectives (assuming positive values)
            normalized_objectives = [max(0, obj) for obj in solution.objective_values]
            volume = np.prod(normalized_objectives) if normalized_objectives else 0
            total_volume += volume
        
        return total_volume
    
    def _check_convergence(self) -> bool:
        """Check convergence based on hypervolume improvement"""
        
        if len(self.hypervolume_history) < 20:
            return False
        
        # Check if hypervolume has plateaued
        recent_hypervolumes = self.hypervolume_history[-20:]
        improvement = max(recent_hypervolumes) - min(recent_hypervolumes)
        
        return improvement < 1e-6
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the current Pareto front"""
        
        if not self.pareto_fronts or not self.pareto_fronts[0]:
            return []
        
        pareto_front = []
        for solution in self.pareto_fronts[0]:
            pareto_front.append({
                'parameters': solution.parameters,
                'objective_values': solution.objective_values,
                'metrics': solution.metrics,
                'crowding_distance': solution.crowding_distance
            })
        
        return pareto_front
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get detailed optimization statistics"""
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'num_fronts': len(self.pareto_fronts),
            'pareto_front_size': len(self.pareto_fronts[0]) if self.pareto_fronts else 0,
            'hypervolume_history': self.hypervolume_history,
            'pareto_front_size_history': self.pareto_front_size_history,
            'objective_ranges': self._calculate_objective_ranges(),
            'convergence_metrics': {
                'hypervolume_trend': self.hypervolume_history[-10:] if len(self.hypervolume_history) >= 10 else [],
                'front_size_stability': np.std(self.pareto_front_size_history[-10:]) if len(self.pareto_front_size_history) >= 10 else 0
            }
        }
    
    def _calculate_objective_ranges(self) -> Dict[str, Dict[str, float]]:
        """Calculate objective value ranges across population"""
        
        if not self.population:
            return {}
        
        ranges = {}
        
        for i, objective in enumerate(self.objectives):
            values = [sol.objective_values[i] for sol in self.population if sol.is_valid]
            
            if values:
                ranges[f"objective_{i}_{objective.name}"] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return ranges
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """Analyze trade-offs between objectives"""
        
        if not self.pareto_fronts or not self.pareto_fronts[0]:
            return {}
        
        front = self.pareto_fronts[0]
        
        # Calculate correlations between objectives
        correlations = {}
        
        for i in range(len(self.objectives)):
            for j in range(i + 1, len(self.objectives)):
                obj_i_values = [sol.objective_values[i] for sol in front]
                obj_j_values = [sol.objective_values[j] for sol in front]
                
                if len(obj_i_values) > 1:
                    correlation = np.corrcoef(obj_i_values, obj_j_values)[0, 1]
                    correlations[f"obj_{i}_vs_obj_{j}"] = correlation
        
        # Find extreme solutions
        extreme_solutions = {}
        
        for i, objective in enumerate(self.objectives):
            # Best for this objective
            best_solution = max(front, key=lambda x: x.objective_values[i])
            worst_solution = min(front, key=lambda x: x.objective_values[i])
            
            extreme_solutions[f"best_{objective.name}"] = {
                'parameters': best_solution.parameters,
                'objective_values': best_solution.objective_values
            }
            
            extreme_solutions[f"worst_{objective.name}"] = {
                'parameters': worst_solution.parameters,
                'objective_values': worst_solution.objective_values
            }
        
        return {
            'objective_correlations': correlations,
            'extreme_solutions': extreme_solutions,
            'pareto_front_diversity': self._calculate_front_diversity(),
            'dominated_space_coverage': len(self.pareto_fronts[0]) / len(self.population) if self.population else 0
        }
    
    def _calculate_front_diversity(self) -> float:
        """Calculate diversity of the Pareto front"""
        
        if not self.pareto_fronts or len(self.pareto_fronts[0]) < 2:
            return 0.0
        
        front = self.pareto_fronts[0]
        
        # Calculate pairwise distances in objective space
        distances = []
        
        for i in range(len(front)):
            for j in range(i + 1, len(front)):
                distance = np.linalg.norm(
                    np.array(front[i].objective_values) - np.array(front[j].objective_values)
                )
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0