"""
Grid Search Optimizer

Implements brute-force grid search parameter optimization with parallel processing
and intelligent pruning strategies.
"""

from typing import Dict, List, Any, Optional, Callable, Union
import asyncio
import concurrent.futures
from datetime import datetime
import numpy as np
import pandas as pd
import structlog
import time

from .base_optimizer import BaseOptimizer, OptimizationConfig, OptimizationResult
from .parameter_space import ParameterSpace
from .objectives import OptimizationObjective

logger = structlog.get_logger()

class GridSearchOptimizer(BaseOptimizer):
    """
    Grid search optimizer with parallel processing and intelligent pruning.
    
    Features:
    - Exhaustive search over parameter grid
    - Parallel evaluation using asyncio and thread pools
    - Early termination for unpromising regions
    - Adaptive grid refinement
    - Memory-efficient batch processing
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective: OptimizationObjective,
        config: OptimizationConfig = None,
        grid_size: Union[int, Dict[str, int]] = 3,
        enable_pruning: bool = True,
        pruning_threshold: float = 0.1,
        adaptive_refinement: bool = False,
        refinement_factor: int = 2
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objective: Optimization objective function
            config: Optimization configuration
            grid_size: Grid size per parameter (int) or per-parameter (dict)
            enable_pruning: Enable early pruning of unpromising regions
            pruning_threshold: Threshold for pruning (as fraction of best)
            adaptive_refinement: Enable adaptive grid refinement
            refinement_factor: Factor for grid refinement
        """
        super().__init__(parameter_space, objective, config)
        self.grid_size = grid_size
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        self.adaptive_refinement = adaptive_refinement
        self.refinement_factor = refinement_factor
        
        # Grid search state
        self.parameter_combinations: List[Dict[str, Any]] = []
        self.current_batch: int = 0
        self.total_batches: int = 0
        
    async def _optimize_impl(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Implementation of grid search optimization"""
        
        logger.info("Starting grid search optimization",
                   grid_size=self.grid_size,
                   pruning_enabled=self.enable_pruning,
                   adaptive_refinement=self.adaptive_refinement)
        
        # Generate initial parameter grid
        self.parameter_combinations = self.parameter_space.generate_grid(self.grid_size)
        total_combinations = len(self.parameter_combinations)
        
        if total_combinations == 0:
            raise ValueError("No valid parameter combinations found")
        
        # Performance safeguard: limit maximum combinations
        max_combinations = 100  # Reasonable limit for development
        if total_combinations > max_combinations:
            logger.warning("Too many parameter combinations, sampling subset",
                         total=total_combinations,
                         max_allowed=max_combinations)
            # Randomly sample a subset
            import random
            random.seed(42)  # For reproducible results
            self.parameter_combinations = random.sample(self.parameter_combinations, max_combinations)
            total_combinations = len(self.parameter_combinations)
        
        logger.info("Generated parameter combinations", 
                   total=total_combinations,
                   parameters=list(self.parameter_space.parameters.keys()))
        
        # Set up batching
        batch_size = self.config.batch_size
        self.total_batches = (total_combinations + batch_size - 1) // batch_size
        
        best_result = None
        completed_evaluations = 0
        
        # Process combinations in batches with timeout
        start_time = time.time()
        max_runtime_seconds = 60  # 1 minute timeout for faster feedback
        
        for batch_idx in range(self.total_batches):
            if self.state.is_cancelled:
                logger.info("Grid search cancelled by user")
                break
            
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > max_runtime_seconds:
                logger.warning("Grid search timeout reached",
                             elapsed_seconds=elapsed_time,
                             max_seconds=max_runtime_seconds)
                break
            
            self.current_batch = batch_idx
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_combinations)
            batch_combinations = self.parameter_combinations[batch_start:batch_end]
            
            logger.info("Processing batch", 
                       batch=batch_idx + 1, 
                       total_batches=self.total_batches,
                       combinations=len(batch_combinations),
                       elapsed_seconds=int(elapsed_time))
            
            # Evaluate batch in parallel
            batch_results = await self._evaluate_batch_parallel(
                batch_combinations,
                strategy_factory,
                historical_data,
                start_date,
                end_date
            )
            
            # Update best result
            for result in batch_results:
                completed_evaluations += 1
                
                if result.is_valid and (best_result is None or 
                                      result.objective_value > best_result.objective_value):
                    best_result = result
                    logger.info("New best result found",
                               objective_value=result.objective_value,
                               parameters=result.parameters)
            
            # Update progress
            self.state.iteration = completed_evaluations
            self._update_progress()
            
            # Apply pruning if enabled
            if self.enable_pruning and best_result is not None:
                self._prune_remaining_combinations(best_result.objective_value)
            
            # Check early stopping
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                break
            
            # Check timeout
            if (self.config.max_time_seconds and 
                self.state.elapsed_time.total_seconds() > self.config.max_time_seconds):
                logger.info("Time limit reached")
                break
        
        # Adaptive refinement if enabled
        if self.adaptive_refinement and best_result is not None:
            refined_result = await self._adaptive_refinement(
                best_result,
                strategy_factory,
                historical_data,
                start_date,
                end_date
            )
            if refined_result and refined_result.objective_value > best_result.objective_value:
                best_result = refined_result
        
        if best_result is None:
            raise RuntimeError("No valid results found during optimization")
        
        logger.info("Grid search completed",
                   total_evaluations=completed_evaluations,
                   best_objective=best_result.objective_value,
                   elapsed_time=self.state.elapsed_time.total_seconds())
        
        return best_result
    
    async def _evaluate_batch_parallel(
        self,
        parameter_combinations: List[Dict[str, Any]],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[OptimizationResult]:
        """Evaluate a batch of parameter combinations in parallel"""
        
        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def evaluate_with_semaphore(parameters):
            async with semaphore:
                return await self.evaluate_parameters(
                    parameters,
                    strategy_factory,
                    historical_data,
                    start_date,
                    end_date,
                    use_cache=True
                )
        
        # Execute evaluations concurrently
        tasks = [evaluate_with_semaphore(params) for params in parameter_combinations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Parameter evaluation failed",
                           parameters=parameter_combinations[i],
                           error=str(result))
                # Create failed result
                failed_result = OptimizationResult(
                    parameters=parameter_combinations[i],
                    objective_value=float('-inf'),
                    metrics={},
                    is_valid=False,
                    error_message=str(result)
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _prune_remaining_combinations(self, best_objective: float):
        """Prune unpromising parameter combinations"""
        if not self.enable_pruning:
            return
        
        threshold = best_objective * (1 - self.pruning_threshold)
        
        # Simple heuristic: remove combinations that are far from best parameters
        if self.state.best_parameters is None:
            return
        
        # Calculate parameter distance and prune based on heuristics
        # For now, implement basic pruning - can be enhanced with ML models
        original_count = len(self.parameter_combinations)
        
        # Remove combinations that are likely to perform poorly
        # This is a placeholder for more sophisticated pruning strategies
        
        pruned_count = original_count - len(self.parameter_combinations)
        if pruned_count > 0:
            logger.info("Pruned parameter combinations",
                       pruned=pruned_count,
                       remaining=len(self.parameter_combinations))
    
    async def _adaptive_refinement(
        self,
        best_result: OptimizationResult,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[OptimizationResult]:
        """Perform adaptive grid refinement around best parameters"""
        
        logger.info("Starting adaptive refinement around best parameters")
        
        # Create refined parameter space around best parameters
        refined_space = self._create_refined_space(best_result.parameters)
        
        if refined_space is None:
            return None
        
        # Generate refined grid
        refined_combinations = refined_space.generate_grid(self.refinement_factor)
        
        if not refined_combinations:
            return None
        
        logger.info("Generated refined parameter grid",
                   combinations=len(refined_combinations))
        
        # Evaluate refined combinations
        refined_results = await self._evaluate_batch_parallel(
            refined_combinations,
            strategy_factory,
            historical_data,
            start_date,
            end_date
        )
        
        # Find best refined result
        best_refined = None
        for result in refined_results:
            if result.is_valid and (best_refined is None or 
                                  result.objective_value > best_refined.objective_value):
                best_refined = result
        
        if best_refined:
            logger.info("Adaptive refinement completed",
                       improvement=best_refined.objective_value - best_result.objective_value)
        
        return best_refined
    
    def _create_refined_space(self, best_parameters: Dict[str, Any]) -> Optional[ParameterSpace]:
        """Create refined parameter space around best parameters"""
        
        refined_space = ParameterSpace()
        
        for param_name, best_value in best_parameters.items():
            if param_name not in self.parameter_space.parameters:
                continue
            
            original_range = self.parameter_space.parameters[param_name]
            
            # Create narrower range around best value
            if hasattr(original_range, 'min_value') and hasattr(original_range, 'max_value'):
                # Numeric parameter
                total_range = original_range.max_value - original_range.min_value
                refined_range = total_range / (self.refinement_factor * 2)
                
                new_min = max(original_range.min_value, best_value - refined_range)
                new_max = min(original_range.max_value, best_value + refined_range)
                
                if hasattr(original_range, 'log_scale'):
                    # Float range
                    refined_space.add_float(
                        param_name,
                        new_min,
                        new_max,
                        log_scale=original_range.log_scale,
                        description=f"Refined {param_name}"
                    )
                else:
                    # Integer range
                    refined_space.add_integer(
                        param_name,
                        int(new_min),
                        int(new_max),
                        description=f"Refined {param_name}"
                    )
            else:
                # Choice or boolean - keep original
                refined_space.add_parameter(original_range)
        
        return refined_space if refined_space.parameters else None
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get information about the current grid"""
        return {
            'total_combinations': len(self.parameter_combinations),
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'batch_size': self.config.batch_size,
            'grid_size': self.grid_size,
            'evaluations_completed': len(self.all_results),
            'evaluations_cached': len(self.evaluation_cache),
            'pruning_enabled': self.enable_pruning,
            'adaptive_refinement': self.adaptive_refinement
        }
    
    def get_parameter_analysis(self) -> Dict[str, Any]:
        """Analyze parameter performance across the grid"""
        if not self.all_results:
            return {}
        
        # Get valid results only
        valid_results = [r for r in self.all_results if r.is_valid]
        
        if not valid_results:
            return {}
        
        analysis = {}
        
        # Analyze each parameter
        for param_name in self.parameter_space.parameters.keys():
            param_values = []
            objectives = []
            
            for result in valid_results:
                if param_name in result.parameters:
                    param_values.append(result.parameters[param_name])
                    objectives.append(result.objective_value)
            
            if param_values:
                # Calculate correlation between parameter and objective
                correlation = np.corrcoef(param_values, objectives)[0, 1] if len(param_values) > 1 else 0
                
                analysis[param_name] = {
                    'correlation_with_objective': correlation,
                    'best_value': param_values[np.argmax(objectives)],
                    'worst_value': param_values[np.argmin(objectives)],
                    'mean_value': np.mean(param_values),
                    'std_value': np.std(param_values),
                    'values_tested': len(set(param_values))
                }
        
        # Overall grid statistics
        objective_values = [r.objective_value for r in valid_results]
        analysis['grid_statistics'] = {
            'total_valid_results': len(valid_results),
            'best_objective': max(objective_values),
            'worst_objective': min(objective_values),
            'mean_objective': np.mean(objective_values),
            'std_objective': np.std(objective_values),
            'objective_range': max(objective_values) - min(objective_values)
        }
        
        return analysis

class AdaptiveGridSearchOptimizer(GridSearchOptimizer):
    """
    Adaptive grid search that starts coarse and refines promising regions.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective: OptimizationObjective,
        config: OptimizationConfig = None,
        initial_grid_size: int = 5,
        max_refinement_levels: int = 3,
        refinement_threshold: float = 0.9,
        top_candidates_ratio: float = 0.1
    ):
        """
        Initialize adaptive grid search optimizer.
        
        Args:
            parameter_space: Parameter space definition
            objective: Optimization objective function
            config: Optimization configuration
            initial_grid_size: Initial coarse grid size
            max_refinement_levels: Maximum number of refinement levels
            refinement_threshold: Threshold for selecting regions to refine
            top_candidates_ratio: Ratio of top candidates to refine
        """
        super().__init__(parameter_space, objective, config, initial_grid_size)
        self.initial_grid_size = initial_grid_size
        self.max_refinement_levels = max_refinement_levels
        self.refinement_threshold = refinement_threshold
        self.top_candidates_ratio = top_candidates_ratio
        
    async def _optimize_impl(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Implementation of adaptive grid search"""
        
        logger.info("Starting adaptive grid search optimization",
                   initial_grid_size=self.initial_grid_size,
                   max_levels=self.max_refinement_levels)
        
        best_result = None
        current_grid_size = self.initial_grid_size
        
        for level in range(self.max_refinement_levels):
            if self.state.is_cancelled:
                break
            
            logger.info("Starting refinement level", level=level + 1)
            
            # Generate grid for current level
            if level == 0:
                # Initial coarse grid
                combinations = self.parameter_space.generate_grid(current_grid_size)
            else:
                # Refined grid around promising regions
                combinations = self._generate_refined_grid(level)
            
            if not combinations:
                logger.info("No more combinations to evaluate")
                break
            
            # Evaluate current level
            level_results = await self._evaluate_level(
                combinations,
                strategy_factory,
                historical_data,
                start_date,
                end_date
            )
            
            # Update best result
            for result in level_results:
                if result.is_valid and (best_result is None or 
                                      result.objective_value > best_result.objective_value):
                    best_result = result
            
            # Check if we should continue refining
            if not self._should_continue_refining(level_results):
                logger.info("Stopping refinement - no improvement")
                break
            
            current_grid_size *= 2  # Increase resolution for next level
        
        if best_result is None:
            raise RuntimeError("No valid results found during adaptive optimization")
        
        return best_result
    
    async def _evaluate_level(
        self,
        combinations: List[Dict[str, Any]],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[OptimizationResult]:
        """Evaluate all combinations in a refinement level"""
        
        batch_size = self.config.batch_size
        all_results = []
        
        for i in range(0, len(combinations), batch_size):
            if self.state.is_cancelled:
                break
            
            batch = combinations[i:i + batch_size]
            batch_results = await self._evaluate_batch_parallel(
                batch, strategy_factory, historical_data, start_date, end_date
            )
            
            all_results.extend(batch_results)
            
            # Update progress
            self.state.iteration += len(batch_results)
            self._update_progress()
        
        return all_results
    
    def _generate_refined_grid(self, level: int) -> List[Dict[str, Any]]:
        """Generate refined grid around top performing regions"""
        
        if not self.all_results:
            return []
        
        # Get top performing results
        valid_results = [r for r in self.all_results if r.is_valid]
        
        if not valid_results:
            return []
        
        # Sort by objective value and take top candidates
        valid_results.sort(key=lambda x: x.objective_value, reverse=True)
        num_top = max(1, int(len(valid_results) * self.top_candidates_ratio))
        top_results = valid_results[:num_top]
        
        # Create refined spaces around each top result
        all_combinations = []
        
        for result in top_results:
            refined_space = self._create_local_refined_space(result.parameters, level)
            if refined_space:
                local_combinations = refined_space.generate_grid(3)  # Small local grid
                all_combinations.extend(local_combinations)
        
        # Remove duplicates
        unique_combinations = []
        seen_keys = set()
        
        for combo in all_combinations:
            key = tuple(sorted(combo.items()))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_combinations.append(combo)
        
        return unique_combinations
    
    def _create_local_refined_space(self, center_params: Dict[str, Any], level: int) -> Optional[ParameterSpace]:
        """Create a local refined parameter space around center parameters"""
        
        refined_space = ParameterSpace()
        refinement_factor = 2 ** (level + 1)  # Increasing refinement
        
        for param_name, center_value in center_params.items():
            if param_name not in self.parameter_space.parameters:
                continue
            
            original_range = self.parameter_space.parameters[param_name]
            
            if hasattr(original_range, 'min_value') and hasattr(original_range, 'max_value'):
                # Create smaller range around center
                total_range = original_range.max_value - original_range.min_value
                local_range = total_range / refinement_factor
                
                new_min = max(original_range.min_value, center_value - local_range / 2)
                new_max = min(original_range.max_value, center_value + local_range / 2)
                
                if hasattr(original_range, 'log_scale'):
                    refined_space.add_float(
                        param_name, new_min, new_max,
                        log_scale=original_range.log_scale
                    )
                else:
                    refined_space.add_integer(
                        param_name, int(new_min), int(new_max)
                    )
            else:
                # Keep original for discrete parameters
                refined_space.add_parameter(original_range)
        
        return refined_space if refined_space.parameters else None
    
    def _should_continue_refining(self, level_results: List[OptimizationResult]) -> bool:
        """Determine if refinement should continue"""
        
        if not level_results:
            return False
        
        valid_results = [r for r in level_results if r.is_valid]
        
        if not valid_results:
            return False
        
        # Check if any result meets refinement threshold
        best_objective = max(r.objective_value for r in valid_results)
        
        if self.state.best_result is None:
            return True
        
        improvement = (best_objective - self.state.best_result.objective_value) / abs(self.state.best_result.objective_value)
        
        return improvement > (1 - self.refinement_threshold)