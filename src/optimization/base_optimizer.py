"""
Base Optimizer - Abstract base class for all parameter optimization algorithms
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import structlog
import numpy as np
import pandas as pd

from .parameter_space import ParameterSpace
from .objectives import OptimizationObjective, OptimizationResult

logger = structlog.get_logger()

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs"""
    # General settings
    max_iterations: int = 1000
    max_time_seconds: Optional[int] = None
    tolerance: float = 1e-6
    
    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10
    
    # Validation settings
    validation_method: str = 'walk_forward'  # 'walk_forward', 'cross_validation', 'holdout'
    validation_windows: int = 5
    out_of_sample_ratio: float = 0.2
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
    
    # Logging and persistence
    save_intermediate_results: bool = True
    log_progress_interval: int = 10
    
    # Risk management
    max_drawdown_threshold: float = 0.3  # Stop if DD > 30%
    min_sharpe_threshold: float = 0.5    # Stop if Sharpe < 0.5

@dataclass 
class OptimizationState:
    """Tracks the state of an optimization run"""
    iteration: int = 0
    best_result: Optional[OptimizationResult] = None
    best_parameters: Optional[Dict[str, Any]] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    is_running: bool = False
    is_cancelled: bool = False
    convergence_history: List[float] = field(default_factory=list)
    
    @property
    def elapsed_time(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def is_converged(self) -> bool:
        if len(self.convergence_history) < 2:
            return False
        return abs(self.convergence_history[-1] - self.convergence_history[-2]) < 1e-6

class BaseOptimizer(ABC):
    """
    Abstract base class for all parameter optimization algorithms.
    
    Provides common functionality for:
    - Parameter space management
    - Objective function evaluation
    - Progress tracking and cancellation
    - Result storage and analysis
    - Overfitting prevention through validation
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective: OptimizationObjective,
        config: OptimizationConfig = None
    ):
        self.parameter_space = parameter_space
        self.objective = objective
        self.config = config or OptimizationConfig()
        self.state = OptimizationState()
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.iteration_callback: Optional[Callable] = None
        
        # Results storage
        self.all_results: List[OptimizationResult] = []
        self.evaluation_cache: Dict[str, OptimizationResult] = {}
        
    async def optimize(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run the optimization algorithm.
        
        Args:
            strategy_factory: Function that creates strategy instances
            historical_data: Market data for backtesting
            start_date: Start date for optimization
            end_date: End date for optimization  
            progress_callback: Optional callback for progress updates
            
        Returns:
            Best optimization result found
        """
        logger.info("Starting optimization", 
                   algorithm=self.__class__.__name__,
                   parameter_count=len(self.parameter_space.parameters),
                   max_iterations=self.config.max_iterations)
        
        self.progress_callback = progress_callback
        self.state.is_running = True
        self.state.start_time = datetime.now()
        
        try:
            # Validate data and parameters
            self._validate_inputs(historical_data, start_date, end_date)
            
            # Run the optimization algorithm
            best_result = await self._optimize_impl(
                strategy_factory, historical_data, start_date, end_date
            )
            
            self.state.best_result = best_result
            self.state.best_parameters = best_result.parameters if best_result else None
            
            logger.info("Optimization completed",
                       iterations=self.state.iteration,
                       best_objective=best_result.objective_value if best_result else None,
                       elapsed_time=self.state.elapsed_time.total_seconds())
            
            return best_result
            
        except Exception as e:
            logger.error("Optimization failed", error=str(e))
            raise
        finally:
            self.state.is_running = False
            self.state.end_time = datetime.now()
    
    @abstractmethod
    async def _optimize_impl(
        self,
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """
        Implementation-specific optimization logic.
        Must be implemented by subclasses.
        """
        pass
    
    async def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> OptimizationResult:
        """
        Evaluate a single parameter set using the objective function.
        
        Args:
            parameters: Parameter values to evaluate
            strategy_factory: Function to create strategy instances
            historical_data: Market data
            start_date: Start date for evaluation
            end_date: End date for evaluation
            use_cache: Whether to use cached results
            
        Returns:
            Optimization result for the parameter set
        """
        # Check cache first
        param_key = self._parameters_to_key(parameters)
        if use_cache and param_key in self.evaluation_cache:
            return self.evaluation_cache[param_key]
        
        try:
            # Validate parameters are within bounds
            self.parameter_space.validate_parameters(parameters)
            
            # Evaluate using the objective function
            result = await self.objective.evaluate(
                parameters=parameters,
                strategy_factory=strategy_factory,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Cache the result
            if use_cache:
                self.evaluation_cache[param_key] = result
            
            # Store in results history
            self.all_results.append(result)
            
            # Update best result
            if (self.state.best_result is None or 
                result.objective_value > self.state.best_result.objective_value):
                self.state.best_result = result
                self.state.best_parameters = parameters.copy()
            
            return result
            
        except Exception as e:
            logger.error("Parameter evaluation failed", 
                        parameters=parameters, error=str(e))
            # Return a failed result
            return OptimizationResult(
                parameters=parameters,
                objective_value=float('-inf'),
                metrics={},
                validation_results={},
                is_valid=False,
                error_message=str(e)
            )
    
    def set_progress_callback(self, callback: Callable):
        """Set progress callback function"""
        self.progress_callback = callback
        logger.debug("Progress callback set")
    
    def cancel(self):
        """Cancel the optimization run"""
        self.state.is_cancelled = True
        logger.info("Optimization cancelled by user")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current optimization progress"""
        return {
            'iteration': self.state.iteration,
            'max_iterations': self.config.max_iterations,
            'progress_pct': (self.state.iteration / self.config.max_iterations) * 100,
            'elapsed_time': self.state.elapsed_time.total_seconds(),
            'is_running': self.state.is_running,
            'is_cancelled': self.state.is_cancelled,
            'best_objective': self.state.best_result.objective_value if self.state.best_result else None,
            'best_parameters': self.state.best_parameters,
            'evaluations_completed': len(self.all_results)
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.all_results:
            return {'message': 'No results available'}
        
        objective_values = [r.objective_value for r in self.all_results if r.is_valid]
        
        if not objective_values:
            return {'message': 'No valid results found'}
        
        return {
            'total_evaluations': len(self.all_results),
            'valid_evaluations': len(objective_values),
            'best_objective': max(objective_values),
            'worst_objective': min(objective_values),
            'mean_objective': np.mean(objective_values),
            'std_objective': np.std(objective_values),
            'best_parameters': self.state.best_parameters,
            'convergence_history': self.state.convergence_history,
            'elapsed_time': self.state.elapsed_time.total_seconds()
        }
    
    def _validate_inputs(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ):
        """Validate optimization inputs"""
        if not historical_data:
            raise ValueError("Historical data cannot be empty")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Check data coverage
        for symbol, df in historical_data.items():
            if df.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            # Handle timestamp as either column or index
            if 'timestamp' in df.columns:
                data_start = pd.to_datetime(df['timestamp'].min(), unit='ms')
                data_end = pd.to_datetime(df['timestamp'].max(), unit='ms')
            else:
                # Timestamp is likely the index
                data_start = df.index.min()
                data_end = df.index.max()
                # Convert to datetime if needed
                if not isinstance(data_start, pd.Timestamp):
                    data_start = pd.to_datetime(data_start, unit='ms')
                    data_end = pd.to_datetime(data_end, unit='ms')
            
            if start_date < data_start or end_date > data_end:
                raise ValueError(
                    f"Requested date range [{start_date}, {end_date}] "
                    f"exceeds available data range [{data_start}, {data_end}] "
                    f"for symbol {symbol}"
                )
    
    def _parameters_to_key(self, parameters: Dict[str, Any]) -> str:
        """Convert parameters dict to cache key"""
        sorted_items = sorted(parameters.items())
        return str(hash(tuple(sorted_items)))
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early"""
        if not self.config.early_stopping_enabled:
            return False
        
        if len(self.state.convergence_history) < self.config.early_stopping_patience:
            return False
        
        # Check for convergence
        recent_values = self.state.convergence_history[-self.config.early_stopping_patience:]
        improvement = max(recent_values) - min(recent_values)
        
        return improvement < self.config.early_stopping_min_delta
    
    def _update_progress(self, iteration: int = None):
        """Update optimization progress and call callbacks"""
        if iteration is not None:
            self.state.iteration = iteration
        
        # Update convergence history
        if self.state.best_result:
            self.state.convergence_history.append(self.state.best_result.objective_value)
        
        # Call progress callback
        if self.progress_callback and self.state.iteration % self.config.log_progress_interval == 0:
            try:
                self.progress_callback(self.get_progress())
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))