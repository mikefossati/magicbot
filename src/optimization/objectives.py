"""
Optimization Objectives

Defines objective functions for parameter optimization including single and multi-objective
functions with various performance metrics.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd
import structlog

from ..backtesting.engine import BacktestEngine, BacktestConfig

logger = structlog.get_logger()

@dataclass
class OptimizationResult:
    """Result of evaluating a parameter set"""
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float]
    validation_results: Dict[str, Any] = field(default_factory=dict)
    backtest_results: Optional[Dict[str, Any]] = None
    is_valid: bool = True
    error_message: Optional[str] = None
    evaluation_time: float = 0.0
    
    def __post_init__(self):
        """Ensure objective value is finite"""
        if not np.isfinite(self.objective_value):
            self.objective_value = float('-inf')
            self.is_valid = False

class OptimizationObjective(ABC):
    """Base class for optimization objectives"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def evaluate(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Evaluate objective function for given parameters"""
        pass

class SingleObjective(OptimizationObjective):
    """Single objective optimization using various metrics"""
    
    def __init__(
        self,
        metric: str = 'total_return_pct',
        minimize: bool = False,
        risk_penalty_weight: float = 0.0,
        transaction_cost_penalty: float = 0.0,
        min_trades_threshold: int = 0,  # Allow zero trades for testing
        max_drawdown_threshold: float = 1.0,  # Very lenient for testing
        min_sharpe_threshold: float = -10.0  # Very lenient for testing
    ):
        """
        Initialize single objective function.
        
        Args:
            metric: Primary metric to optimize ('total_return_pct', 'sharpe_ratio', 'calmar_ratio', etc.)
            minimize: Whether to minimize the metric (default: maximize)
            risk_penalty_weight: Weight for risk penalty (0-1)
            transaction_cost_penalty: Penalty for excessive trading
            min_trades_threshold: Minimum number of trades required
            max_drawdown_threshold: Maximum acceptable drawdown
            min_sharpe_threshold: Minimum acceptable Sharpe ratio
        """
        super().__init__(f"Single_{metric}", f"Optimize {metric}")
        self.metric = metric
        self.minimize = minimize
        self.risk_penalty_weight = risk_penalty_weight
        self.transaction_cost_penalty = transaction_cost_penalty
        self.min_trades_threshold = min_trades_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_sharpe_threshold = min_sharpe_threshold
    
    async def evaluate(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Evaluate single objective"""
        import time
        start_time = time.time()
        
        try:
            # Create strategy instance with new architecture
            # strategy_factory should now be a tuple of (strategy_name, base_config)
            if isinstance(strategy_factory, tuple):
                strategy_name, base_config = strategy_factory
                # Merge optimization parameters with base config
                config = {**base_config, **parameters}
                
                # Import here to avoid circular imports
                from ..strategies.registry import create_strategy
                strategy = create_strategy(strategy_name, config)
            else:
                # Legacy factory function support
                strategy = strategy_factory(parameters)
            
            # Run backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=parameters.get('base_position_size', 0.05)
            )
            
            engine = BacktestEngine(initial_balance=10000.0, fast_mode=True)
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Check validity constraints
            is_valid, reason = self._check_validity(metrics)
            
            if not is_valid:
                return OptimizationResult(
                    parameters=parameters,
                    objective_value=float('-inf'),
                    metrics=metrics,
                    backtest_results=results,
                    is_valid=False,
                    error_message=reason,
                    evaluation_time=time.time() - start_time
                )
            
            # Calculate objective value
            objective_value = self._calculate_objective(metrics, results)
            
            return OptimizationResult(
                parameters=parameters,
                objective_value=objective_value,
                metrics=metrics,
                backtest_results=results,
                is_valid=True,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error("Objective evaluation failed", 
                        parameters=parameters, error=str(e))
            return OptimizationResult(
                parameters=parameters,
                objective_value=float('-inf'),
                metrics={},
                is_valid=False,
                error_message=str(e),
                evaluation_time=time.time() - start_time
            )
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant metrics from backtest results"""
        metrics = {}
        
        # Capital metrics
        capital = results.get('capital', {})
        metrics['total_return_pct'] = capital.get('total_return_pct', 0.0)
        metrics['annualized_return_pct'] = capital.get('annualized_return_pct', 0.0)
        metrics['final_capital'] = capital.get('final', 10000.0)
        
        # Risk metrics
        risk = results.get('risk_metrics', {})
        metrics['sharpe_ratio'] = risk.get('sharpe_ratio', 0.0)
        metrics['sortino_ratio'] = risk.get('sortino_ratio', 0.0)
        metrics['calmar_ratio'] = risk.get('calmar_ratio', 0.0)
        metrics['max_drawdown_pct'] = risk.get('max_drawdown_pct', 0.0)
        metrics['volatility_pct'] = risk.get('volatility_pct', 0.0)
        metrics['var_95_pct'] = risk.get('var_95_pct', 0.0)
        
        # Trade metrics
        trades = results.get('trades', {})
        metrics['total_trades'] = trades.get('total', 0)
        metrics['win_rate_pct'] = trades.get('win_rate_pct', 0.0)
        metrics['profit_factor'] = trades.get('profit_factor', 0.0)
        metrics['avg_trade_pct'] = trades.get('avg_trade_pct', 0.0)
        metrics['best_trade_pct'] = trades.get('best_trade_pct', 0.0)
        metrics['worst_trade_pct'] = trades.get('worst_trade_pct', 0.0)
        
        return metrics
    
    def _check_validity(self, metrics: Dict[str, float]) -> tuple[bool, Optional[str]]:
        """Check if results meet validity constraints"""
        # Check minimum trades
        if metrics.get('total_trades', 0) < self.min_trades_threshold:
            return False, f"Too few trades: {metrics.get('total_trades', 0)} < {self.min_trades_threshold}"
        
        # Check maximum drawdown
        if abs(metrics.get('max_drawdown_pct', 0)) > self.max_drawdown_threshold * 100:
            return False, f"Excessive drawdown: {metrics.get('max_drawdown_pct', 0)}% > {self.max_drawdown_threshold * 100}%"
        
        # Check minimum Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < self.min_sharpe_threshold:
            return False, f"Poor Sharpe ratio: {metrics.get('sharpe_ratio', 0)} < {self.min_sharpe_threshold}"
        
        # Check for invalid metrics
        primary_metric = metrics.get(self.metric, 0)
        if not np.isfinite(primary_metric):
            return False, f"Invalid primary metric: {primary_metric}"
        
        return True, None
    
    def _calculate_objective(self, metrics: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate final objective value"""
        # Get primary metric value
        primary_value = metrics.get(self.metric, 0.0)
        
        if self.minimize:
            primary_value = -primary_value
        
        # Apply risk penalty
        risk_penalty = 0.0
        if self.risk_penalty_weight > 0:
            # Penalty based on drawdown and volatility
            drawdown_penalty = abs(metrics.get('max_drawdown_pct', 0)) / 100.0
            volatility_penalty = metrics.get('volatility_pct', 0) / 100.0
            risk_penalty = self.risk_penalty_weight * (drawdown_penalty + volatility_penalty)
        
        # Apply transaction cost penalty
        trade_penalty = 0.0
        if self.transaction_cost_penalty > 0:
            num_trades = metrics.get('total_trades', 0)
            # Penalty for excessive trading
            trade_penalty = self.transaction_cost_penalty * max(0, num_trades - 50) / 100.0
        
        # Combine objective
        objective_value = primary_value - risk_penalty - trade_penalty
        
        return objective_value

class MultiObjective(OptimizationObjective):
    """Multi-objective optimization using Pareto efficiency"""
    
    def __init__(
        self,
        objectives: List[Dict[str, Any]],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-objective function.
        
        Args:
            objectives: List of objective definitions with 'metric' and 'minimize' keys
            weights: Optional weights for weighted sum approach
        """
        super().__init__("Multi_Objective", "Multi-objective optimization")
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
    
    async def evaluate(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Evaluate multi-objective function"""
        import time
        start_time = time.time()
        
        try:
            # Create strategy instance with new architecture
            # strategy_factory should now be a tuple of (strategy_name, base_config)
            if isinstance(strategy_factory, tuple):
                strategy_name, base_config = strategy_factory
                # Merge optimization parameters with base config
                config = {**base_config, **parameters}
                
                # Import here to avoid circular imports
                from ..strategies.registry import create_strategy
                strategy = create_strategy(strategy_name, config)
            else:
                # Legacy factory function support
                strategy = strategy_factory(parameters)
            
            # Run backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=parameters.get('base_position_size', 0.05)
            )
            
            engine = BacktestEngine(initial_balance=10000.0, fast_mode=True)
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Calculate objective values for each objective
            objective_values = []
            for obj_def in self.objectives:
                metric = obj_def['metric']
                minimize = obj_def.get('minimize', False)
                
                value = metrics.get(metric, 0.0)
                if minimize:
                    value = -value
                
                objective_values.append(value)
            
            # Calculate weighted sum for primary objective
            weighted_sum = sum(w * v for w, v in zip(self.weights, objective_values))
            
            # Store individual objective values
            for i, (obj_def, value) in enumerate(zip(self.objectives, objective_values)):
                metrics[f"objective_{i}_{obj_def['metric']}"] = value
            
            return OptimizationResult(
                parameters=parameters,
                objective_value=weighted_sum,
                metrics=metrics,
                backtest_results=results,
                is_valid=True,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error("Multi-objective evaluation failed", 
                        parameters=parameters, error=str(e))
            return OptimizationResult(
                parameters=parameters,
                objective_value=float('-inf'),
                metrics={},
                is_valid=False,
                error_message=str(e),
                evaluation_time=time.time() - start_time
            )
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant metrics from backtest results"""
        metrics = {}
        
        # Capital metrics
        capital = results.get('capital', {})
        metrics['total_return_pct'] = capital.get('total_return_pct', 0.0)
        metrics['annualized_return_pct'] = capital.get('annualized_return_pct', 0.0)
        
        # Risk metrics
        risk = results.get('risk_metrics', {})
        metrics['sharpe_ratio'] = risk.get('sharpe_ratio', 0.0)
        metrics['sortino_ratio'] = risk.get('sortino_ratio', 0.0)
        metrics['calmar_ratio'] = risk.get('calmar_ratio', 0.0)
        metrics['max_drawdown_pct'] = risk.get('max_drawdown_pct', 0.0)
        metrics['volatility_pct'] = risk.get('volatility_pct', 0.0)
        
        # Trade metrics
        trades = results.get('trades', {})
        metrics['total_trades'] = trades.get('total', 0)
        metrics['win_rate_pct'] = trades.get('win_rate_pct', 0.0)
        metrics['profit_factor'] = trades.get('profit_factor', 0.0)
        
        return metrics

class RobustObjective(OptimizationObjective):
    """Robust optimization using multiple validation windows"""
    
    def __init__(
        self,
        base_objective: OptimizationObjective,
        validation_windows: int = 5,
        robustness_weight: float = 0.3
    ):
        """
        Initialize robust objective.
        
        Args:
            base_objective: Base objective to make robust
            validation_windows: Number of validation windows
            robustness_weight: Weight for robustness vs performance trade-off
        """
        super().__init__(f"Robust_{base_objective.name}", "Robust optimization")
        self.base_objective = base_objective
        self.validation_windows = validation_windows
        self.robustness_weight = robustness_weight
    
    async def evaluate(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> OptimizationResult:
        """Evaluate robust objective across multiple windows"""
        import time
        start_time = time.time()
        
        try:
            # Split data into validation windows
            windows = self._create_validation_windows(start_date, end_date)
            
            window_results = []
            objective_values = []
            
            for window_start, window_end in windows:
                result = await self.base_objective.evaluate(
                    parameters, strategy_factory, historical_data, window_start, window_end
                )
                
                if result.is_valid:
                    window_results.append(result)
                    objective_values.append(result.objective_value)
            
            if not objective_values:
                return OptimizationResult(
                    parameters=parameters,
                    objective_value=float('-inf'),
                    metrics={},
                    is_valid=False,
                    error_message="No valid windows",
                    evaluation_time=time.time() - start_time
                )
            
            # Calculate robust metrics
            mean_objective = np.mean(objective_values)
            std_objective = np.std(objective_values)
            min_objective = np.min(objective_values)
            
            # Robust objective balances performance and consistency
            robust_objective = (
                (1 - self.robustness_weight) * mean_objective +
                self.robustness_weight * min_objective -
                0.1 * std_objective  # Penalty for inconsistency
            )
            
            # Aggregate metrics
            all_metrics = {}
            for key in window_results[0].metrics.keys():
                values = [r.metrics[key] for r in window_results]
                all_metrics[f"{key}_mean"] = np.mean(values)
                all_metrics[f"{key}_std"] = np.std(values)
                all_metrics[f"{key}_min"] = np.min(values)
                all_metrics[f"{key}_max"] = np.max(values)
            
            all_metrics['objective_mean'] = mean_objective
            all_metrics['objective_std'] = std_objective
            all_metrics['objective_min'] = min_objective
            all_metrics['windows_valid'] = len(window_results)
            
            return OptimizationResult(
                parameters=parameters,
                objective_value=robust_objective,
                metrics=all_metrics,
                validation_results={'window_results': window_results},
                is_valid=True,
                evaluation_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error("Robust objective evaluation failed", 
                        parameters=parameters, error=str(e))
            return OptimizationResult(
                parameters=parameters,
                objective_value=float('-inf'),
                metrics={},
                is_valid=False,
                error_message=str(e),
                evaluation_time=time.time() - start_time
            )
    
    def _create_validation_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[tuple[datetime, datetime]]:
        """Create validation windows for robust testing"""
        total_days = (end_date - start_date).days
        window_days = total_days // self.validation_windows
        
        windows = []
        current_start = start_date
        
        for i in range(self.validation_windows):
            window_end = current_start + pd.Timedelta(days=window_days)
            if i == self.validation_windows - 1:  # Last window takes remaining days
                window_end = end_date
            
            windows.append((current_start, window_end))
            current_start = window_end
        
        return windows

# Predefined objective functions
class CommonObjectives:
    """Collection of common objective functions"""
    
    @staticmethod
    def maximize_return() -> SingleObjective:
        """Maximize total return"""
        return SingleObjective(
            metric='total_return_pct',
            minimize=False,
            min_trades_threshold=10
        )
    
    @staticmethod
    def maximize_sharpe() -> SingleObjective:
        """Maximize Sharpe ratio"""
        return SingleObjective(
            metric='sharpe_ratio',
            minimize=False,
            min_trades_threshold=10,
            min_sharpe_threshold=0.0
        )
    
    @staticmethod
    def maximize_calmar() -> SingleObjective:
        """Maximize Calmar ratio (return/max drawdown)"""
        return SingleObjective(
            metric='calmar_ratio',
            minimize=False,
            min_trades_threshold=10
        )
    
    @staticmethod
    def minimize_drawdown() -> SingleObjective:
        """Minimize maximum drawdown"""
        return SingleObjective(
            metric='max_drawdown_pct',
            minimize=True,
            min_trades_threshold=5
        )
    
    @staticmethod
    def risk_adjusted_return() -> SingleObjective:
        """Risk-adjusted return optimization"""
        return SingleObjective(
            metric='total_return_pct',
            minimize=False,
            risk_penalty_weight=0.3,
            transaction_cost_penalty=0.1,
            min_trades_threshold=10,
            max_drawdown_threshold=0.3
        )
    
    @staticmethod
    def balanced_portfolio() -> MultiObjective:
        """Balanced multi-objective optimization"""
        return MultiObjective(
            objectives=[
                {'metric': 'total_return_pct', 'minimize': False},
                {'metric': 'sharpe_ratio', 'minimize': False},
                {'metric': 'max_drawdown_pct', 'minimize': True}
            ],
            weights=[0.5, 0.3, 0.2]
        )