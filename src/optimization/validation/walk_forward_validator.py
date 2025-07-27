"""
Walk-Forward Validation

Implements walk-forward analysis to prevent overfitting by simulating
real-time trading conditions where future data is not available.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog
import asyncio

from ..objectives import OptimizationObjective, OptimizationResult

logger = structlog.get_logger()

@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""
    window_id: int
    optimization_start: datetime
    optimization_end: datetime
    validation_start: datetime
    validation_end: datetime
    optimization_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    validation_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

@dataclass
class WalkForwardResult:
    """Result from walk-forward validation"""
    window: WalkForwardWindow
    best_parameters: Dict[str, Any]
    optimization_result: OptimizationResult
    validation_result: OptimizationResult
    performance_degradation: float = 0.0

class WalkForwardValidator:
    """
    Walk-Forward Validator for robust parameter optimization.
    
    Simulates real trading conditions by:
    1. Dividing data into multiple overlapping windows
    2. Optimizing parameters on historical data (in-sample)
    3. Testing optimized parameters on future data (out-of-sample)
    4. Measuring performance degradation and consistency
    """
    
    def __init__(
        self,
        num_windows: int = 5,
        optimization_ratio: float = 0.7,
        overlap_ratio: float = 0.2,
        min_window_days: int = 30,
        reoptimization_frequency: Optional[int] = None,
        stability_threshold: float = 0.1
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            num_windows: Number of walk-forward windows
            optimization_ratio: Ratio of window used for optimization (vs validation)
            overlap_ratio: Overlap between consecutive windows (0-1)
            min_window_days: Minimum days per window
            reoptimization_frequency: How often to reoptimize (None = every window)
            stability_threshold: Threshold for parameter stability detection
        """
        self.num_windows = num_windows
        self.optimization_ratio = optimization_ratio
        self.overlap_ratio = overlap_ratio
        self.min_window_days = min_window_days
        self.reoptimization_frequency = reoptimization_frequency
        self.stability_threshold = stability_threshold
        
        # Validation state
        self.windows: List[WalkForwardWindow] = []
        self.results: List[WalkForwardResult] = []
        
    def create_windows(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows from historical data.
        
        Args:
            historical_data: Market data
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of walk-forward windows
        """
        total_days = (end_date - start_date).days
        
        if total_days < self.min_window_days * self.num_windows:
            raise ValueError(
                f"Insufficient data: {total_days} days < {self.min_window_days * self.num_windows} required"
            )
        
        # Calculate window parameters
        base_window_days = total_days // self.num_windows
        overlap_days = int(base_window_days * self.overlap_ratio)
        step_days = base_window_days - overlap_days
        
        optimization_days = int(base_window_days * self.optimization_ratio)
        validation_days = base_window_days - optimization_days
        
        windows = []
        
        for i in range(self.num_windows):
            # Calculate window boundaries
            window_start = start_date + timedelta(days=i * step_days)
            window_end = window_start + timedelta(days=base_window_days)
            
            # Ensure we don't exceed end date
            if window_end > end_date:
                window_end = end_date
                window_start = max(start_date, window_end - timedelta(days=base_window_days))
            
            # Split into optimization and validation periods
            opt_end = window_start + timedelta(days=optimization_days)
            val_start = opt_end
            val_end = window_end
            
            # Skip if validation period is too short
            if (val_end - val_start).days < 5:
                continue
            
            # Extract data for this window
            opt_data = self._extract_window_data(historical_data, window_start, opt_end)
            val_data = self._extract_window_data(historical_data, val_start, val_end)
            
            # Skip if insufficient data
            if not opt_data or not val_data:
                continue
            
            window = WalkForwardWindow(
                window_id=i,
                optimization_start=window_start,
                optimization_end=opt_end,
                validation_start=val_start,
                validation_end=val_end,
                optimization_data=opt_data,
                validation_data=val_data
            )
            
            windows.append(window)
        
        self.windows = windows
        
        logger.info("Created walk-forward windows",
                   num_windows=len(windows),
                   avg_opt_days=optimization_days,
                   avg_val_days=validation_days,
                   overlap_days=overlap_days)
        
        return windows
    
    async def validate_parameters(
        self,
        optimizer: Any,  # BaseOptimizer
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation of optimization process.
        
        Args:
            optimizer: Parameter optimizer to validate
            strategy_factory: Function to create strategy instances
            historical_data: Market data
            start_date: Start date for validation
            end_date: End date for validation
            objective: Objective function for evaluation
            
        Returns:
            Validation results and metrics
        """
        
        logger.info("Starting walk-forward validation",
                   num_windows=self.num_windows,
                   date_range=(start_date, end_date))
        
        # Create windows if not already done
        if not self.windows:
            self.create_windows(historical_data, start_date, end_date)
        
        if not self.windows:
            raise ValueError("No valid walk-forward windows created")
        
        self.results = []
        previous_best_params = None
        
        # Process each window
        for i, window in enumerate(self.windows):
            logger.info("Processing walk-forward window",
                       window_id=window.window_id,
                       optimization_period=(window.optimization_start, window.optimization_end),
                       validation_period=(window.validation_start, window.validation_end))
            
            # Decide whether to reoptimize
            should_reoptimize = (
                previous_best_params is None or
                self.reoptimization_frequency is None or
                i % self.reoptimization_frequency == 0
            )
            
            if should_reoptimize:
                # Run optimization on this window's training data
                logger.info("Running optimization for window", window_id=window.window_id)
                
                try:
                    optimization_result = await optimizer.optimize(
                        strategy_factory=strategy_factory,
                        historical_data=window.optimization_data,
                        start_date=window.optimization_start,
                        end_date=window.optimization_end
                    )
                    
                    best_parameters = optimization_result.parameters
                    
                except Exception as e:
                    logger.error("Optimization failed for window",
                               window_id=window.window_id, error=str(e))
                    continue
            else:
                # Use previous parameters
                best_parameters = previous_best_params
                optimization_result = None
            
            # Validate on out-of-sample data
            logger.info("Validating parameters on out-of-sample data",
                       window_id=window.window_id)
            
            try:
                validation_result = await objective.evaluate(
                    parameters=best_parameters,
                    strategy_factory=strategy_factory,
                    historical_data=window.validation_data,
                    start_date=window.validation_start,
                    end_date=window.validation_end
                )
                
                # Calculate performance degradation
                if optimization_result:
                    in_sample_performance = optimization_result.objective_value
                    out_sample_performance = validation_result.objective_value
                    degradation = (in_sample_performance - out_sample_performance) / abs(in_sample_performance)
                else:
                    degradation = 0.0
                
                # Store result
                wf_result = WalkForwardResult(
                    window=window,
                    best_parameters=best_parameters,
                    optimization_result=optimization_result,
                    validation_result=validation_result,
                    performance_degradation=degradation
                )
                
                self.results.append(wf_result)
                previous_best_params = best_parameters
                
                logger.info("Window validation completed",
                           window_id=window.window_id,
                           out_sample_objective=validation_result.objective_value,
                           degradation_pct=degradation * 100)
                
            except Exception as e:
                logger.error("Validation failed for window",
                           window_id=window.window_id, error=str(e))
                continue
        
        # Analyze results
        analysis = self._analyze_walk_forward_results()
        
        logger.info("Walk-forward validation completed",
                   windows_processed=len(self.results),
                   avg_degradation=analysis.get('avg_performance_degradation', 0) * 100,
                   consistency_score=analysis.get('consistency_score', 0))
        
        return analysis
    
    def _extract_window_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Extract data for a specific time window"""
        
        window_data = {}
        
        for symbol, df in historical_data.items():
            if df.empty:
                continue
            
            # Convert timestamps to datetime
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter by date range
            mask = (timestamps >= start_date) & (timestamps <= end_date)
            window_df = df[mask].copy()
            
            if len(window_df) >= 10:  # Minimum data points
                window_data[symbol] = window_df
        
        return window_data
    
    def _analyze_walk_forward_results(self) -> Dict[str, Any]:
        """Analyze walk-forward validation results"""
        
        if not self.results:
            return {'error': 'No validation results available'}
        
        # Extract metrics
        validation_objectives = []
        degradations = []
        parameter_changes = []
        
        for i, result in enumerate(self.results):
            if result.validation_result.is_valid:
                validation_objectives.append(result.validation_result.objective_value)
                degradations.append(result.performance_degradation)
                
                # Track parameter stability
                if i > 0:
                    param_change = self._calculate_parameter_change(
                        self.results[i-1].best_parameters,
                        result.best_parameters
                    )
                    parameter_changes.append(param_change)
        
        if not validation_objectives:
            return {'error': 'No valid validation results'}
        
        # Calculate statistics
        analysis = {
            'total_windows': len(self.results),
            'valid_windows': len(validation_objectives),
            
            # Performance metrics
            'avg_out_sample_objective': np.mean(validation_objectives),
            'std_out_sample_objective': np.std(validation_objectives),
            'min_out_sample_objective': np.min(validation_objectives),
            'max_out_sample_objective': np.max(validation_objectives),
            
            # Performance degradation
            'avg_performance_degradation': np.mean(degradations),
            'std_performance_degradation': np.std(degradations),
            'max_performance_degradation': np.max(degradations),
            'windows_with_degradation': sum(1 for d in degradations if d > 0),
            
            # Consistency metrics
            'consistency_score': self._calculate_consistency_score(validation_objectives),
            'stability_score': np.mean(parameter_changes) if parameter_changes else 1.0,
            
            # Risk metrics
            'worst_case_objective': np.min(validation_objectives),
            'objective_volatility': np.std(validation_objectives) / np.mean(validation_objectives) if np.mean(validation_objectives) != 0 else float('inf'),
            
            # Window-by-window results
            'window_results': [
                {
                    'window_id': r.window.window_id,
                    'validation_objective': r.validation_result.objective_value,
                    'degradation': r.performance_degradation,
                    'is_valid': r.validation_result.is_valid
                }
                for r in self.results
            ]
        }
        
        # Overfitting detection
        analysis['overfitting_detected'] = self._detect_overfitting(analysis)
        analysis['robustness_score'] = self._calculate_robustness_score(analysis)
        
        return analysis
    
    def _calculate_parameter_change(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate normalized change between parameter sets"""
        
        if not params1 or not params2:
            return 1.0
        
        changes = []
        
        for param_name in set(params1.keys()) | set(params2.keys()):
            val1 = params1.get(param_name, 0)
            val2 = params2.get(param_name, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 != 0:
                    change = abs(val2 - val1) / abs(val1)
                else:
                    change = 1.0 if val2 != 0 else 0.0
            else:
                change = 1.0 if val1 != val2 else 0.0
            
            changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _calculate_consistency_score(self, objectives: List[float]) -> float:
        """Calculate consistency score (inverse of coefficient of variation)"""
        
        if len(objectives) < 2:
            return 1.0
        
        mean_obj = np.mean(objectives)
        if mean_obj == 0:
            return 0.0
        
        cv = np.std(objectives) / abs(mean_obj)
        return 1.0 / (1.0 + cv)  # Higher score = more consistent
    
    def _detect_overfitting(self, analysis: Dict[str, Any]) -> bool:
        """Detect signs of overfitting"""
        
        # Check for excessive performance degradation
        avg_degradation = analysis.get('avg_performance_degradation', 0)
        max_degradation = analysis.get('max_performance_degradation', 0)
        
        # Check for high volatility in out-of-sample performance
        objective_volatility = analysis.get('objective_volatility', 0)
        consistency_score = analysis.get('consistency_score', 1)
        
        overfitting_signals = [
            avg_degradation > 0.2,  # Average degradation > 20%
            max_degradation > 0.5,  # Any window with > 50% degradation
            objective_volatility > 1.0,  # High volatility
            consistency_score < 0.3  # Low consistency
        ]
        
        return sum(overfitting_signals) >= 2  # Multiple signals indicate overfitting
    
    def _calculate_robustness_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall robustness score (0-1)"""
        
        # Combine multiple factors
        consistency = analysis.get('consistency_score', 0)
        stability = analysis.get('stability_score', 0)
        degradation_penalty = max(0, 1 - analysis.get('avg_performance_degradation', 0) * 2)
        volatility_penalty = max(0, 1 - analysis.get('objective_volatility', 0) / 2)
        
        robustness = (consistency + stability + degradation_penalty + volatility_penalty) / 4
        return max(0, min(1, robustness))
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        
        if not self.results:
            return {'status': 'No validation performed'}
        
        analysis = self._analyze_walk_forward_results()
        
        return {
            'validation_method': 'Walk-Forward Analysis',
            'windows_analyzed': len(self.results),
            'avg_out_sample_performance': analysis.get('avg_out_sample_objective', 0),
            'performance_consistency': analysis.get('consistency_score', 0),
            'parameter_stability': analysis.get('stability_score', 0),
            'robustness_score': analysis.get('robustness_score', 0),
            'overfitting_detected': analysis.get('overfitting_detected', False),
            'avg_degradation_pct': analysis.get('avg_performance_degradation', 0) * 100,
            'recommendation': self._get_recommendation(analysis)
        }
    
    def _get_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get recommendation based on validation results"""
        
        robustness_score = analysis.get('robustness_score', 0)
        overfitting_detected = analysis.get('overfitting_detected', False)
        
        if overfitting_detected:
            return "CAUTION: Overfitting detected. Consider regularization or simpler parameters."
        elif robustness_score > 0.8:
            return "EXCELLENT: Parameters show strong robustness across time periods."
        elif robustness_score > 0.6:
            return "GOOD: Parameters show reasonable robustness with minor concerns."
        elif robustness_score > 0.4:
            return "MODERATE: Parameters show mixed performance. Consider refinement."
        else:
            return "POOR: Parameters lack robustness. Significant revision needed."