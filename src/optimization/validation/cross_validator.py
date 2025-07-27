"""
Cross Validation for Time Series

Implements time series cross-validation to assess parameter robustness
across different market conditions and time periods.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog
import asyncio

from ..objectives import OptimizationObjective, OptimizationResult

logger = structlog.get_logger()

@dataclass
class CrossValidationFold:
    """Represents a single cross-validation fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime  
    test_end: datetime
    train_data: Dict[str, pd.DataFrame]
    test_data: Dict[str, pd.DataFrame]

class CrossValidator:
    """
    Time Series Cross-Validator for parameter validation.
    
    Uses time-series aware cross-validation that respects temporal ordering
    and prevents data leakage from future to past.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size_ratio: float = 0.2,
        gap_days: int = 0,
        method: str = 'rolling'  # 'rolling' or 'expanding'
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size_ratio: Ratio of data used for testing in each split
            gap_days: Gap between train and test sets (to avoid lookahead bias)
            method: 'rolling' (fixed window) or 'expanding' (growing window)
        """
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.gap_days = gap_days
        self.method = method
        
        self.folds: List[CrossValidationFold] = []
    
    def create_folds(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> List[CrossValidationFold]:
        """
        Create time-series cross-validation folds.
        
        Args:
            historical_data: Market data
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of cross-validation folds
        """
        total_days = (end_date - start_date).days
        test_days = int(total_days * self.test_size_ratio)
        
        if self.method == 'rolling':
            train_days = int(total_days * (1 - self.test_size_ratio)) // self.n_splits
        else:  # expanding
            train_days = None  # Will be calculated per fold
        
        folds = []
        
        for i in range(self.n_splits):
            if self.method == 'rolling':
                # Rolling window: fixed-size training window
                fold_start = start_date + timedelta(days=i * (train_days + test_days + self.gap_days))
                train_start = fold_start
                train_end = train_start + timedelta(days=train_days)
                test_start = train_end + timedelta(days=self.gap_days)
                test_end = test_start + timedelta(days=test_days)
            else:
                # Expanding window: growing training window
                train_start = start_date
                fold_end = start_date + timedelta(days=(i + 1) * (total_days // self.n_splits))
                test_end = min(fold_end, end_date)
                test_start = test_end - timedelta(days=test_days)
                train_end = test_start - timedelta(days=self.gap_days)
            
            # Ensure we don't exceed bounds
            if test_end > end_date:
                break
            
            if train_start >= train_end or test_start >= test_end:
                continue
            
            # Extract data for this fold
            train_data = self._extract_data_window(historical_data, train_start, train_end)
            test_data = self._extract_data_window(historical_data, test_start, test_end)
            
            if not train_data or not test_data:
                continue
            
            fold = CrossValidationFold(
                fold_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_data=train_data,
                test_data=test_data
            )
            
            folds.append(fold)
        
        self.folds = folds
        
        logger.info("Created cross-validation folds",
                   n_folds=len(folds),
                   method=self.method,
                   avg_train_days=(train_end - train_start).days if folds else 0,
                   test_days=test_days)
        
        return folds
    
    async def validate_parameters(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """
        Validate parameters using cross-validation.
        
        Args:
            parameters: Parameters to validate
            strategy_factory: Function to create strategy instances
            historical_data: Market data
            start_date: Start date for validation
            end_date: End date for validation
            objective: Objective function
            
        Returns:
            Cross-validation results and metrics
        """
        
        logger.info("Starting cross-validation",
                   n_splits=self.n_splits,
                   parameters=parameters)
        
        # Create folds if not already done
        if not self.folds:
            self.create_folds(historical_data, start_date, end_date)
        
        if not self.folds:
            raise ValueError("No valid cross-validation folds created")
        
        # Evaluate each fold
        fold_results = []
        
        for fold in self.folds:
            logger.info("Evaluating fold",
                       fold_id=fold.fold_id,
                       train_period=(fold.train_start, fold.train_end),
                       test_period=(fold.test_start, fold.test_end))
            
            try:
                # Evaluate on test set
                result = await objective.evaluate(
                    parameters=parameters,
                    strategy_factory=strategy_factory,
                    historical_data=fold.test_data,
                    start_date=fold.test_start,
                    end_date=fold.test_end
                )
                
                fold_results.append({
                    'fold_id': fold.fold_id,
                    'result': result,
                    'train_period': (fold.train_start, fold.train_end),
                    'test_period': (fold.test_start, fold.test_end)
                })
                
                logger.info("Fold evaluation completed",
                           fold_id=fold.fold_id,
                           objective_value=result.objective_value,
                           is_valid=result.is_valid)
                
            except Exception as e:
                logger.error("Fold evaluation failed",
                           fold_id=fold.fold_id, error=str(e))
                continue
        
        # Analyze results
        analysis = self._analyze_cross_validation_results(fold_results)
        
        logger.info("Cross-validation completed",
                   valid_folds=len([r for r in fold_results if r['result'].is_valid]),
                   avg_score=analysis.get('mean_score', 0),
                   score_std=analysis.get('std_score', 0))
        
        return analysis
    
    def _extract_data_window(
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
            
            # Convert timestamps
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filter by date range
            mask = (timestamps >= start_date) & (timestamps <= end_date)
            window_df = df[mask].copy()
            
            if len(window_df) >= 5:  # Minimum data points
                window_data[symbol] = window_df
        
        return window_data
    
    def _analyze_cross_validation_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cross-validation results"""
        
        if not fold_results:
            return {'error': 'No fold results available'}
        
        # Extract valid results
        valid_results = [r for r in fold_results if r['result'].is_valid]
        
        if not valid_results:
            return {'error': 'No valid fold results'}
        
        # Extract scores and metrics
        scores = [r['result'].objective_value for r in valid_results]
        
        # Extract additional metrics from each fold
        all_metrics = {}
        for metric_name in valid_results[0]['result'].metrics.keys():
            metric_values = [r['result'].metrics[metric_name] for r in valid_results]
            all_metrics[metric_name] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }
        
        analysis = {
            'total_folds': len(fold_results),
            'valid_folds': len(valid_results),
            'validation_method': f'Time Series CV ({self.method})',
            
            # Score statistics
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_range': np.max(scores) - np.min(scores),
            
            # Reliability metrics
            'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else float('inf'),
            'stability_score': 1.0 / (1.0 + np.std(scores) / abs(np.mean(scores))) if np.mean(scores) != 0 else 0,
            
            # Confidence intervals (assuming normal distribution)
            'confidence_interval_95': self._calculate_confidence_interval(scores, 0.95),
            'confidence_interval_90': self._calculate_confidence_interval(scores, 0.90),
            
            # Individual fold results
            'fold_scores': scores,
            'fold_details': [
                {
                    'fold_id': r['fold_id'],
                    'score': r['result'].objective_value,
                    'train_period': r['train_period'],
                    'test_period': r['test_period'],
                    'metrics': r['result'].metrics
                }
                for r in valid_results
            ],
            
            # Aggregated metrics
            'aggregated_metrics': all_metrics
        }
        
        # Add time-based analysis
        analysis.update(self._analyze_temporal_patterns(valid_results))
        
        return analysis
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for scores"""
        
        if len(scores) < 2:
            mean_score = scores[0] if scores else 0
            return (mean_score, mean_score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)  # Sample standard deviation
        n = len(scores)
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * std_score / np.sqrt(n)
        
        return (mean_score - margin_error, mean_score + margin_error)
    
    def _analyze_temporal_patterns(self, valid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in cross-validation results"""
        
        if len(valid_results) < 3:
            return {'temporal_analysis': 'Insufficient data for temporal analysis'}
        
        # Sort results by test period start
        sorted_results = sorted(valid_results, key=lambda x: x['test_period'][0])
        scores = [r['result'].objective_value for r in sorted_results]
        
        # Calculate trend
        x = np.arange(len(scores))
        trend_slope = np.polyfit(x, scores, 1)[0]
        
        # Calculate autocorrelation
        autocorr = np.corrcoef(scores[:-1], scores[1:])[0, 1] if len(scores) > 2 else 0
        
        # Performance consistency over time
        score_diff = np.diff(scores)
        volatility = np.std(score_diff)
        
        return {
            'temporal_analysis': {
                'trend_slope': trend_slope,
                'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                'autocorrelation': autocorr,
                'volatility': volatility,
                'score_progression': scores,
                'most_stable_period': self._find_most_stable_period(sorted_results),
                'performance_consistency': 1.0 / (1.0 + volatility) if volatility > 0 else 1.0
            }
        }
    
    def _find_most_stable_period(self, sorted_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most stable performance period"""
        
        if len(sorted_results) < 3:
            return {}
        
        window_size = 3
        min_volatility = float('inf')
        best_period = None
        
        for i in range(len(sorted_results) - window_size + 1):
            window_scores = [sorted_results[i + j]['result'].objective_value for j in range(window_size)]
            volatility = np.std(window_scores)
            
            if volatility < min_volatility:
                min_volatility = volatility
                best_period = {
                    'start_fold': sorted_results[i]['fold_id'],
                    'end_fold': sorted_results[i + window_size - 1]['fold_id'],
                    'period_start': sorted_results[i]['test_period'][0],
                    'period_end': sorted_results[i + window_size - 1]['test_period'][1],
                    'volatility': volatility,
                    'mean_score': np.mean(window_scores)
                }
        
        return best_period or {}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of cross-validation results"""
        
        if not hasattr(self, '_last_analysis') or not self._last_analysis:
            return {'status': 'No cross-validation performed'}
        
        analysis = self._last_analysis
        
        stability_score = analysis.get('stability_score', 0)
        mean_score = analysis.get('mean_score', 0)
        
        # Determine reliability rating
        if stability_score > 0.8 and analysis.get('valid_folds', 0) >= 4:
            reliability = 'HIGH'
        elif stability_score > 0.6 and analysis.get('valid_folds', 0) >= 3:
            reliability = 'MEDIUM'
        else:
            reliability = 'LOW'
        
        return {
            'validation_method': analysis.get('validation_method', 'Time Series CV'),
            'folds_validated': analysis.get('valid_folds', 0),
            'mean_performance': mean_score,
            'performance_stability': stability_score,
            'reliability_rating': reliability,
            'confidence_interval_95': analysis.get('confidence_interval_95', (0, 0)),
            'temporal_trend': analysis.get('temporal_analysis', {}).get('trend_direction', 'unknown'),
            'recommendation': self._get_recommendation(analysis)
        }
    
    def _get_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get recommendation based on cross-validation results"""
        
        stability_score = analysis.get('stability_score', 0)
        valid_folds = analysis.get('valid_folds', 0)
        temporal_analysis = analysis.get('temporal_analysis', {})
        
        if valid_folds < 3:
            return "INSUFFICIENT: Need more validation folds for reliable assessment."
        elif stability_score > 0.8:
            return "ROBUST: Parameters show consistent performance across time periods."
        elif stability_score > 0.6:
            if temporal_analysis.get('trend_direction') == 'declining':
                return "CAUTION: Stable but showing declining trend over time."
            else:
                return "ACCEPTABLE: Reasonably stable with minor variations."
        else:
            return "UNSTABLE: Parameters show high variability across time periods."

# Store last analysis for summary
CrossValidator._last_analysis = None

def store_analysis(self, analysis):
    """Store analysis for later summary"""
    CrossValidator._last_analysis = analysis

# Monkey patch to store analysis
original_analyze = CrossValidator._analyze_cross_validation_results
def patched_analyze(self, fold_results):
    analysis = original_analyze(self, fold_results)
    CrossValidator._last_analysis = analysis
    return analysis

CrossValidator._analyze_cross_validation_results = patched_analyze