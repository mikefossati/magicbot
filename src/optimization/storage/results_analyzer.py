"""
Optimization Results Analyzer

Comprehensive analysis tools for optimization results including
parameter sensitivity, convergence analysis, and performance comparison.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from .postgres_database import OptimizationDatabase, OptimizationRun, ParameterEvaluation

logger = structlog.get_logger()

class ResultsAnalyzer:
    """
    Comprehensive analyzer for optimization results.
    
    Features:
    - Parameter sensitivity analysis
    - Convergence analysis
    - Performance comparison across runs
    - Statistical significance testing
    - Visualization generation
    - Regression analysis
    """
    
    def __init__(self, database: OptimizationDatabase):
        """
        Initialize results analyzer.
        
        Args:
            database: Optimization database instance
        """
        self.database = database
    
    async def analyze_parameter_sensitivity(
        self,
        run_id: str,
        top_n_parameters: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze parameter sensitivity using random forest importance.
        
        Args:
            run_id: Optimization run ID
            top_n_parameters: Number of top parameters to analyze
            
        Returns:
            Parameter sensitivity analysis results
        """
        
        logger.info("Analyzing parameter sensitivity", run_id=run_id)
        
        # Get evaluation data
        evaluations = await self.database.get_parameter_evaluations(run_id, valid_only=True)
        
        if len(evaluations) < 10:
            return {'error': 'Insufficient data for sensitivity analysis'}
        
        # Prepare data
        df = self._evaluations_to_dataframe(evaluations)
        
        # Separate features and target
        param_columns = [col for col in df.columns if col.startswith('param_')]
        X = df[param_columns]
        y = df['objective_value']
        
        # Handle categorical parameters
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Train random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_encoded, y)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'parameter': X_encoded.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate R² score
        y_pred = rf.predict(X_encoded)
        r2 = r2_score(y, y_pred)
        
        # Calculate parameter correlations with objective
        correlations = {}
        for col in param_columns:
            if df[col].dtype in ['int64', 'float64']:
                corr = df[col].corr(df['objective_value'])
                correlations[col] = corr
        
        # Statistical significance tests
        significance_tests = {}
        for col in param_columns:
            if df[col].dtype in ['int64', 'float64']:
                # Pearson correlation test
                corr, p_value = stats.pearsonr(df[col], df['objective_value'])
                significance_tests[col] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Parameter ranges and distributions
        parameter_stats = {}
        for col in param_columns:
            if df[col].dtype in ['int64', 'float64']:
                parameter_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range': df[col].max() - df[col].min(),
                    'unique_values': df[col].nunique()
                }
            else:
                value_counts = df[col].value_counts()
                parameter_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': value_counts.index[0],
                    'distribution': value_counts.to_dict()
                }
        
        # Identify most important parameters
        top_parameters = feature_importance.head(top_n_parameters)
        
        return {
            'run_id': run_id,
            'model_performance': {
                'r2_score': r2,
                'n_samples': len(evaluations),
                'n_features': len(X_encoded.columns)
            },
            'feature_importance': top_parameters.to_dict('records'),
            'parameter_correlations': correlations,
            'significance_tests': significance_tests,
            'parameter_statistics': parameter_stats,
            'top_parameters': top_parameters['parameter'].tolist()
        }
    
    async def analyze_convergence(self, run_id: str) -> Dict[str, Any]:
        """
        Analyze optimization convergence characteristics.
        
        Args:
            run_id: Optimization run ID
            
        Returns:
            Convergence analysis results
        """
        
        logger.info("Analyzing convergence", run_id=run_id)
        
        evaluations = await self.database.get_parameter_evaluations(run_id, valid_only=True)
        
        if len(evaluations) < 5:
            return {'error': 'Insufficient data for convergence analysis'}
        
        # Sort by iteration
        evaluations.sort(key=lambda x: x.iteration)
        
        # Extract objective values over time
        iterations = [e.iteration for e in evaluations]
        objectives = [e.objective_value for e in evaluations]
        
        # Calculate running best
        running_best = []
        current_best = float('-inf')
        for obj in objectives:
            if obj > current_best:
                current_best = obj
            running_best.append(current_best)
        
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(running_best)):
            if running_best[i] > running_best[i-1]:
                improvement = running_best[i] - running_best[i-1]
                improvements.append(improvement)
        
        # Convergence metrics
        total_improvement = running_best[-1] - running_best[0] if running_best else 0
        final_plateau_length = self._calculate_plateau_length(running_best)
        convergence_rate = self._calculate_convergence_rate(running_best)
        
        # Stagnation detection
        stagnation_periods = self._detect_stagnation_periods(running_best)
        
        # Statistical measures
        objective_trend = self._calculate_trend(iterations, objectives)
        
        return {
            'run_id': run_id,
            'convergence_metrics': {
                'total_improvement': total_improvement,
                'improvement_rate': np.mean(improvements) if improvements else 0,
                'convergence_rate': convergence_rate,
                'final_plateau_length': final_plateau_length,
                'n_improvements': len(improvements),
                'improvement_frequency': len(improvements) / len(evaluations)
            },
            'objective_progression': {
                'iterations': iterations,
                'objectives': objectives,
                'running_best': running_best
            },
            'trend_analysis': objective_trend,
            'stagnation_periods': stagnation_periods,
            'convergence_assessment': self._assess_convergence(running_best, stagnation_periods)
        }
    
    async def compare_optimization_runs(
        self,
        run_ids: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple optimization runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison analysis results
        """
        
        logger.info("Comparing optimization runs", run_ids=run_ids)
        
        if metrics is None:
            metrics = ['best_objective_value', 'total_evaluations', 'convergence_rate']
        
        comparison_data = []
        
        for run_id in run_ids:
            run = await self.database.get_optimization_run(run_id)
            if not run:
                continue
            
            evaluations = await self.database.get_parameter_evaluations(run_id, valid_only=True)
            
            # Calculate run metrics
            run_metrics = {
                'run_id': run_id,
                'strategy_name': run.strategy_name,
                'optimizer_type': run.optimizer_type,
                'best_objective_value': run.best_objective_value or 0,
                'total_evaluations': run.total_evaluations,
                'valid_evaluations': len(evaluations),
                'success_rate': len(evaluations) / run.total_evaluations if run.total_evaluations > 0 else 0,
                'optimization_time': (run.end_time - run.start_time).total_seconds() if run.end_time else None
            }
            
            # Add convergence metrics
            if evaluations:
                convergence = await self.analyze_convergence(run_id)
                if 'convergence_metrics' in convergence:
                    run_metrics.update(convergence['convergence_metrics'])
            
            comparison_data.append(run_metrics)
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical comparison
        statistical_tests = {}
        for metric in metrics:
            if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
                values = df[metric].dropna()
                if len(values) > 1:
                    # ANOVA test if more than 2 groups
                    if len(values) > 2:
                        f_stat, p_value = stats.f_oneway(*[values])
                        statistical_tests[metric] = {
                            'test': 'ANOVA',
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    else:
                        # T-test for 2 groups
                        t_stat, p_value = stats.ttest_ind(values[:len(values)//2], values[len(values)//2:])
                        statistical_tests[metric] = {
                            'test': 't-test',
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        # Rankings
        rankings = {}
        for metric in metrics:
            if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
                rankings[metric] = df.nlargest(len(df), metric)[['run_id', metric]].to_dict('records')
        
        # Summary statistics
        summary_stats = {}
        for metric in metrics:
            if metric in df.columns and df[metric].dtype in ['int64', 'float64']:
                summary_stats[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'range': df[metric].max() - df[metric].min()
                }
        
        return {
            'comparison_data': comparison_data,
            'statistical_tests': statistical_tests,
            'rankings': rankings,
            'summary_statistics': summary_stats,
            'best_run': df.loc[df['best_objective_value'].idxmax()].to_dict() if not df.empty else None
        }
    
    async def analyze_strategy_performance(
        self,
        strategy_name: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to analyze
            days_back: Number of days to look back
            
        Returns:
            Strategy performance analysis
        """
        
        logger.info("Analyzing strategy performance", strategy=strategy_name, days_back=days_back)
        
        runs = await self.database.get_optimization_runs(
            strategy_name=strategy_name,
            days_back=days_back,
            status='completed'
        )
        
        if not runs:
            return {'error': f'No completed runs found for strategy {strategy_name}'}
        
        # Performance over time
        performance_data = []
        for run in runs:
            performance_data.append({
                'run_id': run.run_id,
                'start_time': run.start_time,
                'best_objective_value': run.best_objective_value or 0,
                'total_evaluations': run.total_evaluations,
                'optimizer_type': run.optimizer_type
            })
        
        df = pd.DataFrame(performance_data)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df = df.sort_values('start_time')
        
        # Performance trends
        time_trend = self._calculate_trend(
            df['start_time'].astype(int) // 10**9,  # Convert to unix timestamp
            df['best_objective_value']
        )
        
        # Optimizer comparison
        optimizer_performance = df.groupby('optimizer_type').agg({
            'best_objective_value': ['mean', 'std', 'max', 'count'],
            'total_evaluations': 'mean'
        }).round(4)
        
        # Best parameters analysis
        best_parameters = await self.database.get_best_parameters_by_strategy(strategy_name, top_n=10)
        
        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability(best_parameters)
        
        return {
            'strategy_name': strategy_name,
            'analysis_period': {
                'days_back': days_back,
                'total_runs': len(runs),
                'date_range': {
                    'start': min(run.start_time for run in runs),
                    'end': max(run.start_time for run in runs)
                }
            },
            'performance_trends': {
                'time_trend': time_trend,
                'best_performance': df['best_objective_value'].max(),
                'average_performance': df['best_objective_value'].mean(),
                'performance_volatility': df['best_objective_value'].std(),
                'improvement_over_time': df['best_objective_value'].iloc[-1] - df['best_objective_value'].iloc[0] if len(df) > 1 else 0
            },
            'optimizer_comparison': optimizer_performance.to_dict(),
            'best_parameters': best_parameters,
            'parameter_stability': parameter_stability
        }
    
    def _evaluations_to_dataframe(self, evaluations: List[ParameterEvaluation]) -> pd.DataFrame:
        """Convert evaluations to DataFrame with expanded columns"""
        
        data = []
        for eval in evaluations:
            row = {
                'objective_value': eval.objective_value,
                'iteration': eval.iteration,
                'evaluation_time': eval.evaluation_time,
                'timestamp': eval.timestamp
            }
            
            # Add parameters with prefix
            for key, value in eval.parameters.items():
                row[f'param_{key}'] = value
            
            # Add metrics with prefix
            for key, value in eval.metrics.items():
                row[f'metric_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _calculate_plateau_length(self, running_best: List[float]) -> int:
        """Calculate length of final plateau in convergence"""
        
        if len(running_best) < 2:
            return 0
        
        final_best = running_best[-1]
        plateau_length = 0
        
        for i in range(len(running_best) - 1, -1, -1):
            if running_best[i] == final_best:
                plateau_length += 1
            else:
                break
        
        return plateau_length
    
    def _calculate_convergence_rate(self, running_best: List[float]) -> float:
        """Calculate convergence rate"""
        
        if len(running_best) < 2:
            return 0.0
        
        # Calculate rate of improvement
        total_improvement = running_best[-1] - running_best[0]
        total_iterations = len(running_best)
        
        return total_improvement / total_iterations if total_iterations > 0 else 0.0
    
    def _calculate_trend(self, x: List[float], y: List[float]) -> Dict[str, float]:
        """Calculate linear trend"""
        
        if len(x) < 2 or len(y) < 2:
            return {'slope': 0, 'r_squared': 0, 'p_value': 1}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err
        }
    
    def _detect_stagnation_periods(self, running_best: List[float], min_length: int = 10) -> List[Dict[str, int]]:
        """Detect periods of stagnation in optimization"""
        
        stagnation_periods = []
        current_start = None
        current_value = None
        
        for i, value in enumerate(running_best):
            if current_value is None or value != current_value:
                # End previous stagnation period
                if current_start is not None and i - current_start >= min_length:
                    stagnation_periods.append({
                        'start': current_start,
                        'end': i - 1,
                        'length': i - current_start,
                        'value': current_value
                    })
                
                # Start new period
                current_start = i
                current_value = value
        
        # Check final period
        if current_start is not None and len(running_best) - current_start >= min_length:
            stagnation_periods.append({
                'start': current_start,
                'end': len(running_best) - 1,
                'length': len(running_best) - current_start,
                'value': current_value
            })
        
        return stagnation_periods
    
    def _assess_convergence(self, running_best: List[float], stagnation_periods: List[Dict]) -> str:
        """Assess convergence quality"""
        
        if not running_best:
            return 'insufficient_data'
        
        plateau_length = self._calculate_plateau_length(running_best)
        total_length = len(running_best)
        
        # Check for premature convergence
        if plateau_length > total_length * 0.7:
            return 'premature_convergence'
        
        # Check for good convergence
        if plateau_length > 20 and plateau_length < total_length * 0.3:
            return 'good_convergence'
        
        # Check for ongoing improvement
        if plateau_length < 10:
            return 'still_improving'
        
        return 'moderate_convergence'
    
    def _analyze_parameter_stability(self, best_parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stability of best parameters across runs"""
        
        if len(best_parameters) < 2:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Extract parameter values
        all_params = {}
        for result in best_parameters:
            for param, value in result['parameters'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        # Calculate stability metrics
        stability_metrics = {}
        
        for param, values in all_params.items():
            if isinstance(values[0], (int, float)):
                # Numeric parameter
                stability_metrics[param] = {
                    'type': 'numeric',
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                    'range': max(values) - min(values),
                    'stability_score': 1 / (1 + np.std(values) / abs(np.mean(values))) if np.mean(values) != 0 else 0
                }
            else:
                # Categorical parameter
                unique_values = list(set(values))
                value_counts = {val: values.count(val) for val in unique_values}
                most_common = max(value_counts, key=value_counts.get)
                
                stability_metrics[param] = {
                    'type': 'categorical',
                    'unique_values': len(unique_values),
                    'most_common': most_common,
                    'most_common_frequency': value_counts[most_common] / len(values),
                    'stability_score': value_counts[most_common] / len(values)
                }
        
        # Overall stability score
        stability_scores = [metrics['stability_score'] for metrics in stability_metrics.values()]
        overall_stability = np.mean(stability_scores) if stability_scores else 0
        
        return {
            'parameter_stability': stability_metrics,
            'overall_stability_score': overall_stability,
            'stability_assessment': 'high' if overall_stability > 0.8 else 'medium' if overall_stability > 0.5 else 'low'
        }