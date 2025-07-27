"""
Monte Carlo Validation

Uses Monte Carlo simulation to assess parameter robustness across
random data samples and market conditions.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import structlog
import asyncio

from ..objectives import OptimizationObjective, OptimizationResult

logger = structlog.get_logger()

class MonteCarloValidator:
    """
    Monte Carlo validator for robust parameter assessment.
    
    Uses random sampling and bootstrap methods to test parameter
    performance across different market conditions.
    """
    
    def __init__(
        self,
        n_simulations: int = 100,
        sample_ratio: float = 0.8,
        bootstrap: bool = True,
        noise_level: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo validator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            sample_ratio: Ratio of data to sample in each simulation
            bootstrap: Whether to use bootstrap sampling
            noise_level: Level of noise to add to data (0-1)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.sample_ratio = sample_ratio
        self.bootstrap = bootstrap
        self.noise_level = noise_level
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    async def validate_parameters(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """Validate parameters using Monte Carlo simulation"""
        
        logger.info("Starting Monte Carlo validation",
                   n_simulations=self.n_simulations,
                   sample_ratio=self.sample_ratio)
        
        simulation_results = []
        
        for i in range(self.n_simulations):
            try:
                # Generate sample data
                sample_data = self._generate_sample_data(historical_data)
                
                # Evaluate parameters on sample
                result = await objective.evaluate(
                    parameters=parameters,
                    strategy_factory=strategy_factory,
                    historical_data=sample_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                simulation_results.append({
                    'simulation_id': i,
                    'result': result,
                    'objective_value': result.objective_value,
                    'is_valid': result.is_valid
                })
                
                if (i + 1) % 20 == 0:
                    logger.info("Monte Carlo progress", completed=i + 1, total=self.n_simulations)
                
            except Exception as e:
                logger.warning("Monte Carlo simulation failed", simulation=i, error=str(e))
                continue
        
        # Analyze results
        analysis = self._analyze_monte_carlo_results(simulation_results)
        
        logger.info("Monte Carlo validation completed",
                   valid_simulations=len([r for r in simulation_results if r['is_valid']]),
                   mean_score=analysis.get('mean_objective', 0),
                   score_std=analysis.get('std_objective', 0))
        
        return analysis
    
    def _generate_sample_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate sample data for Monte Carlo simulation"""
        
        sample_data = {}
        
        for symbol, df in historical_data.items():
            if df.empty:
                continue
            
            n_samples = int(len(df) * self.sample_ratio)
            
            if self.bootstrap:
                # Bootstrap sampling with replacement
                indices = np.random.choice(len(df), size=n_samples, replace=True)
                sample_df = df.iloc[indices].copy()
                # Sort by timestamp to maintain time order
                sample_df = sample_df.sort_values('timestamp').reset_index(drop=True)
            else:
                # Random sampling without replacement
                indices = np.random.choice(len(df), size=n_samples, replace=False)
                indices = np.sort(indices)  # Maintain time order
                sample_df = df.iloc[indices].copy()
            
            # Add noise if specified
            if self.noise_level > 0:
                sample_df = self._add_noise(sample_df)
            
            sample_data[symbol] = sample_df
        
        return sample_data
    
    def _add_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add random noise to price data"""
        
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                # Add proportional noise
                noise = np.random.normal(0, self.noise_level, len(df))
                df[col] = df[col] * (1 + noise)
        
        # Ensure OHLC consistency
        if all(col in df.columns for col in price_cols):
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _analyze_monte_carlo_results(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        
        if not simulation_results:
            return {'error': 'No simulation results available'}
        
        # Extract valid results
        valid_results = [r for r in simulation_results if r['is_valid']]
        
        if not valid_results:
            return {'error': 'No valid simulation results'}
        
        objectives = [r['objective_value'] for r in valid_results]
        
        # Calculate statistics
        analysis = {
            'validation_method': 'Monte Carlo Simulation',
            'total_simulations': len(simulation_results),
            'valid_simulations': len(valid_results),
            'success_rate': len(valid_results) / len(simulation_results),
            
            # Objective statistics
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'min_objective': np.min(objectives),
            'max_objective': np.max(objectives),
            'median_objective': np.median(objectives),
            
            # Percentiles
            'percentile_5': np.percentile(objectives, 5),
            'percentile_25': np.percentile(objectives, 25),
            'percentile_75': np.percentile(objectives, 75),
            'percentile_95': np.percentile(objectives, 95),
            
            # Risk metrics
            'value_at_risk_5': np.percentile(objectives, 5),
            'expected_shortfall_5': np.mean([o for o in objectives if o <= np.percentile(objectives, 5)]),
            'coefficient_of_variation': np.std(objectives) / np.mean(objectives) if np.mean(objectives) != 0 else float('inf'),
            
            # Robustness metrics
            'robustness_score': self._calculate_robustness_score(objectives),
            'stability_index': 1.0 / (1.0 + np.std(objectives) / abs(np.mean(objectives))) if np.mean(objectives) != 0 else 0,
            
            # Distribution properties
            'skewness': self._calculate_skewness(objectives),
            'kurtosis': self._calculate_kurtosis(objectives),
            
            # Confidence intervals
            'confidence_interval_95': (np.percentile(objectives, 2.5), np.percentile(objectives, 97.5)),
            'confidence_interval_90': (np.percentile(objectives, 5), np.percentile(objectives, 95))
        }
        
        return analysis
    
    def _calculate_robustness_score(self, objectives: List[float]) -> float:
        """Calculate robustness score (0-1)"""
        
        if len(objectives) < 2:
            return 1.0
        
        # Score based on consistency and worst-case performance
        mean_obj = np.mean(objectives)
        worst_case = np.min(objectives)
        consistency = 1.0 / (1.0 + np.std(objectives) / abs(mean_obj)) if mean_obj != 0 else 0
        
        # Worst case relative to mean
        worst_case_ratio = worst_case / mean_obj if mean_obj > 0 else 0
        
        # Combined score
        robustness = (consistency + max(0, worst_case_ratio)) / 2
        return max(0, min(1, robustness))
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of distribution"""
        
        if len(data) < 3:
            return 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of distribution"""
        
        if len(data) < 4:
            return 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis