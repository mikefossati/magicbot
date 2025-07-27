"""
Out-of-Sample Validation

Simple holdout validation that reserves a portion of data for final testing.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import structlog

from ..objectives import OptimizationObjective, OptimizationResult

logger = structlog.get_logger()

class OutOfSampleValidator:
    """
    Out-of-sample validator using simple holdout method.
    
    Reserves a portion of data for final validation to assess
    true out-of-sample performance.
    """
    
    def __init__(self, test_ratio: float = 0.2, gap_days: int = 0):
        """
        Initialize out-of-sample validator.
        
        Args:
            test_ratio: Ratio of data to reserve for testing
            gap_days: Gap between train and test data
        """
        self.test_ratio = test_ratio
        self.gap_days = gap_days
    
    def split_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data into train and test sets"""
        
        total_days = (end_date - start_date).days
        test_days = int(total_days * self.test_ratio)
        
        # Calculate split point
        test_start = end_date - timedelta(days=test_days)
        train_end = test_start - timedelta(days=self.gap_days)
        
        # Extract train data
        train_data = {}
        for symbol, df in historical_data.items():
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            mask = (timestamps >= start_date) & (timestamps <= train_end)
            train_df = df[mask].copy()
            if not train_df.empty:
                train_data[symbol] = train_df
        
        # Extract test data
        test_data = {}
        for symbol, df in historical_data.items():
            timestamps = pd.to_datetime(df['timestamp'], unit='ms')
            mask = (timestamps >= test_start) & (timestamps <= end_date)
            test_df = df[mask].copy()
            if not test_df.empty:
                test_data[symbol] = test_df
        
        logger.info("Split data for out-of-sample validation",
                   train_days=(train_end - start_date).days,
                   test_days=test_days,
                   gap_days=self.gap_days)
        
        return train_data, test_data
    
    async def validate_parameters(
        self,
        parameters: Dict[str, Any],
        strategy_factory: Callable,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """Validate parameters using holdout method"""
        
        # Split data
        train_data, test_data = self.split_data(historical_data, start_date, end_date)
        
        if not train_data or not test_data:
            raise ValueError("Insufficient data for out-of-sample validation")
        
        # Calculate date ranges
        total_days = (end_date - start_date).days
        test_days = int(total_days * self.test_ratio)
        test_start = end_date - timedelta(days=test_days)
        train_end = test_start - timedelta(days=self.gap_days)
        
        # Evaluate on test set
        result = await objective.evaluate(
            parameters=parameters,
            strategy_factory=strategy_factory,
            historical_data=test_data,
            start_date=test_start,
            end_date=end_date
        )
        
        return {
            'validation_method': 'Out-of-Sample Holdout',
            'test_ratio': self.test_ratio,
            'gap_days': self.gap_days,
            'train_period': (start_date, train_end),
            'test_period': (test_start, end_date),
            'out_sample_result': result,
            'is_valid': result.is_valid,
            'objective_value': result.objective_value,
            'metrics': result.metrics
        }