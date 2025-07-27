"""
Optimization Engine Integration Example

Demonstrates how to use the parameter optimization engine with existing
trading strategies and backtesting infrastructure.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import structlog

from .parameter_space import CommonParameterSpaces
from .objectives import CommonObjectives
from .grid_search_optimizer import GridSearchOptimizer, AdaptiveGridSearchOptimizer
from .validation import WalkForwardValidator
from ..strategies.momentum_trading_strategy import MomentumTradingStrategy

logger = structlog.get_logger()

async def optimize_momentum_strategy_example():
    """
    Example: Optimize momentum trading strategy parameters using grid search
    with walk-forward validation.
    """
    
    logger.info("Starting momentum strategy optimization example")
    
    # 1. Define parameter space for momentum strategy
    parameter_space = CommonParameterSpaces.momentum_strategy()
    
    # 2. Choose optimization objective
    objective = CommonObjectives.risk_adjusted_return()
    
    # 3. Create grid search optimizer
    optimizer = GridSearchOptimizer(
        parameter_space=parameter_space,
        objective=objective,
        grid_size=5,  # Small grid for example
        enable_pruning=True
    )
    
    # 4. Strategy factory function
    def create_momentum_strategy(parameters: Dict[str, Any]) -> MomentumTradingStrategy:
        """Create momentum strategy with given parameters"""
        config = {
            'symbols': ['BTCUSDT'],
            **parameters
        }
        return MomentumTradingStrategy(config)
    
    # 5. Generate example historical data (in practice, load real data)
    historical_data = generate_example_data()
    
    # 6. Define optimization period
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now() - timedelta(days=1)
    
    # 7. Run optimization
    try:
        best_result = await optimizer.optimize(
            strategy_factory=create_momentum_strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Optimization completed",
                   best_objective=best_result.objective_value,
                   best_parameters=best_result.parameters)
        
        # 8. Validate with walk-forward analysis
        validator = WalkForwardValidator(num_windows=3)
        
        validation_results = await validator.validate_parameters(
            optimizer=optimizer,
            strategy_factory=create_momentum_strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date,
            objective=objective
        )
        
        logger.info("Walk-forward validation completed",
                   robustness_score=validation_results.get('robustness_score', 0))
        
        return best_result, validation_results
        
    except Exception as e:
        logger.error("Optimization failed", error=str(e))
        raise

async def adaptive_optimization_example():
    """
    Example: Use adaptive grid search for more efficient parameter optimization.
    """
    
    logger.info("Starting adaptive optimization example")
    
    # 1. Create adaptive grid search optimizer
    parameter_space = CommonParameterSpaces.momentum_strategy()
    objective = CommonObjectives.maximize_sharpe()
    
    optimizer = AdaptiveGridSearchOptimizer(
        parameter_space=parameter_space,
        objective=objective,
        initial_grid_size=3,
        max_refinement_levels=2
    )
    
    # 2. Strategy factory
    def create_strategy(parameters: Dict[str, Any]) -> MomentumTradingStrategy:
        config = {'symbols': ['BTCUSDT'], **parameters}
        return MomentumTradingStrategy(config)
    
    # 3. Run adaptive optimization
    historical_data = generate_example_data()
    start_date = datetime.now() - timedelta(days=60)
    end_date = datetime.now() - timedelta(days=1)
    
    try:
        best_result = await optimizer.optimize(
            strategy_factory=create_strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get optimization analysis
        grid_info = optimizer.get_grid_info()
        param_analysis = optimizer.get_parameter_analysis()
        
        logger.info("Adaptive optimization completed",
                   best_objective=best_result.objective_value,
                   evaluations=grid_info['evaluations_completed'],
                   param_correlations=param_analysis.get('grid_statistics', {}))
        
        return best_result
        
    except Exception as e:
        logger.error("Adaptive optimization failed", error=str(e))
        raise

def generate_example_data() -> Dict[str, pd.DataFrame]:
    """
    Generate example market data for demonstration.
    In practice, this would load real historical data.
    """
    
    import numpy as np
    
    # Generate 90 days of hourly data
    periods = 90 * 24
    timestamps = []
    prices = []
    
    base_price = 50000
    current_price = base_price
    
    for i in range(periods):
        timestamp = datetime.now() - timedelta(hours=periods-i)
        
        # Simple trending price movement with noise
        trend = 0.0001  # Small upward trend
        noise = np.random.normal(0, 0.01)
        price_change = trend + noise
        current_price *= (1 + price_change)
        
        timestamps.append(int(timestamp.timestamp() * 1000))
        prices.append(current_price)
    
    # Create OHLCV data
    data_points = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        # Simple OHLC based on close price
        open_price = prices[i-1] if i > 0 else price
        high_price = price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = price * (1 - abs(np.random.normal(0, 0.005)))
        close_price = price
        volume = 1000 + np.random.uniform(0, 2000)
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        data_points.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data_points)
    return {'BTCUSDT': df}

async def comprehensive_optimization_workflow():
    """
    Complete optimization workflow demonstrating all features.
    """
    
    logger.info("Starting comprehensive optimization workflow")
    
    try:
        # 1. Basic grid search optimization
        logger.info("Phase 1: Grid search optimization")
        basic_result, validation = await optimize_momentum_strategy_example()
        
        # 2. Adaptive optimization for refinement
        logger.info("Phase 2: Adaptive optimization")
        refined_result = await adaptive_optimization_example()
        
        # 3. Compare results
        logger.info("Optimization workflow completed",
                   basic_objective=basic_result.objective_value,
                   refined_objective=refined_result.objective_value,
                   improvement=refined_result.objective_value - basic_result.objective_value)
        
        return {
            'basic_optimization': basic_result,
            'validation_results': validation,
            'refined_optimization': refined_result
        }
        
    except Exception as e:
        logger.error("Comprehensive optimization workflow failed", error=str(e))
        raise

# CLI interface for testing
if __name__ == "__main__":
    import structlog
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run comprehensive workflow
    results = asyncio.run(comprehensive_optimization_workflow())
    print("Optimization completed successfully!")
    print(f"Final results: {results}")