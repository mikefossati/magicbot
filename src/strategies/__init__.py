from .base import BaseStrategy
from .signal import Signal
from .ma_crossover import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .breakout_strategy import BreakoutStrategy
from .macd_strategy import MACDStrategy
from .momentum_strategy import MomentumStrategy
from .stochastic_strategy import StochasticStrategy
from .mean_reversion_rsi import MeanReversionRSI
from .ema_scalping_strategy import EMAScalpingStrategy
from .registry import (
    get_strategy_class,
    create_strategy,
    get_available_strategies,
    register_strategy,
    STRATEGY_REGISTRY
)

# Strategy factory function for optimization
def get_strategy_factory(strategy_name: str):
    """
    Get strategy factory function by name for optimization.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class
    """
    return get_strategy_class(strategy_name)

__all__ = [
    'BaseStrategy',
    'Signal',
    'MovingAverageCrossover',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'BreakoutStrategy',
    'MACDStrategy',
    'MomentumStrategy',
    'StochasticStrategy',
    'MeanReversionRSI',
    'EMAScalpingStrategy',
    'get_strategy_class',
    'create_strategy',
    'get_available_strategies',
    'register_strategy',
    'get_strategy_factory',
    'STRATEGY_REGISTRY'
]