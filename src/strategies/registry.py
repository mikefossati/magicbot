from typing import Dict, Type
import structlog

from .base import BaseStrategy
from .ma_crossover import MovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .breakout_strategy import BreakoutStrategy
from .macd_strategy import MACDStrategy
from .momentum_strategy import MomentumStrategy
from .stochastic_strategy import StochasticStrategy
from .mean_reversion_rsi import MeanReversionRSI
from .ema_scalping_strategy import EMAScalpingStrategy
from .day_trading_strategy import DayTradingStrategy

logger = structlog.get_logger()

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'ma_crossover': MovingAverageCrossover,
    'rsi_strategy': RSIStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'breakout_strategy': BreakoutStrategy,
    'macd_strategy': MACDStrategy,
    'momentum_strategy': MomentumStrategy,
    'stochastic_strategy': StochasticStrategy,
    'mean_reversion_rsi': MeanReversionRSI,
    'ema_scalping_strategy': EMAScalpingStrategy,
    'day_trading_strategy': DayTradingStrategy,
}

def get_strategy_class(strategy_name: str) -> Type[BaseStrategy]:
    """Get strategy class by name"""
    if strategy_name not in STRATEGY_REGISTRY:
        available_strategies = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available_strategies}")
    
    return STRATEGY_REGISTRY[strategy_name]

def create_strategy(strategy_name: str, config: Dict) -> BaseStrategy:
    """Create a strategy instance"""
    strategy_class = get_strategy_class(strategy_name)
    
    try:
        strategy = strategy_class(config)
        logger.info("Strategy created", strategy=strategy_name, config=config)
        return strategy
    except Exception as e:
        logger.error("Failed to create strategy", strategy=strategy_name, error=str(e))
        raise

def get_available_strategies() -> Dict[str, Type[BaseStrategy]]:
    """Get all available strategies"""
    return STRATEGY_REGISTRY.copy()

def register_strategy(name: str, strategy_class: Type[BaseStrategy]) -> None:
    """Register a new strategy dynamically"""
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError(f"Strategy class must inherit from BaseStrategy")
    
    STRATEGY_REGISTRY[name] = strategy_class
    logger.info("Strategy registered", name=name, class_name=strategy_class.__name__)