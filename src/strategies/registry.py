from typing import Dict, Type, Any
import structlog

from .base import BaseStrategy
from .config import ConfigLoader, ValidationError
from .ma_crossover import MovingAverageCrossover
from .ma_crossover_simple import SimpleMovingAverageCrossover
from .rsi_strategy import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .breakout_strategy import BreakoutStrategy
from .macd_strategy import MACDStrategy
from .momentum_strategy import MomentumStrategy
from .stochastic_strategy import StochasticStrategy
from .mean_reversion_rsi import MeanReversionRSI
from .ema_scalping_strategy import EMAScalpingStrategy
from .vlam_consolidation_strategy import VLAMConsolidationStrategy
from .momentum_trading_strategy import MomentumTradingStrategy

logger = structlog.get_logger()

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'ma_crossover': MovingAverageCrossover,
    'ma_crossover_simple': SimpleMovingAverageCrossover,
    'rsi_strategy': RSIStrategy,
    'bollinger_bands': BollingerBandsStrategy,
    'breakout_strategy': BreakoutStrategy,
    'macd_strategy': MACDStrategy,
    'momentum_strategy': MomentumStrategy,
    'stochastic_strategy': StochasticStrategy,
    'mean_reversion_rsi': MeanReversionRSI,
    'ema_scalping_strategy': EMAScalpingStrategy,
    'vlam_consolidation_strategy': VLAMConsolidationStrategy,
    'momentum_trading_strategy': MomentumTradingStrategy,
}

def get_strategy_class(strategy_name: str) -> Type[BaseStrategy]:
    """Get strategy class by name"""
    if strategy_name not in STRATEGY_REGISTRY:
        available_strategies = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available strategies: {available_strategies}")
    
    return STRATEGY_REGISTRY[strategy_name]

def create_strategy(strategy_name: str, config: Dict[str, Any]) -> BaseStrategy:
    """
    Create a strategy instance using new architecture.
    
    Args:
        strategy_name: Name of the strategy (must match schema)
        config: Raw configuration from YAML
        
    Returns:
        Configured strategy instance
        
    Raises:
        ValidationError: If configuration is invalid
        ValueError: If strategy is unknown
    """
    strategy_class = get_strategy_class(strategy_name)
    
    try:
        # Strategy will validate config internally using new architecture
        strategy = strategy_class(config)
        logger.info("Strategy created with new architecture", 
                   strategy=strategy_name, 
                   parameter_count=len(strategy.params))
        return strategy
    except ValidationError as e:
        logger.error("Strategy configuration validation failed", 
                    strategy=strategy_name, error=str(e))
        raise
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

def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """
    Get parameter information for a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary with parameter definitions and metadata
    """
    return ConfigLoader.get_parameter_info(strategy_name)

def validate_strategy_config(strategy_name: str, config: Dict[str, Any]) -> bool:
    """
    Validate strategy configuration without creating the strategy.
    
    Args:
        strategy_name: Name of the strategy
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        ConfigLoader.load_strategy_params(strategy_name, config)
        return True
    except ValidationError:
        return False