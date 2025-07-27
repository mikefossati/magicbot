"""
Strategy parameter schema definitions and validation.

This module defines the centralized parameter schema for all trading strategies,
ensuring YAML configuration is the single source of truth.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union, List
import structlog

logger = structlog.get_logger()

class ParameterType(Enum):
    """Parameter data types for validation"""
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    LIST = "list"
    DICT = "dict"

@dataclass
class ParameterDefinition:
    """Definition of a strategy parameter with validation rules"""
    name: str
    param_type: ParameterType
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value against this definition"""
        if value is None:
            return not self.required
            
        # Type validation
        if self.param_type == ParameterType.INTEGER and not isinstance(value, int):
            return False
        elif self.param_type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            return False
        elif self.param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.param_type == ParameterType.STRING and not isinstance(value, str):
            return False
        elif self.param_type == ParameterType.LIST and not isinstance(value, list):
            return False
        elif self.param_type == ParameterType.DICT and not isinstance(value, dict):
            return False
            
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
            
        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            return False
            
        return True

class StrategyParameterSchema:
    """Centralized parameter schema for all trading strategies"""
    
    # Global parameter definitions used across multiple strategies
    GLOBAL_PARAMETERS = {
        'symbols': ParameterDefinition(
            name='symbols',
            param_type=ParameterType.LIST,
            required=True,
            description='List of trading symbols'
        ),
        'position_size': ParameterDefinition(
            name='position_size',
            param_type=ParameterType.FLOAT,
            required=True,
            min_value=0.001,
            max_value=1.0,
            description='Position size as fraction of portfolio'
        ),
        'timeframes': ParameterDefinition(
            name='timeframes',
            param_type=ParameterType.LIST,
            default=['1h'],
            description='Required timeframes for data'
        ),
        'lookback_periods': ParameterDefinition(
            name='lookback_periods',
            param_type=ParameterType.INTEGER,
            default=100,
            min_value=10,
            description='Number of historical periods required'
        ),
        
        # RSI parameters
        'rsi_period': ParameterDefinition(
            name='rsi_period',
            param_type=ParameterType.INTEGER,
            default=14,
            min_value=2,
            max_value=100,
            description='RSI calculation period'
        ),
        'rsi_oversold': ParameterDefinition(
            name='rsi_oversold',
            param_type=ParameterType.FLOAT,
            default=30,
            min_value=5,
            max_value=50,
            description='RSI oversold threshold'
        ),
        'rsi_overbought': ParameterDefinition(
            name='rsi_overbought',
            param_type=ParameterType.FLOAT,
            default=70,
            min_value=50,
            max_value=95,
            description='RSI overbought threshold'
        ),
        
        # Volume parameters
        'volume_confirmation': ParameterDefinition(
            name='volume_confirmation',
            param_type=ParameterType.BOOLEAN,
            default=True,
            description='Require volume confirmation for signals'
        ),
        'volume_period': ParameterDefinition(
            name='volume_period',
            param_type=ParameterType.INTEGER,
            default=20,
            min_value=5,
            max_value=100,
            description='Volume moving average period'
        ),
        'volume_multiplier': ParameterDefinition(
            name='volume_multiplier',
            param_type=ParameterType.FLOAT,
            default=1.5,
            min_value=1.0,
            max_value=5.0,
            description='Volume threshold multiplier'
        ),
        
        # Moving average parameters
        'fast_period': ParameterDefinition(
            name='fast_period',
            param_type=ParameterType.INTEGER,
            default=10,
            min_value=2,
            max_value=50,
            description='Fast moving average period'
        ),
        'slow_period': ParameterDefinition(
            name='slow_period',
            param_type=ParameterType.INTEGER,
            default=30,
            min_value=5,
            max_value=200,
            description='Slow moving average period'
        ),
        
        # MACD parameters
        'macd_fast': ParameterDefinition(
            name='macd_fast',
            param_type=ParameterType.INTEGER,
            default=12,
            min_value=2,
            max_value=50,
            description='MACD fast EMA period'
        ),
        'macd_slow': ParameterDefinition(
            name='macd_slow',
            param_type=ParameterType.INTEGER,
            default=26,
            min_value=5,
            max_value=100,
            description='MACD slow EMA period'
        ),
        'macd_signal': ParameterDefinition(
            name='macd_signal',
            param_type=ParameterType.INTEGER,
            default=9,
            min_value=2,
            max_value=30,
            description='MACD signal line period'
        ),
        
        # Risk management parameters
        'stop_loss_pct': ParameterDefinition(
            name='stop_loss_pct',
            param_type=ParameterType.FLOAT,
            default=2.0,
            min_value=0.1,
            max_value=10.0,
            description='Stop loss percentage'
        ),
        'take_profit_pct': ParameterDefinition(
            name='take_profit_pct',
            param_type=ParameterType.FLOAT,
            default=4.0,
            min_value=0.5,
            max_value=20.0,
            description='Take profit percentage'
        ),
        'max_daily_trades': ParameterDefinition(
            name='max_daily_trades',
            param_type=ParameterType.INTEGER,
            default=3,
            min_value=1,
            max_value=50,
            description='Maximum trades per day'
        ),
    }
    
    # Strategy-specific parameter schemas
    STRATEGY_SCHEMAS = {
        'ma_crossover': {
            'required_params': ['symbols', 'position_size', 'fast_period', 'slow_period'],
            'optional_params': ['timeframes', 'lookback_periods', 'stop_loss_pct', 'take_profit_pct', 'volume_confirmation', 'volume_period', 'volume_multiplier'],
            'custom_params': {
                'momentum_confirmation': ParameterDefinition(
                    name='momentum_confirmation',
                    param_type=ParameterType.BOOLEAN,
                    default=True,
                    description='Require momentum confirmation for signals'
                ),
                'momentum_period': ParameterDefinition(
                    name='momentum_period',
                    param_type=ParameterType.INTEGER,
                    default=14,
                    min_value=5,
                    max_value=50,
                    description='Period for momentum calculation'
                ),
                'min_momentum_threshold': ParameterDefinition(
                    name='min_momentum_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.5,
                    min_value=0.1,
                    max_value=5.0,
                    description='Minimum momentum threshold for signal confirmation'
                ),
                'atr_period': ParameterDefinition(
                    name='atr_period',
                    param_type=ParameterType.INTEGER,
                    default=14,
                    min_value=5,
                    max_value=50,
                    description='ATR period for volatility-based position sizing'
                ),
                'volatility_multiplier': ParameterDefinition(
                    name='volatility_multiplier',
                    param_type=ParameterType.FLOAT,
                    default=2.0,
                    min_value=0.5,
                    max_value=5.0,
                    description='Multiplier for volatility-based position sizing'
                ),
                'trend_strength_threshold': ParameterDefinition(
                    name='trend_strength_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.001,
                    min_value=0.0001,
                    max_value=0.01,
                    description='Minimum trend strength for signal generation'
                ),
                'trailing_stop_multiplier': ParameterDefinition(
                    name='trailing_stop_multiplier',
                    param_type=ParameterType.FLOAT,
                    default=2.0,
                    min_value=1.0,
                    max_value=5.0,
                    description='ATR multiplier for trailing stop calculation'
                ),
                'profit_target_multiplier': ParameterDefinition(
                    name='profit_target_multiplier',
                    param_type=ParameterType.FLOAT,
                    default=3.0,
                    min_value=1.5,
                    max_value=10.0,
                    description='ATR multiplier for profit target calculation'
                ),
                'ma_separation_threshold': ParameterDefinition(
                    name='ma_separation_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.002,
                    min_value=0.0005,
                    max_value=0.01,
                    description='Minimum separation between MAs for strong trend'
                )
            }
        },
        
        'rsi_strategy': {
            'required_params': ['symbols', 'position_size', 'rsi_period'],
            'optional_params': ['rsi_oversold', 'rsi_overbought', 'timeframes', 'lookback_periods'],
            'custom_params': {}
        },
        
        'bollinger_bands': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['timeframes', 'lookback_periods'],
            'custom_params': {
                'period': ParameterDefinition(
                    name='period',
                    param_type=ParameterType.INTEGER,
                    default=20,
                    min_value=5,
                    max_value=100,
                    description='Bollinger Bands period'
                ),
                'std_dev': ParameterDefinition(
                    name='std_dev',
                    param_type=ParameterType.FLOAT,
                    default=2.0,
                    min_value=0.5,
                    max_value=5.0,
                    description='Standard deviation multiplier'
                ),
                'mean_reversion_threshold': ParameterDefinition(
                    name='mean_reversion_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.02,
                    min_value=0.005,
                    max_value=0.1,
                    description='Mean reversion threshold'
                )
            }
        },
        
        'breakout_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['volume_confirmation', 'timeframes', 'lookback_periods'],
            'custom_params': {
                'lookback_period': ParameterDefinition(
                    name='lookback_period',
                    param_type=ParameterType.INTEGER,
                    default=20,
                    min_value=5,
                    max_value=100,
                    description='Lookback period for breakout detection'
                ),
                'breakout_threshold': ParameterDefinition(
                    name='breakout_threshold',
                    param_type=ParameterType.FLOAT,
                    default=1.02,
                    min_value=1.001,
                    max_value=1.1,
                    description='Breakout threshold multiplier'
                ),
                'min_volatility': ParameterDefinition(
                    name='min_volatility',
                    param_type=ParameterType.FLOAT,
                    default=0.005,
                    min_value=0.001,
                    max_value=0.05,
                    description='Minimum volatility requirement'
                )
            }
        },
        
        'macd_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['macd_fast', 'macd_slow', 'macd_signal', 'timeframes', 'lookback_periods'],
            'custom_params': {
                'histogram_threshold': ParameterDefinition(
                    name='histogram_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.0,
                    min_value=-1.0,
                    max_value=1.0,
                    description='MACD histogram threshold'
                )
            }
        },
        
        
        'momentum_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['rsi_period', 'fast_period', 'slow_period', 'volume_period', 'volume_multiplier', 'timeframes', 'lookback_periods'],
            'custom_params': {}
        },
        
        'stochastic_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['timeframes', 'lookback_periods'],
            'custom_params': {
                'k_period': ParameterDefinition(
                    name='k_period',
                    param_type=ParameterType.INTEGER,
                    default=14,
                    min_value=5,
                    max_value=50,
                    description='Stochastic %K period'
                ),
                'd_period': ParameterDefinition(
                    name='d_period',
                    param_type=ParameterType.INTEGER,
                    default=3,
                    min_value=1,
                    max_value=10,
                    description='Stochastic %D period'
                ),
                'overbought': ParameterDefinition(
                    name='overbought',
                    param_type=ParameterType.FLOAT,
                    default=80,
                    min_value=70,
                    max_value=90,
                    description='Stochastic overbought level'
                ),
                'oversold': ParameterDefinition(
                    name='oversold',
                    param_type=ParameterType.FLOAT,
                    default=20,
                    min_value=10,
                    max_value=30,
                    description='Stochastic oversold level'
                )
            }
        },
        
        'mean_reversion_rsi': {
            'required_params': ['symbols', 'position_size', 'rsi_period'],
            'optional_params': ['rsi_oversold', 'rsi_overbought', 'fast_period', 'volume_confirmation', 'timeframes', 'lookback_periods'],
            'custom_params': {}
        },
        
        'ema_scalping_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['rsi_period', 'volume_period', 'volume_multiplier', 'timeframes', 'lookback_periods'],
            'custom_params': {
                'scalping_ema_fast': ParameterDefinition(
                    name='scalping_ema_fast',
                    param_type=ParameterType.INTEGER,
                    default=8,
                    min_value=3,
                    max_value=20,
                    description='Fast EMA for scalping'
                ),
                'scalping_ema_slow': ParameterDefinition(
                    name='scalping_ema_slow',
                    param_type=ParameterType.INTEGER,
                    default=21,
                    min_value=10,
                    max_value=50,
                    description='Slow EMA for scalping'
                )
            }
        },
        
        'vlam_consolidation_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['timeframes', 'lookback_periods'],
            'custom_params': {
                'vlam_period': ParameterDefinition(
                    name='vlam_period',
                    param_type=ParameterType.INTEGER,
                    default=10,
                    min_value=5,
                    max_value=30,
                    description='VLAM calculation period'
                ),
                'consolidation_min_length': ParameterDefinition(
                    name='consolidation_min_length',
                    param_type=ParameterType.INTEGER,
                    default=4,
                    min_value=3,
                    max_value=20,
                    description='Minimum consolidation length'
                )
            }
        },
        
        'momentum_trading_strategy': {
            'required_params': ['symbols', 'position_size'],
            'optional_params': ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 'volume_period', 'volume_multiplier', 'timeframes', 'lookback_periods'],
            'custom_params': {
                'trend_ema_fast': ParameterDefinition(
                    name='trend_ema_fast',
                    param_type=ParameterType.INTEGER,
                    default=5,
                    min_value=3,
                    max_value=20,
                    description='Fast EMA for trend detection'
                ),
                'trend_ema_slow': ParameterDefinition(
                    name='trend_ema_slow',
                    param_type=ParameterType.INTEGER,
                    default=10,
                    min_value=5,
                    max_value=30,
                    description='Slow EMA for trend detection'
                ),
                'momentum_threshold': ParameterDefinition(
                    name='momentum_threshold',
                    param_type=ParameterType.FLOAT,
                    default=0.001,
                    min_value=0.0005,
                    max_value=0.01,
                    description='Minimum momentum threshold'
                )
            }
        }
    }
    
    @classmethod
    def get_schema(cls, strategy_name: str) -> Dict[str, Any]:
        """Get the complete parameter schema for a strategy"""
        if strategy_name not in cls.STRATEGY_SCHEMAS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        schema = cls.STRATEGY_SCHEMAS[strategy_name]
        
        # Combine global and custom parameters
        all_params = {}
        
        # Add required global parameters
        for param_name in schema['required_params']:
            if param_name in cls.GLOBAL_PARAMETERS:
                all_params[param_name] = cls.GLOBAL_PARAMETERS[param_name]
                
        # Add optional global parameters
        for param_name in schema['optional_params']:
            if param_name in cls.GLOBAL_PARAMETERS:
                all_params[param_name] = cls.GLOBAL_PARAMETERS[param_name]
                
        # Add custom parameters
        all_params.update(schema['custom_params'])
        
        return {
            'required_params': schema['required_params'],
            'optional_params': schema['optional_params'],
            'parameters': all_params
        }
    
    @classmethod
    def get_required_data_structure(cls, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get required data structure for strategy based on parameters"""
        schema = cls.get_schema(strategy_name)
        
        # Calculate lookback periods based on strategy parameters
        max_period = 100  # default
        
        if 'slow_period' in params:
            max_period = max(max_period, params['slow_period'] + 10)
        if 'rsi_period' in params:
            max_period = max(max_period, params['rsi_period'] + 10)
        if 'period' in params:  # Bollinger Bands
            max_period = max(max_period, params['period'] + 10)
        if 'volume_period' in params:
            max_period = max(max_period, params['volume_period'] + 10)
            
        return {
            'timeframes': params.get('timeframes', ['1h']),
            'lookback_periods': max_period,
            'indicators': cls._get_required_indicators(strategy_name)
        }
    
    @classmethod
    def _get_required_indicators(cls, strategy_name: str) -> List[str]:
        """Get list of required indicators for strategy"""
        indicator_map = {
            'ma_crossover': ['sma'],
            'rsi_strategy': ['rsi'],
            'bollinger_bands': ['bb', 'sma'],
            'breakout_strategy': ['volume', 'volatility'],
            'macd_strategy': ['macd'],
            'momentum_strategy': ['momentum', 'sma', 'volume'],
            'stochastic_strategy': ['stochastic'],
            'ema_scalping_strategy': ['ema', 'rsi', 'volume'],
            'mean_reversion_rsi': ['rsi', 'sma', 'volume'],
            'vlam_consolidation_strategy': ['volume', 'atr'],
            'momentum_trading_strategy': ['ema', 'rsi', 'macd', 'volume']
        }
        
        return indicator_map.get(strategy_name, [])