"""
Configuration validation for trading strategies.

This module validates strategy parameters against the defined schema
and ensures YAML configuration integrity.
"""

from typing import Dict, Any, List, Tuple
import structlog

from .schema import StrategyParameterSchema, ParameterDefinition

logger = structlog.get_logger()

class ValidationError(Exception):
    """Raised when strategy configuration validation fails"""
    pass

class ConfigValidator:
    """Validates strategy configurations against schema definitions"""
    
    @classmethod
    def validate_strategy_config(cls, strategy_name: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate strategy configuration against schema.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary from YAML
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            schema = StrategyParameterSchema.get_schema(strategy_name)
        except ValueError as e:
            return False, [str(e)]
        
        # Check required parameters
        for param_name in schema['required_params']:
            if param_name not in config:
                errors.append(f"Missing required parameter: {param_name}")
                continue
                
            param_def = schema['parameters'][param_name]
            if not param_def.validate(config[param_name]):
                errors.append(f"Invalid value for parameter '{param_name}': {config[param_name]}")
        
        # Validate optional parameters if present
        for param_name in schema['optional_params']:
            if param_name in config:
                param_def = schema['parameters'][param_name]
                if not param_def.validate(config[param_name]):
                    errors.append(f"Invalid value for optional parameter '{param_name}': {config[param_name]}")
        
        # Validate custom parameters if present
        for param_name, param_def in schema['parameters'].items():
            if param_name in config and param_name not in schema['required_params'] and param_name not in schema['optional_params']:
                if not param_def.validate(config[param_name]):
                    errors.append(f"Invalid value for parameter '{param_name}': {config[param_name]}")
        
        # Strategy-specific validation rules
        strategy_errors = cls._validate_strategy_specific_rules(strategy_name, config, schema)
        errors.extend(strategy_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_strategy_specific_rules(cls, strategy_name: str, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Apply strategy-specific validation rules"""
        errors = []
        
        if strategy_name == 'ma_crossover':
            if 'fast_period' in config and 'slow_period' in config:
                if config['fast_period'] >= config['slow_period']:
                    errors.append("fast_period must be less than slow_period")
        
        elif strategy_name == 'rsi_strategy':
            if 'rsi_oversold' in config and 'rsi_overbought' in config:
                if config['rsi_oversold'] >= config['rsi_overbought']:
                    errors.append("rsi_oversold must be less than rsi_overbought")
        
        elif strategy_name == 'macd_strategy':
            if 'macd_fast' in config and 'macd_slow' in config:
                if config['macd_fast'] >= config['macd_slow']:
                    errors.append("macd_fast must be less than macd_slow")
        
        elif strategy_name == 'day_trading_strategy':
            # EMA periods validation
            if all(param in config for param in ['fast_ema', 'medium_ema', 'slow_ema']):
                if not (config['fast_ema'] < config['medium_ema'] < config['slow_ema']):
                    errors.append("EMA periods must be in ascending order: fast_ema < medium_ema < slow_ema")
            
            # RSI thresholds validation
            rsi_params = ['rsi_oversold', 'rsi_neutral_low', 'rsi_neutral_high', 'rsi_overbought']
            if all(param in config for param in rsi_params):
                values = [config[param] for param in rsi_params]
                if values != sorted(values):
                    errors.append("RSI thresholds must be in ascending order: oversold < neutral_low < neutral_high < overbought")
            
            # Risk management validation
            if 'stop_loss_pct' in config and 'take_profit_pct' in config:
                if config['stop_loss_pct'] >= config['take_profit_pct']:
                    errors.append("take_profit_pct should be greater than stop_loss_pct for positive risk/reward")
            
            # Leverage validation
            if config.get('use_leverage', False):
                if 'leverage' in config and 'max_leverage' in config:
                    if config['leverage'] > config['max_leverage']:
                        errors.append("leverage cannot exceed max_leverage")
                
                if 'leverage_risk_factor' in config:
                    if not (0 < config['leverage_risk_factor'] <= 1):
                        errors.append("leverage_risk_factor must be between 0 and 1")
            
            # Signal scoring validation
            if 'min_signal_score' in config and 'strong_signal_score' in config:
                if config['min_signal_score'] >= config['strong_signal_score']:
                    errors.append("strong_signal_score must be greater than min_signal_score")
        
        return errors
    
    @classmethod
    def validate_full_config(cls, config: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate complete configuration file including all strategies.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (is_valid, dict_of_errors_by_strategy)
        """
        all_errors = {}
        
        if 'strategies' not in config:
            return False, {'global': ['Missing strategies section in configuration']}
        
        strategies_config = config['strategies']
        
        # Get list of enabled strategies
        enabled_strategies = strategies_config.get('enabled', [])
        
        # Validate each enabled strategy
        for strategy_name in enabled_strategies:
            if strategy_name not in strategies_config:
                all_errors[strategy_name] = [f"Strategy '{strategy_name}' is enabled but has no configuration"]
                continue
            
            strategy_config = strategies_config[strategy_name]
            is_valid, errors = cls.validate_strategy_config(strategy_name, strategy_config)
            
            if not is_valid:
                all_errors[strategy_name] = errors
        
        return len(all_errors) == 0, all_errors
    
    @classmethod
    def validate_and_raise(cls, strategy_name: str, config: Dict[str, Any]) -> None:
        """
        Validate strategy configuration and raise ValidationError if invalid.
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        is_valid, errors = cls.validate_strategy_config(strategy_name, config)
        
        if not is_valid:
            error_msg = f"Strategy '{strategy_name}' configuration validation failed:\n"
            error_msg += "\n".join(f"  - {error}" for error in errors)
            logger.error("Strategy configuration validation failed", 
                        strategy=strategy_name, 
                        errors=errors)
            raise ValidationError(error_msg)