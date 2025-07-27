"""
Configuration loader for trading strategies.

This module loads and processes strategy configurations from YAML files,
applying defaults and ensuring parameter completeness.
"""

from typing import Dict, Any
import structlog

from .schema import StrategyParameterSchema
from .validator import ConfigValidator, ValidationError

logger = structlog.get_logger()

class ConfigLoader:
    """Loads and processes strategy configurations with validation"""
    
    @classmethod
    def load_strategy_params(cls, strategy_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and validate strategy parameters from configuration.
        
        Args:
            strategy_name: Name of the strategy
            config: Raw configuration dictionary from YAML
            
        Returns:
            Processed configuration with defaults applied
            
        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate configuration first
        ConfigValidator.validate_and_raise(strategy_name, config)
        
        # Get schema for the strategy
        schema = StrategyParameterSchema.get_schema(strategy_name)
        
        # Apply defaults for missing optional parameters
        processed_config = config.copy()
        
        for param_name, param_def in schema['parameters'].items():
            if param_name not in processed_config and param_def.default is not None:
                processed_config[param_name] = param_def.default
                logger.debug("Applied default parameter", 
                           strategy=strategy_name,
                           parameter=param_name, 
                           default_value=param_def.default)
        
        logger.info("Strategy configuration loaded successfully", 
                   strategy=strategy_name,
                   parameter_count=len(processed_config))
        
        return processed_config
    
    @classmethod
    def get_data_requirements(cls, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get data requirements for strategy based on its parameters.
        
        Args:
            strategy_name: Name of the strategy
            params: Strategy parameters
            
        Returns:
            Data requirements dictionary
        """
        return StrategyParameterSchema.get_required_data_structure(strategy_name, params)
    
    @classmethod
    def get_parameter_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get parameter information for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with parameter definitions and metadata
        """
        schema = StrategyParameterSchema.get_schema(strategy_name)
        
        param_info = {}
        for param_name, param_def in schema['parameters'].items():
            param_info[param_name] = {
                'type': param_def.param_type.value,
                'required': param_name in schema['required_params'],
                'default': param_def.default,
                'min_value': param_def.min_value,
                'max_value': param_def.max_value,
                'allowed_values': param_def.allowed_values,
                'description': param_def.description
            }
        
        return {
            'strategy_name': strategy_name,
            'parameters': param_info,
            'required_params': schema['required_params'],
            'optional_params': schema['optional_params']
        }