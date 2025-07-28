from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
import structlog
import pandas as pd

from .signal import Signal
from .config import ConfigLoader, ConfigValidator, ValidationError
from .components import IndicatorCalculator, SignalManager, RiskManager

logger = structlog.get_logger()

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies using centralized configuration.
    
    All strategies must now get parameters from YAML configuration only.
    No hardcoded defaults are allowed in strategy implementations.
    """
    
    def __init__(self, strategy_name: str, config: Dict[str, Any]):
        """
        Initialize strategy with validated configuration from YAML.
        
        Args:
            strategy_name: Name of the strategy (must match schema)
            config: Raw configuration from YAML
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.strategy_name = strategy_name
        self.name = self.__class__.__name__
        
        # Load and validate configuration using centralized system
        try:
            self.params = ConfigLoader.load_strategy_params(strategy_name, config)
        except ValidationError as e:
            logger.error("Strategy configuration failed validation", 
                        strategy=strategy_name, error=str(e))
            raise
        
        # Extract common parameters (guaranteed to exist after validation)
        self.symbols = self.params['symbols']
        self.position_size = Decimal(str(self.params['position_size']))
        self.is_active = True
        
        # Initialize shared components
        self.indicator_calculator = IndicatorCalculator()
        self.signal_manager = SignalManager(strategy_name, self.params)
        self.risk_manager = RiskManager(strategy_name, self.params)
        
        # Get data requirements
        self.data_requirements = ConfigLoader.get_data_requirements(strategy_name, self.params)
        
        logger.info("Strategy initialized with new architecture", 
                   strategy=strategy_name,
                   symbols=self.symbols,
                   parameter_count=len(self.params))
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        This method should use the shared components (indicator_calculator,
        signal_manager, risk_manager) to analyze data and generate signals.
        
        Args:
            market_data: Dictionary with symbol -> OHLCV data
            
        Returns:
            List of Signal objects
        """
        pass
    
    def get_required_data(self) -> Dict[str, Any]:
        """
        Return requirements for market data based on strategy parameters.
        
        This now uses the centralized data requirements calculation.
        """
        return self.data_requirements
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all required indicators for this strategy.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with calculated indicators
        """
        return self.indicator_calculator.calculate_all_indicators(data, self.params)
    
    def create_signal(self, symbol: str, action: str, price: float, 
                     confidence: float, metadata: Optional[Dict[str, Any]] = None) -> Optional[Signal]:
        """
        Create a signal using the signal manager.
        
        Args:
            symbol: Trading symbol
            action: Signal action (BUY/SELL/HOLD)
            price: Current price
            confidence: Signal confidence (0-1)
            metadata: Additional signal metadata
            
        Returns:
            Signal object or None if signal should not be generated
        """
        if not self.signal_manager.should_generate_signal(symbol, action):
            return None
        
        # Record the signal for deduplication
        self.signal_manager.record_signal(symbol, action)
        
        # Create signal with risk management
        signal = self.signal_manager.create_signal(symbol, action, price, confidence, metadata)
        
        # Add risk management information
        if metadata is None:
            metadata = {}
        
        # Calculate stop levels if ATR is available in metadata
        atr = metadata.get('atr', None)
        stops = self.risk_manager.calculate_stop_levels(price, action, atr)
        
        # Add trailing stop loss parameters if enabled
        trailing_stop_enabled = self.get_parameter_value('trailing_stop_enabled', False)
        trailing_stop_distance = self.get_parameter_value('trailing_stop_distance', None)
        trailing_stop_type = self.get_parameter_value('trailing_stop_type', 'percentage')
        
        signal.metadata.update({
            'stop_loss': stops['stop_loss'],
            'take_profit': stops['take_profit'],
            'trailing_stop_enabled': trailing_stop_enabled,
            'trailing_stop_distance': trailing_stop_distance,
            'trailing_stop_type': trailing_stop_type,
            'risk_manager_params': {
                'stop_loss_pct': self.risk_manager.stop_loss_pct,
                'take_profit_pct': self.risk_manager.take_profit_pct,
                'use_leverage': self.risk_manager.use_leverage,
                'leverage': self.risk_manager.leverage,
                'trailing_stop_enabled': trailing_stop_enabled,
                'trailing_stop_distance': trailing_stop_distance,
                'trailing_stop_type': trailing_stop_type
            }
        })
        
        return signal
    
    def get_parameter_value(self, param_name: str, default: Any = None) -> Any:
        """
        Get a parameter value from the validated configuration.
        
        Args:
            param_name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.params.get(param_name, default)
    
    async def on_market_data_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle real-time market data updates"""
        pass
    
    async def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Handle trade execution feedback"""
        pass
    
    def set_active(self, active: bool) -> None:
        """Enable or disable the strategy"""
        self.is_active = active
        logger.info("Strategy status changed", strategy=self.name, active=active)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy information.
        
        Returns:
            Dictionary with strategy metadata and parameters
        """
        return {
            'name': self.name,
            'strategy_name': self.strategy_name,
            'symbols': self.symbols,
            'is_active': self.is_active,
            'parameters': self.params,
            'data_requirements': self.data_requirements,
            'risk_parameters': {
                'stop_loss_pct': self.risk_manager.stop_loss_pct,
                'take_profit_pct': self.risk_manager.take_profit_pct,
                'position_size': float(self.position_size),
                'use_leverage': self.risk_manager.use_leverage,
                'leverage': self.risk_manager.leverage
            }
        }