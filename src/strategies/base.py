from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import structlog

logger = structlog.get_logger()

@dataclass
class Signal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: Decimal
    confidence: float  # 0.0 to 1.0
    price: Optional[Decimal] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.symbols = config.get('symbols', [])
        self.is_active = True
        self.position_size = Decimal(str(config.get('position_size', 0.01)))
        
        # Validate configuration
        if not self.validate_parameters():
            raise ValueError(f"Invalid parameters for strategy {self.name}")
        
        logger.info("Strategy initialized", strategy=self.name, symbols=self.symbols)
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        pass
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return requirements for market data (timeframes, indicators, etc.)"""
        return {
            'timeframes': ['1h'],  # Default timeframe
            'lookback_periods': 100,
            'indicators': []
        }
    
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