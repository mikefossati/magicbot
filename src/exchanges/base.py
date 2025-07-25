from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class Order:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: str = 'MARKET'
    order_id: Optional[str] = None

@dataclass
class Balance:
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal

@dataclass
class MarketData:
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: int

class ExchangeInterface(ABC):
    """Abstract base class for exchange implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to exchange"""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> List[Balance]:
        """Get account balance for all assets"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place a trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get historical kline/candlestick data"""
        pass