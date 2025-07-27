"""
Signal data class for trading strategies.

This module contains the Signal class to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from decimal import Decimal

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