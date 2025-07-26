import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) Trading Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = config.get('oversold', 30)
        self.overbought_threshold = config.get('overbought', 70)
        
        super().__init__(config)
        
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.rsi_period < 2:
            logger.error("RSI period must be at least 2")
            return False
        
        if not (0 < self.oversold_threshold < self.overbought_threshold < 100):
            logger.error("Invalid RSI thresholds", 
                        oversold=self.oversold_threshold, 
                        overbought=self.overbought_threshold)
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.rsi_period + 10,
            'indicators': ['rsi']
        }
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI using pandas"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on RSI"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                logger.warning("No market data available", symbol=symbol)
                continue
            
            try:
                signal = await self._analyze_symbol(symbol, market_data[symbol])
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
        
        return signals
    
    async def _analyze_symbol(self, symbol: str, data: List[Dict]) -> Optional[Signal]:
        """Analyze a single symbol for RSI signals"""
        if len(data) < self.rsi_period + 1:
            logger.warning("Insufficient data for RSI analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=self.rsi_period + 1)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        
        df['rsi'] = self._calculate_rsi(df['close'])
        
        if len(df) < 2:
            return None
        
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        
        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            return None
        
        current_price = Decimal(str(df['close'].iloc[-1]))
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Buy signal: RSI is oversold and starting to recover
        if current_rsi <= self.oversold_threshold and current_rsi > prev_rsi:
            signal_action = 'BUY'
            confidence = 0.6
            
            # Increase confidence if RSI was deeply oversold
            if current_rsi < 20:
                confidence += 0.2
            
            # Increase confidence if recovery is strong
            rsi_momentum = current_rsi - prev_rsi
            if rsi_momentum > 2:
                confidence += 0.1
        
        # Sell signal: RSI is overbought and starting to decline
        elif current_rsi >= self.overbought_threshold and current_rsi < prev_rsi:
            signal_action = 'SELL'
            confidence = 0.6
            
            # Increase confidence if RSI was deeply overbought
            if current_rsi > 80:
                confidence += 0.2
            
            # Increase confidence if decline is strong
            rsi_momentum = prev_rsi - current_rsi
            if rsi_momentum > 2:
                confidence += 0.1
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("RSI signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       rsi=current_rsi,
                       prev_rsi=prev_rsi)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price,
                metadata={
                    'rsi': float(current_rsi),
                    'prev_rsi': float(prev_rsi),
                    'rsi_period': self.rsi_period,
                    'oversold_threshold': self.oversold_threshold,
                    'overbought_threshold': self.overbought_threshold
                }
            )
        
        return None