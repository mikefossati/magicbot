import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class BreakoutStrategy(BaseStrategy):
    """Breakout Trading Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.lookback_period = config.get('lookback_period', 20)
        self.breakout_threshold = config.get('breakout_threshold', 1.02)  # 2% breakout
        self.volume_confirmation = config.get('volume_confirmation', True)
        self.min_volatility = config.get('min_volatility', 0.005)  # 0.5% minimum volatility
        
        super().__init__(config)
        
        self.last_signals = {}
        self.resistance_levels = {}
        self.support_levels = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.lookback_period < 5:
            logger.error("Lookback period must be at least 5")
            return False
        
        if self.breakout_threshold <= 1.0:
            logger.error("Breakout threshold must be greater than 1.0")
            return False
        
        if self.min_volatility <= 0:
            logger.error("Minimum volatility must be positive")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.lookback_period + 10,
            'indicators': ['high', 'low', 'volume']
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        high_prices = df['high'].rolling(window=self.lookback_period)
        low_prices = df['low'].rolling(window=self.lookback_period)
        
        resistance = high_prices.max()
        support = low_prices.min()
        
        # Calculate volatility to filter out low-volatility periods
        price_range = df['high'] - df['low']
        avg_range = price_range.rolling(window=self.lookback_period).mean()
        volatility = avg_range / df['close']
        
        return {
            'resistance': resistance.iloc[-1] if not pd.isna(resistance.iloc[-1]) else None,
            'support': support.iloc[-1] if not pd.isna(support.iloc[-1]) else None,
            'volatility': volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0
        }
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if volume confirms the breakout"""
        if not self.volume_confirmation or 'volume' not in df.columns:
            return True
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        # Volume should be above average for valid breakout
        return current_volume > avg_volume * 1.2
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on breakouts"""
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
        """Analyze a single symbol for breakout signals"""
        if len(data) < self.lookback_period + 2:
            logger.warning("Insufficient data for breakout analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=self.lookback_period + 2)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
        
        # Calculate support and resistance levels
        levels = self._calculate_support_resistance(df)
        
        if levels['resistance'] is None or levels['support'] is None:
            return None
        
        # Check minimum volatility requirement
        if levels['volatility'] < self.min_volatility:
            return None
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        resistance = levels['resistance']
        support = levels['support']
        
        # Store levels for this symbol
        self.resistance_levels[symbol] = resistance
        self.support_levels[symbol] = support
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Bullish breakout: Price breaks above resistance
        if (prev_price <= resistance and 
            current_high > resistance * self.breakout_threshold and
            current_price > resistance):
            
            if self._check_volume_confirmation(df):
                signal_action = 'BUY'
                confidence = 0.7
                
                # Increase confidence based on breakout strength
                breakout_strength = (current_price - resistance) / resistance
                confidence = min(1.0, confidence + breakout_strength * 5)
                
                # Increase confidence if volatility is high
                if levels['volatility'] > 0.02:  # 2%
                    confidence += 0.1
        
        # Bearish breakout: Price breaks below support
        elif (prev_price >= support and 
              current_low < support / self.breakout_threshold and
              current_price < support):
            
            if self._check_volume_confirmation(df):
                signal_action = 'SELL'
                confidence = 0.7
                
                # Increase confidence based on breakout strength
                breakout_strength = (support - current_price) / support
                confidence = min(1.0, confidence + breakout_strength * 5)
                
                # Increase confidence if volatility is high
                if levels['volatility'] > 0.02:
                    confidence += 0.1
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("Breakout signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       current_price=current_price,
                       resistance=resistance,
                       support=support,
                       volatility=levels['volatility'])
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=Decimal(str(current_price)),
                metadata={
                    'resistance': float(resistance),
                    'support': float(support),
                    'volatility': float(levels['volatility']),
                    'lookback_period': self.lookback_period,
                    'breakout_threshold': self.breakout_threshold,
                    'volume_confirmation': self.volume_confirmation
                }
            )
        
        return None