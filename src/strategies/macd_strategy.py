import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class MACDStrategy(BaseStrategy):
    """MACD (Moving Average Convergence Divergence) Trend Following Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        self.histogram_threshold = config.get('histogram_threshold', 0.0)
        
        super().__init__(config)
        
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.fast_period >= self.slow_period:
            logger.error("Fast period must be less than slow period")
            return False
        
        if any(period < 2 for period in [self.fast_period, self.slow_period, self.signal_period]):
            logger.error("All periods must be at least 2")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.slow_period + self.signal_period + 10,
            'indicators': ['macd']
        }
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD, Signal Line, and Histogram"""
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast_period).mean()
        ema_slow = prices.ewm(span=self.slow_period).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line (EMA of MACD)
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on MACD"""
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
        """Analyze a single symbol for MACD signals"""
        required_data = self.slow_period + self.signal_period + 1
        if len(data) < required_data:
            logger.warning("Insufficient data for MACD analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        
        macd_data = self._calculate_macd(df['close'])
        for key, values in macd_data.items():
            df[key] = values
        
        if len(df) < 2:
            return None
        
        current_macd = df['macd'].iloc[-1]
        current_signal = df['signal'].iloc[-1]
        current_histogram = df['histogram'].iloc[-1]
        prev_macd = df['macd'].iloc[-2]
        prev_signal = df['signal'].iloc[-2]
        prev_histogram = df['histogram'].iloc[-2]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_macd, current_signal, current_histogram,
                                       prev_macd, prev_signal, prev_histogram]):
            return None
        
        current_price = Decimal(str(df['close'].iloc[-1]))
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Bullish signal: MACD crosses above signal line
        if prev_macd <= prev_signal and current_macd > current_signal:
            signal_action = 'BUY'
            confidence = 0.7
            
            # Increase confidence if MACD is above zero (strong trend)
            if current_macd > 0:
                confidence += 0.1
            
            # Increase confidence based on histogram momentum
            if current_histogram > prev_histogram:
                confidence += 0.1
        
        # Bearish signal: MACD crosses below signal line
        elif prev_macd >= prev_signal and current_macd < current_signal:
            signal_action = 'SELL'
            confidence = 0.7
            
            # Increase confidence if MACD is below zero (strong downtrend)
            if current_macd < 0:
                confidence += 0.1
            
            # Increase confidence based on histogram momentum
            if current_histogram < prev_histogram:
                confidence += 0.1
        
        # Additional confirmation: Histogram zero line crossover
        elif (prev_histogram <= self.histogram_threshold and 
              current_histogram > self.histogram_threshold and
              current_macd > current_signal):
            signal_action = 'BUY'
            confidence = 0.6
        elif (prev_histogram >= self.histogram_threshold and 
              current_histogram < self.histogram_threshold and
              current_macd < current_signal):
            signal_action = 'SELL'
            confidence = 0.6
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("MACD signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       macd=current_macd,
                       signal_line=current_signal,
                       histogram=current_histogram)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price,
                metadata={
                    'macd': float(current_macd),
                    'signal_line': float(current_signal),
                    'histogram': float(current_histogram),
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period,
                    'signal_period': self.signal_period
                }
            )
        
        return None