import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class MovingAverageCrossover(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        # Set periods before calling super().__init__ since validate_parameters needs them
        self.fast_period = config.get('fast_period', 10)
        self.slow_period = config.get('slow_period', 30)
        
        super().__init__(config)
        
        # Store previous signals to avoid duplicate trades
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.fast_period >= self.slow_period:
            logger.error("Fast period must be less than slow period")
            return False
        
        if self.fast_period < 2 or self.slow_period < 2:
            logger.error("Periods must be at least 2")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.slow_period + 10,  # Extra buffer
            'indicators': ['sma']
        }
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on MA crossover"""
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
        """Analyze a single symbol for trading signals"""
        if len(data) < self.slow_period:
            logger.warning("Insufficient data for analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=self.slow_period)
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Get the last few values to determine crossover
        if len(df) < 2:
            return None
        
        current_fast = df['fast_ma'].iloc[-1]
        current_slow = df['slow_ma'].iloc[-1]
        prev_fast = df['fast_ma'].iloc[-2]
        prev_slow = df['slow_ma'].iloc[-2]
        
        # Check for NaN values
        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None
        
        current_price = Decimal(str(df['close'].iloc[-1]))
        
        # Determine signal type
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Bullish crossover: fast MA crosses above slow MA
        if prev_fast <= prev_slow and current_fast > current_slow:
            signal_action = 'BUY'
            confidence = 0.7  # Base confidence for crossover
            
            # Increase confidence based on the strength of the crossover
            crossover_strength = (current_fast - current_slow) / current_slow
            confidence = min(1.0, confidence + crossover_strength * 2)
        
        # Bearish crossover: fast MA crosses below slow MA
        elif prev_fast >= prev_slow and current_fast < current_slow:
            signal_action = 'SELL'
            confidence = 0.7
            
            crossover_strength = (current_slow - current_fast) / current_slow
            confidence = min(1.0, confidence + crossover_strength * 2)
        
        # Only generate signal if it's different from the last one
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        # Store the signal
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("Signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       fast_ma=current_fast,
                       slow_ma=current_slow)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price,
                metadata={
                    'fast_ma': float(current_fast),
                    'slow_ma': float(current_slow),
                    'fast_period': self.fast_period,
                    'slow_period': self.slow_period
                }
            )
        
        return None