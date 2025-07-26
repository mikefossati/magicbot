import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Trading Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2.0)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 0.02)
        
        super().__init__(config)
        
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.period < 2:
            logger.error("Bollinger Bands period must be at least 2")
            return False
        
        if self.std_dev <= 0:
            logger.error("Standard deviation must be positive")
            return False
        
        if self.mean_reversion_threshold <= 0:
            logger.error("Mean reversion threshold must be positive")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.period + 10,
            'indicators': ['bollinger_bands']
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)
        
        return {
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (prices - lower_band) / (upper_band - lower_band)
        }
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on Bollinger Bands"""
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
        """Analyze a single symbol for Bollinger Bands signals"""
        if len(data) < self.period + 1:
            logger.warning("Insufficient data for Bollinger Bands analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=self.period + 1)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        
        bb_data = self._calculate_bollinger_bands(df['close'])
        for key, values in bb_data.items():
            df[key] = values
        
        if len(df) < 2:
            return None
        
        current_price = df['close'].iloc[-1]
        current_upper = df['upper_band'].iloc[-1]
        current_lower = df['lower_band'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        current_bb_position = df['bb_position'].iloc[-1]
        prev_bb_position = df['bb_position'].iloc[-2]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_upper, current_lower, current_sma, 
                                       current_bb_position, prev_bb_position]):
            return None
        
        current_price_decimal = Decimal(str(current_price))
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Buy signal: Price is near lower band and showing upward momentum
        if (current_bb_position <= 0.2 and current_price > df['close'].iloc[-2]):
            signal_action = 'BUY'
            confidence = 0.6
            
            # Increase confidence if price is very close to lower band
            if current_bb_position <= 0.1:
                confidence += 0.2
            
            # Increase confidence if bands are wide (high volatility)
            bb_width = df['bb_width'].iloc[-1]
            if bb_width > 0.05:  # 5% width
                confidence += 0.1
        
        # Sell signal: Price is near upper band and showing downward momentum
        elif (current_bb_position >= 0.8 and current_price < df['close'].iloc[-2]):
            signal_action = 'SELL'
            confidence = 0.6
            
            # Increase confidence if price is very close to upper band
            if current_bb_position >= 0.9:
                confidence += 0.2
            
            # Increase confidence if bands are wide
            bb_width = df['bb_width'].iloc[-1]
            if bb_width > 0.05:
                confidence += 0.1
        
        # Mean reversion signal: Price moves back toward middle band
        elif (0.4 <= current_bb_position <= 0.6 and 
              abs(prev_bb_position - 0.5) > abs(current_bb_position - 0.5)):
            # Close positions when price returns to middle
            if self.last_signals.get(symbol) in ['BUY', 'SELL']:
                signal_action = 'HOLD'  # This will trigger position closure
                confidence = 0.5
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' and last_signal != 'HOLD':
            # Allow HOLD signals to close positions
            pass
        elif signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD' or last_signal in ['BUY', 'SELL']:
            logger.info("Bollinger Bands signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       bb_position=current_bb_position,
                       price=current_price,
                       upper_band=current_upper,
                       lower_band=current_lower)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price_decimal,
                metadata={
                    'bb_position': float(current_bb_position),
                    'upper_band': float(current_upper),
                    'lower_band': float(current_lower),
                    'sma': float(current_sma),
                    'bb_width': float(df['bb_width'].iloc[-1]),
                    'period': self.period,
                    'std_dev': self.std_dev
                }
            )
        
        return None