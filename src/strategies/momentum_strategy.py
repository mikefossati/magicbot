import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class MomentumStrategy(BaseStrategy):
    """Price Momentum Trend Following Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.momentum_period = config.get('momentum_period', 14)
        self.sma_period = config.get('sma_period', 20)
        self.momentum_threshold = config.get('momentum_threshold', 5.0)  # Percentage
        self.volume_confirmation = config.get('volume_confirmation', True)
        
        super().__init__(config)
        
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.momentum_period < 2:
            logger.error("Momentum period must be at least 2")
            return False
        
        if self.sma_period < 2:
            logger.error("SMA period must be at least 2")
            return False
        
        if self.momentum_threshold <= 0:
            logger.error("Momentum threshold must be positive")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': max(self.momentum_period, self.sma_period) + 10,
            'indicators': ['momentum', 'sma', 'volume']
        }
    
    def _calculate_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate price momentum as percentage change"""
        return ((prices - prices.shift(self.momentum_period)) / prices.shift(self.momentum_period)) * 100
    
    def _calculate_volume_confirmation(self, volumes: pd.Series, period: int = 5) -> pd.Series:
        """Calculate volume moving average for confirmation"""
        return volumes.rolling(window=period).mean()
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on momentum"""
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
        """Analyze a single symbol for momentum signals"""
        required_data = max(self.momentum_period, self.sma_period) + 1
        if len(data) < required_data:
            logger.warning("Insufficient data for momentum analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float) if 'volume' in df.columns else 1.0
        
        # Calculate indicators
        df['momentum'] = self._calculate_momentum(df['close'])
        df['sma'] = df['close'].rolling(window=self.sma_period).mean()
        df['volume_ma'] = self._calculate_volume_confirmation(df['volume'])
        
        if len(df) < 2:
            return None
        
        current_price = df['close'].iloc[-1]
        current_momentum = df['momentum'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        current_volume_ma = df['volume_ma'].iloc[-1]
        prev_momentum = df['momentum'].iloc[-2]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_momentum, current_sma, prev_momentum]):
            return None
        
        current_price_decimal = Decimal(str(current_price))
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Strong upward momentum
        if (current_momentum > self.momentum_threshold and 
            current_price > current_sma and
            current_momentum > prev_momentum):
            
            signal_action = 'BUY'
            confidence = 0.6
            
            # Increase confidence for very strong momentum
            if current_momentum > self.momentum_threshold * 1.5:
                confidence += 0.2
            
            # Volume confirmation
            if self.volume_confirmation and current_volume > current_volume_ma:
                confidence += 0.1
            
            # Price well above SMA
            price_above_sma = ((current_price - current_sma) / current_sma) * 100
            if price_above_sma > 2:  # 2% above SMA
                confidence += 0.1
        
        # Strong downward momentum
        elif (current_momentum < -self.momentum_threshold and 
              current_price < current_sma and
              current_momentum < prev_momentum):
            
            signal_action = 'SELL'
            confidence = 0.6
            
            # Increase confidence for very strong negative momentum
            if current_momentum < -self.momentum_threshold * 1.5:
                confidence += 0.2
            
            # Volume confirmation
            if self.volume_confirmation and current_volume > current_volume_ma:
                confidence += 0.1
            
            # Price well below SMA
            price_below_sma = ((current_sma - current_price) / current_sma) * 100
            if price_below_sma > 2:  # 2% below SMA
                confidence += 0.1
        
        # Momentum reversal signals (trend exhaustion)
        elif (abs(current_momentum) > self.momentum_threshold * 2 and
              ((current_momentum > 0 and current_momentum < prev_momentum) or
               (current_momentum < 0 and current_momentum > prev_momentum))):
            
            # Potential trend reversal - opposite signal with lower confidence
            if current_momentum > 0:  # Weakening uptrend
                signal_action = 'SELL'
                confidence = 0.4
            else:  # Weakening downtrend
                signal_action = 'BUY'
                confidence = 0.4
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("Momentum signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       momentum=current_momentum,
                       price=current_price,
                       sma=current_sma)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price_decimal,
                metadata={
                    'momentum': float(current_momentum),
                    'momentum_period': self.momentum_period,
                    'sma': float(current_sma),
                    'sma_period': self.sma_period,
                    'momentum_threshold': self.momentum_threshold,
                    'volume': float(current_volume),
                    'volume_ma': float(current_volume_ma) if not pd.isna(current_volume_ma) else None
                }
            )
        
        return None