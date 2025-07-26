import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class MeanReversionRSI(BaseStrategy):
    """Advanced RSI Mean Reversion Strategy with Multiple Timeframes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.rsi_period = config.get('rsi_period', 14)
        self.ma_period = config.get('ma_period', 50)
        self.extreme_oversold = config.get('extreme_oversold', 15)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)
        self.extreme_overbought = config.get('extreme_overbought', 85)
        self.rsi_ma_period = config.get('rsi_ma_period', 5)  # RSI smoothing
        self.volume_confirmation = config.get('volume_confirmation', True)
        
        super().__init__(config)
        
        self.last_signals = {}
        self.position_states = {}  # Track position states for scaling
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.rsi_period < 2:
            logger.error("RSI period must be at least 2")
            return False
        
        if self.ma_period < 2:
            logger.error("MA period must be at least 2")
            return False
        
        thresholds = [self.extreme_oversold, self.oversold, self.overbought, self.extreme_overbought]
        if not all(0 < t < 100 for t in thresholds) or thresholds != sorted(thresholds):
            logger.error("Invalid RSI thresholds", thresholds=thresholds)
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': max(self.rsi_period, self.ma_period) + 20,
            'indicators': ['rsi', 'sma', 'volume']
        }
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI with exponential smoothing"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential smoothing instead of simple moving average
        avg_gain = gain.ewm(span=self.rsi_period).mean()
        avg_loss = loss.ewm(span=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Apply smoothing to RSI
        if self.rsi_ma_period > 1:
            rsi = rsi.rolling(window=self.rsi_ma_period).mean()
        
        return rsi
    
    def _calculate_rsi_divergence(self, prices: pd.Series, rsi: pd.Series, lookback: int = 10) -> str:
        """Detect RSI divergence patterns"""
        if len(prices) < lookback or len(rsi) < lookback:
            return 'NONE'
        
        recent_prices = prices.iloc[-lookback:]
        recent_rsi = rsi.iloc[-lookback:]
        
        # Find local peaks and troughs
        price_peaks = recent_prices.rolling(window=3, center=True).max() == recent_prices
        price_troughs = recent_prices.rolling(window=3, center=True).min() == recent_prices
        rsi_peaks = recent_rsi.rolling(window=3, center=True).max() == recent_rsi
        rsi_troughs = recent_rsi.rolling(window=3, center=True).min() == recent_rsi
        
        # Bullish divergence: price lower low, RSI higher low
        if (price_troughs.sum() >= 2 and rsi_troughs.sum() >= 2):
            price_lows = recent_prices[price_troughs].dropna()
            rsi_lows = recent_rsi[rsi_troughs].dropna()
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                    return 'BULLISH'
        
        # Bearish divergence: price higher high, RSI lower high
        if (price_peaks.sum() >= 2 and rsi_peaks.sum() >= 2):
            price_highs = recent_prices[price_peaks].dropna()
            rsi_highs = recent_rsi[rsi_peaks].dropna()
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                    return 'BEARISH'
        
        return 'NONE'
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on advanced RSI mean reversion"""
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
        """Analyze a single symbol for advanced RSI mean reversion signals"""
        required_data = max(self.rsi_period, self.ma_period) + 20
        if len(data) < required_data:
            logger.warning("Insufficient data for RSI mean reversion analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        df = pd.DataFrame(data)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float) if 'volume' in df.columns else 1.0
        
        # Calculate indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        
        if len(df) < 3:
            return None
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        prev2_rsi = df['rsi'].iloc[-3]
        current_ma = df['ma'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_rsi, prev_rsi, prev2_rsi, current_ma]):
            return None
        
        current_price_decimal = Decimal(str(current_price))
        
        # Detect divergence
        divergence = self._calculate_rsi_divergence(df['close'], df['rsi'])
        
        signal_action = 'HOLD'
        confidence = 0.0
        position_size_multiplier = 1.0
        
        # Get current position state
        current_state = self.position_states.get(symbol, 'NEUTRAL')
        
        # Extreme oversold conditions - Strong buy signals
        if current_rsi <= self.extreme_oversold:
            signal_action = 'BUY'
            confidence = 0.8
            position_size_multiplier = 1.5  # Increase position size for extreme conditions
            
            # RSI showing momentum reversal
            if current_rsi > prev_rsi > prev2_rsi:
                confidence += 0.1
            
            # Price below MA (trend context)
            if current_price < current_ma:
                confidence += 0.1
        
        # Regular oversold with confirmation
        elif (current_rsi <= self.oversold and 
              current_rsi > prev_rsi and  # RSI turning up
              (divergence == 'BULLISH' or current_price < current_ma * 0.98)):
            
            signal_action = 'BUY'
            confidence = 0.7
            
            # Multiple confirmations
            if divergence == 'BULLISH':
                confidence += 0.15
            
            if current_rsi > prev_rsi > prev2_rsi:  # RSI momentum
                confidence += 0.1
        
        # Extreme overbought conditions - Strong sell signals
        elif current_rsi >= self.extreme_overbought:
            signal_action = 'SELL'
            confidence = 0.8
            position_size_multiplier = 1.5
            
            # RSI showing momentum reversal
            if current_rsi < prev_rsi < prev2_rsi:
                confidence += 0.1
            
            # Price above MA
            if current_price > current_ma:
                confidence += 0.1
        
        # Regular overbought with confirmation
        elif (current_rsi >= self.overbought and 
              current_rsi < prev_rsi and  # RSI turning down
              (divergence == 'BEARISH' or current_price > current_ma * 1.02)):
            
            signal_action = 'SELL'
            confidence = 0.7
            
            if divergence == 'BEARISH':
                confidence += 0.15
            
            if current_rsi < prev_rsi < prev2_rsi:
                confidence += 0.1
        
        # Mean reversion back to neutral zone
        elif (30 < current_rsi < 70 and 
              ((current_state == 'OVERSOLD' and current_rsi > 50) or
               (current_state == 'OVERBOUGHT' and current_rsi < 50))):
            
            # Close extreme positions when returning to neutral
            if current_state in ['OVERSOLD', 'OVERBOUGHT']:
                signal_action = 'HOLD'  # This will close positions
                confidence = 0.6
        
        # Volume confirmation
        if self.volume_confirmation and signal_action in ['BUY', 'SELL']:
            if current_volume > volume_ma * 1.2:  # Above average volume
                confidence = min(1.0, confidence + 0.1)
            elif current_volume < volume_ma * 0.8:  # Below average volume
                confidence *= 0.9  # Reduce confidence
        
        # Update position state
        if signal_action == 'BUY' and current_rsi <= self.extreme_oversold:
            self.position_states[symbol] = 'OVERSOLD'
        elif signal_action == 'SELL' and current_rsi >= self.extreme_overbought:
            self.position_states[symbol] = 'OVERBOUGHT'
        elif 30 < current_rsi < 70:
            self.position_states[symbol] = 'NEUTRAL'
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' and current_state == 'NEUTRAL':
            return None
        elif signal_action != 'HOLD' and last_signal == signal_action:
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD' or current_state != 'NEUTRAL':
            # Adjust position size
            adjusted_position_size = self.position_size * Decimal(str(position_size_multiplier))
            
            logger.info("Advanced RSI mean reversion signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       current_rsi=current_rsi,
                       prev_rsi=prev_rsi,
                       divergence=divergence,
                       position_state=current_state,
                       position_multiplier=position_size_multiplier)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=adjusted_position_size,
                confidence=confidence,
                price=current_price_decimal,
                metadata={
                    'rsi': float(current_rsi),
                    'prev_rsi': float(prev_rsi),
                    'rsi_period': self.rsi_period,
                    'ma': float(current_ma),
                    'ma_period': self.ma_period,
                    'divergence': divergence,
                    'position_state': current_state,
                    'position_multiplier': position_size_multiplier,
                    'volume_confirmation': current_volume > volume_ma if self.volume_confirmation else None,
                    'extreme_oversold': self.extreme_oversold,
                    'extreme_overbought': self.extreme_overbought
                }
            )
        
        return None