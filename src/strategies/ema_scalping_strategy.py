import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class EMAScalpingStrategy(BaseStrategy):
    """
    EMA Crossover Scalping Strategy with Volume Confirmation
    
    Designed for short-term scalping on 1m-5m timeframes
    Key Features:
    - Fast EMA crossovers for quick entries/exits
    - Volume confirmation to filter false signals
    - RSI filter to avoid overbought/oversold extremes
    - Tight stop-loss and take-profit levels
    - Multi-timeframe confirmation
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Fast EMAs for scalping
        self.fast_ema = config.get('fast_ema', 5)
        self.slow_ema = config.get('slow_ema', 13)
        self.signal_ema = config.get('signal_ema', 21)  # Trend filter
        
        # Volume confirmation
        self.volume_period = config.get('volume_period', 10)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        
        # RSI filter
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.5)  # 0.5%
        self.take_profit_pct = config.get('take_profit_pct', 1.0)  # 1.0%
        self.risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        
        # Signal filtering
        self.min_price_movement = config.get('min_price_movement', 0.1)  # 0.1%
        self.consolidation_period = config.get('consolidation_period', 3)
        
        super().__init__(config)
        
        self.last_signals = {}
        self.position_states = {}  # Track position states
        
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if self.fast_ema >= self.slow_ema:
            logger.error("Fast EMA must be less than slow EMA")
            return False
        
        if any(period < 2 for period in [self.fast_ema, self.slow_ema, self.signal_ema]):
            logger.error("All EMA periods must be at least 2")
            return False
        
        if self.stop_loss_pct <= 0 or self.take_profit_pct <= 0:
            logger.error("Stop loss and take profit must be positive")
            return False
        
        if not self.symbols:
            logger.error("No symbols configured")
            return False
        
        return True
    
    def get_required_data(self) -> Dict[str, Any]:
        """Return data requirements for this strategy"""
        return {
            'timeframes': ['1m', '5m'],  # Primary scalping timeframes
            'lookback_periods': max(self.signal_ema, self.volume_period, self.rsi_period) + 20,
            'indicators': ['ema', 'volume', 'rsi']
        }
    
    def _calculate_emas(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate multiple EMAs"""
        return {
            'fast_ema': prices.ewm(span=self.fast_ema).mean(),
            'slow_ema': prices.ewm(span=self.slow_ema).mean(),
            'signal_ema': prices.ewm(span=self.signal_ema).mean()
        }
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI for overbought/oversold filter"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=self.rsi_period).mean()
        avg_loss = loss.ewm(span=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_volume_confirmation(self, volumes: pd.Series) -> Dict[str, pd.Series]:
        """Calculate volume indicators"""
        volume_ma = volumes.rolling(window=self.volume_period).mean()
        volume_ratio = volumes / volume_ma
        
        return {
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'high_volume': volume_ratio > self.volume_multiplier
        }
    
    def _detect_consolidation(self, highs: pd.Series, lows: pd.Series, period: int = None) -> pd.Series:
        """Detect price consolidation (low volatility periods)"""
        if period is None:
            period = self.consolidation_period
        
        high_max = highs.rolling(window=period).max()
        low_min = lows.rolling(window=period).min()
        price_range = (high_max - low_min) / low_min
        
        # Consider consolidation if price range < 0.5% over the period
        return price_range < 0.005
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 3) -> pd.Series:
        """Calculate short-term momentum"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate scalping signals based on EMA crossovers"""
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
        """Analyze a single symbol for scalping signals"""
        required_data = max(self.signal_ema, self.volume_period, self.rsi_period) + 5
        if len(data) < required_data:
            logger.warning("Insufficient data for scalping analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        df = pd.DataFrame(data)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate all indicators
        emas = self._calculate_emas(df['close'])
        for key, values in emas.items():
            df[key] = values
        
        df['rsi'] = self._calculate_rsi(df['close'])
        
        volume_data = self._calculate_volume_confirmation(df['volume'])
        for key, values in volume_data.items():
            df[key] = values
        
        df['consolidation'] = self._detect_consolidation(df['high'], df['low'])
        df['momentum'] = self._calculate_momentum(df['close'])
        
        if len(df) < 3:
            return None
        
        # Current and previous values
        current_price = df['close'].iloc[-1]
        current_fast = df['fast_ema'].iloc[-1]
        current_slow = df['slow_ema'].iloc[-1]
        current_signal = df['signal_ema'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-1]
        current_momentum = df['momentum'].iloc[-1]
        
        prev_fast = df['fast_ema'].iloc[-2]
        prev_slow = df['slow_ema'].iloc[-2]
        prev2_fast = df['fast_ema'].iloc[-3]
        prev2_slow = df['slow_ema'].iloc[-3]
        
        is_consolidating = df['consolidation'].iloc[-1]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_fast, current_slow, current_signal, 
                                       current_rsi, prev_fast, prev_slow]):
            return None
        
        current_price_decimal = Decimal(str(current_price))
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Get current position state
        current_state = self.position_states.get(symbol, 'NEUTRAL')
        
        # BULLISH SIGNAL: Fast EMA crosses above Slow EMA
        if (prev_fast <= prev_slow and current_fast > current_slow and
            prev2_fast <= prev2_slow):  # Confirm crossover momentum
            
            # Additional filters for quality signals
            bullish_conditions = []
            
            # 1. Trend alignment: Price above signal EMA
            if current_price > current_signal:
                bullish_conditions.append("trend_alignment")
                confidence += 0.2
            
            # 2. RSI not overbought
            if current_rsi < self.rsi_overbought:
                bullish_conditions.append("rsi_filter")
                confidence += 0.2
            
            # 3. Volume confirmation
            if current_volume_ratio > self.volume_multiplier:
                bullish_conditions.append("volume_confirmation")
                confidence += 0.2
            
            # 4. Positive momentum
            if current_momentum > 0:
                bullish_conditions.append("momentum")
                confidence += 0.1
            
            # 5. Not in consolidation (avoid whipsaws)
            if not is_consolidating:
                bullish_conditions.append("breakout")
                confidence += 0.1
            
            # 6. EMA separation (strong crossover)
            ema_separation = abs(current_fast - current_slow) / current_slow
            if ema_separation > 0.001:  # 0.1% separation
                bullish_conditions.append("strong_crossover")
                confidence += 0.1
            
            # Minimum conditions for signal
            if len(bullish_conditions) >= 3:
                signal_action = 'BUY'
                confidence = min(0.9, 0.6 + confidence)
        
        # BEARISH SIGNAL: Fast EMA crosses below Slow EMA
        elif (prev_fast >= prev_slow and current_fast < current_slow and
              prev2_fast >= prev2_slow):  # Confirm crossover momentum
            
            bearish_conditions = []
            
            # 1. Trend alignment: Price below signal EMA
            if current_price < current_signal:
                bearish_conditions.append("trend_alignment")
                confidence += 0.2
            
            # 2. RSI not oversold
            if current_rsi > self.rsi_oversold:
                bearish_conditions.append("rsi_filter")
                confidence += 0.2
            
            # 3. Volume confirmation
            if current_volume_ratio > self.volume_multiplier:
                bearish_conditions.append("volume_confirmation")
                confidence += 0.2
            
            # 4. Negative momentum
            if current_momentum < 0:
                bearish_conditions.append("momentum")
                confidence += 0.1
            
            # 5. Not in consolidation
            if not is_consolidating:
                bearish_conditions.append("breakout")
                confidence += 0.1
            
            # 6. EMA separation
            ema_separation = abs(current_slow - current_fast) / current_slow
            if ema_separation > 0.001:
                bearish_conditions.append("strong_crossover")
                confidence += 0.1
            
            if len(bearish_conditions) >= 3:
                signal_action = 'SELL'
                confidence = min(0.9, 0.6 + confidence)
        
        # EXIT SIGNALS: Early exit conditions
        elif current_state != 'NEUTRAL':
            # Exit long position
            if (current_state == 'LONG' and 
                (current_fast < current_slow or current_rsi > self.rsi_overbought)):
                signal_action = 'SELL'
                confidence = 0.7
            
            # Exit short position  
            elif (current_state == 'SHORT' and 
                  (current_fast > current_slow or current_rsi < self.rsi_oversold)):
                signal_action = 'BUY'
                confidence = 0.7
        
        # Update position state
        if signal_action == 'BUY':
            self.position_states[symbol] = 'LONG'
        elif signal_action == 'SELL':
            self.position_states[symbol] = 'SHORT'
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            # Calculate stop loss and take profit levels
            if signal_action == 'BUY':
                stop_loss = current_price * (1 - self.stop_loss_pct / 100)
                take_profit = current_price * (1 + self.take_profit_pct / 100)
            else:  # SELL
                stop_loss = current_price * (1 + self.stop_loss_pct / 100)
                take_profit = current_price * (1 - self.take_profit_pct / 100)
            
            logger.info("EMA Scalping signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       current_price=current_price,
                       fast_ema=current_fast,
                       slow_ema=current_slow,
                       rsi=current_rsi,
                       volume_ratio=current_volume_ratio,
                       stop_loss=stop_loss,
                       take_profit=take_profit)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price_decimal,
                metadata={
                    'fast_ema': float(current_fast),
                    'slow_ema': float(current_slow),
                    'signal_ema': float(current_signal),
                    'rsi': float(current_rsi),
                    'volume_ratio': float(current_volume_ratio),
                    'momentum': float(current_momentum),
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'consolidation': bool(is_consolidating),
                    'ema_separation': float(abs(current_fast - current_slow) / current_slow * 100),
                    'scalping_timeframe': '1m-5m'
                }
            )
        
        return None