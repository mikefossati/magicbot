import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog
import asyncio

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator Mean Reversion Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.k_period = config.get('k_period', 14)
        self.d_period = config.get('d_period', 3)
        self.smooth_period = config.get('smooth_period', 3)
        self.oversold_threshold = config.get('oversold', 20)
        self.overbought_threshold = config.get('overbought', 80)
        self.divergence_lookback = config.get('divergence_lookback', 5)
        
        super().__init__(config)
        
        self.last_signals = {}
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        if any(period < 2 for period in [self.k_period, self.d_period, self.smooth_period]):
            logger.error("All periods must be at least 2")
            return False
        
        if not (0 < self.oversold_threshold < self.overbought_threshold < 100):
            logger.error("Invalid Stochastic thresholds", 
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
            'lookback_periods': self.k_period + self.d_period + self.smooth_period + 10,
            'indicators': ['stochastic']
        }
    
    def _calculate_stochastic(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator %K and %D"""
        # Calculate %K
        lowest_low = lows.rolling(window=self.k_period).min()
        highest_high = highs.rolling(window=self.k_period).max()
        
        k_percent = 100 * (closes - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K if smooth_period > 1
        if self.smooth_period > 1:
            k_percent = k_percent.rolling(window=self.smooth_period).mean()
        
        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent,
            'highest_high': highest_high,
            'lowest_low': lowest_low
        }
    
    def _calculate_stochastic_indicators(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate Stochastic indicators in a CPU-intensive function"""
        df = pd.DataFrame(data)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        
        stoch_data = self._calculate_stochastic(df['high'], df['low'], df['close'])
        for key, values in stoch_data.items():
            df[key] = values
        
        if len(df) < 2:
            return {'error': 'insufficient_data'}
        
        current_k = df['k_percent'].iloc[-1]
        current_d = df['d_percent'].iloc[-1]
        prev_k = df['k_percent'].iloc[-2]
        prev_d = df['d_percent'].iloc[-2]
        current_price = df['close'].iloc[-1]
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_k, current_d, prev_k, prev_d]):
            return {'error': 'nan_values'}
        
        # Calculate divergence
        divergence = self._detect_divergence(df['close'], df['k_percent'], self.divergence_lookback)
        
        return {
            'current_k': current_k,
            'current_d': current_d,
            'prev_k': prev_k,
            'prev_d': prev_d,
            'current_price': current_price,
            'divergence': divergence
        }
    
    def _detect_divergence(self, prices: pd.Series, oscillator: pd.Series, lookback: int) -> str:
        """Detect bullish or bearish divergence"""
        if len(prices) < lookback or len(oscillator) < lookback:
            return 'NONE'
        
        recent_prices = prices.iloc[-lookback:]
        recent_oscillator = oscillator.iloc[-lookback:]
        
        # Simple divergence detection
        price_trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
        oscillator_trend = recent_oscillator.iloc[-1] - recent_oscillator.iloc[0]
        
        # Bullish divergence: price making lower lows, oscillator making higher lows
        if price_trend < 0 and oscillator_trend > 0:
            return 'BULLISH'
        
        # Bearish divergence: price making higher highs, oscillator making lower highs
        elif price_trend > 0 and oscillator_trend < 0:
            return 'BEARISH'
        
        return 'NONE'
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on Stochastic Oscillator"""
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
        """Analyze a single symbol for Stochastic signals"""
        required_data = self.k_period + self.d_period + self.smooth_period + 1
        if len(data) < required_data:
            logger.warning("Insufficient data for Stochastic analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        # Run CPU-intensive pandas operations in thread pool
        stoch_results = await asyncio.to_thread(self._calculate_stochastic_indicators, data)
        
        # Handle errors from calculation
        if 'error' in stoch_results:
            if stoch_results['error'] in ['insufficient_data', 'nan_values']:
                return None
        
        current_k = stoch_results['current_k']
        current_d = stoch_results['current_d']
        prev_k = stoch_results['prev_k']
        prev_d = stoch_results['prev_d']
        current_price_decimal = Decimal(str(stoch_results['current_price']))
        divergence = stoch_results['divergence']
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Buy signals
        # 1. Oversold bounce: %K crosses above %D in oversold territory
        if (current_k <= self.oversold_threshold and 
            prev_k <= prev_d and current_k > current_d):
            signal_action = 'BUY'
            confidence = 0.7
            
            # Increase confidence if deeply oversold
            if current_k < 10:
                confidence += 0.2
            
            # Bullish divergence confirmation
            if divergence == 'BULLISH':
                confidence += 0.1
        
        # 2. Bullish divergence in oversold region
        elif (current_k <= self.oversold_threshold + 10 and 
              divergence == 'BULLISH'):
            signal_action = 'BUY'
            confidence = 0.6
        
        # Sell signals
        # 1. Overbought reversal: %K crosses below %D in overbought territory
        elif (current_k >= self.overbought_threshold and 
              prev_k >= prev_d and current_k < current_d):
            signal_action = 'SELL'
            confidence = 0.7
            
            # Increase confidence if deeply overbought
            if current_k > 90:
                confidence += 0.2
            
            # Bearish divergence confirmation
            if divergence == 'BEARISH':
                confidence += 0.1
        
        # 2. Bearish divergence in overbought region
        elif (current_k >= self.overbought_threshold - 10 and 
              divergence == 'BEARISH'):
            signal_action = 'SELL'
            confidence = 0.6
        
        # Additional confirmation: %K and %D momentum
        k_momentum = current_k - prev_k
        d_momentum = current_d - prev_d
        
        if signal_action == 'BUY' and k_momentum > 0 and d_momentum > 0:
            confidence = min(1.0, confidence + 0.1)
        elif signal_action == 'SELL' and k_momentum < 0 and d_momentum < 0:
            confidence = min(1.0, confidence + 0.1)
        
        # Avoid duplicate signals
        last_signal = self.last_signals.get(symbol)
        if signal_action == 'HOLD' or (last_signal and last_signal == signal_action):
            return None
        
        self.last_signals[symbol] = signal_action
        
        if signal_action != 'HOLD':
            logger.info("Stochastic signal generated",
                       strategy=self.name,
                       symbol=symbol,
                       action=signal_action,
                       confidence=confidence,
                       k_percent=current_k,
                       d_percent=current_d,
                       divergence=divergence)
            
            return Signal(
                symbol=symbol,
                action=signal_action,
                quantity=self.position_size,
                confidence=confidence,
                price=current_price_decimal,
                metadata={
                    'k_percent': float(current_k),
                    'd_percent': float(current_d),
                    'k_period': self.k_period,
                    'd_period': self.d_period,
                    'smooth_period': self.smooth_period,
                    'oversold_threshold': self.oversold_threshold,
                    'overbought_threshold': self.overbought_threshold,
                    'divergence': divergence,
                    'k_momentum': float(k_momentum),
                    'd_momentum': float(d_momentum)
                }
            )
        
        return None