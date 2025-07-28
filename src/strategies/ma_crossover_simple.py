import pandas as pd
from typing import Dict, List, Any, Optional
import structlog

from .base import BaseStrategy
from .signal import Signal

logger = structlog.get_logger()

class SimpleMovingAverageCrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy following CLAUDE.md architecture.
    
    Uses shared components and proper parameter access patterns.
    Generates signals when fast MA crosses above/below slow MA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Must call super with strategy name matching schema
        super().__init__('ma_crossover_simple', config)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on MA crossover"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            try:
                signal = await self._analyze_symbol(symbol, market_data[symbol])
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
        
        return signals
    
    async def _analyze_symbol(self, symbol: str, data: List[Dict]) -> Optional[Signal]:
        """Analyze a single symbol using new architecture patterns"""
        # Get parameters from validated configuration (no hardcoded defaults!)
        fast_period = self.get_parameter_value('fast_period')
        slow_period = self.get_parameter_value('slow_period')
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Check minimum data requirements (need a few extra points for trend analysis)
        required_data = max(fast_period, slow_period) + 5
        if len(df) < required_data:
            logger.debug("Insufficient data", 
                        symbol=symbol, 
                        data_points=len(df), 
                        required=required_data)
            return None
        
        # Calculate indicators using shared component (following architecture)
        indicators = self.calculate_indicators(df)
        
        # Get MA values from shared indicators
        fast_ma = indicators.get('sma_fast')
        slow_ma = indicators.get('sma_slow')
        
        if fast_ma is None or slow_ma is None or len(fast_ma) < 2 or len(slow_ma) < 2:
            return None
        
        # Current and previous MA values - handle Series properly
        try:
            current_fast_val = fast_ma.iloc[-1]
            current_slow_val = slow_ma.iloc[-1] 
            prev_fast_val = fast_ma.iloc[-2]
            prev_slow_val = slow_ma.iloc[-2]
            
            # Convert to float, handling Series/scalar
            current_fast = float(current_fast_val.iloc[0] if hasattr(current_fast_val, 'iloc') else current_fast_val)
            current_slow = float(current_slow_val.iloc[0] if hasattr(current_slow_val, 'iloc') else current_slow_val)
            prev_fast = float(prev_fast_val.iloc[0] if hasattr(prev_fast_val, 'iloc') else prev_fast_val)
            prev_slow = float(prev_slow_val.iloc[0] if hasattr(prev_slow_val, 'iloc') else prev_slow_val)
            current_price = float(df['close'].iloc[-1])
            
        except Exception as e:
            logger.error("Error extracting MA values", symbol=symbol, error=str(e))
            return None
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_fast, current_slow, prev_fast, prev_slow]):
            return None
        
        # Improved trend-following logic with strong filters
        signal_action = 'HOLD'
        confidence = 0.0
        
        # Calculate comprehensive trend metrics
        # Price momentum over multiple periods
        price_change_short = (current_price - df['close'].iloc[-3]) / df['close'].iloc[-3] if len(df) >= 3 else 0
        price_change_medium = (current_price - df['close'].iloc[-8]) / df['close'].iloc[-8] if len(df) >= 8 else 0
        
        # MA slope analysis (trend direction)
        ma_slope_fast = (current_fast - fast_ma.iloc[-3]) / fast_ma.iloc[-3] if len(fast_ma) >= 3 else 0
        ma_slope_slow = (current_slow - slow_ma.iloc[-3]) / slow_ma.iloc[-3] if len(slow_ma) >= 3 else 0
        
        # MA separation and crossover strength
        ma_separation = (current_fast - current_slow) / current_slow
        ma_separation_prev = (prev_fast - prev_slow) / prev_slow
        crossover_strength = abs(ma_separation - ma_separation_prev)
        
        # Volume-weighted price change (if volume data available)
        recent_volume = df['volume'].iloc[-3:].mean() if len(df) >= 3 else 1
        avg_volume = df['volume'].mean() if len(df) >= 10 else recent_volume
        volume_confirmation = recent_volume > avg_volume * 0.8  # Volume not too low
        
        # Balanced bullish trend conditions (adjusted for 5m timeframe)
        if (current_fast > current_slow and 
            ma_separation > 0.0002 and  # At least 0.02% separation
            ma_slope_fast > 0.00005 and  # Fast MA rising (reduced threshold)
            ma_slope_slow > 0.00002 and  # Slow MA also rising (trend confirmation)
            price_change_short > 0.0008 and  # Short-term momentum (reduced)
            price_change_medium > 0.002 and  # Medium-term trend (reduced)
            crossover_strength > 0.0001):  # Meaningful crossover (reduced)
            
            signal_action = 'BUY'
            # Confidence based on multiple factors
            trend_strength = (abs(ma_separation) + abs(price_change_medium) + crossover_strength) * 200
            momentum_factor = (ma_slope_fast + ma_slope_slow) * 2000
            confidence = min(0.9, 0.5 + trend_strength + momentum_factor)
        
        # Balanced bearish trend conditions (adjusted for 5m timeframe)
        elif (current_fast < current_slow and 
              ma_separation < -0.0002 and  # At least 0.02% separation
              ma_slope_fast < -0.00005 and  # Fast MA falling (reduced threshold)
              ma_slope_slow < -0.00002 and  # Slow MA also falling
              price_change_short < -0.0008 and  # Short-term momentum (reduced)
              price_change_medium < -0.002 and  # Medium-term trend (reduced)
              crossover_strength > 0.0001):  # Meaningful crossover (reduced)
            
            signal_action = 'SELL'
            # Confidence based on multiple factors
            trend_strength = (abs(ma_separation) + abs(price_change_medium) + crossover_strength) * 200
            momentum_factor = (abs(ma_slope_fast) + abs(ma_slope_slow)) * 2000
            confidence = min(0.9, 0.5 + trend_strength + momentum_factor)
        
        # Create signal using shared signal manager (following architecture)
        if signal_action != 'HOLD':
            # Get ATR value safely
            atr_series = indicators.get('atr')
            if atr_series is not None and len(atr_series) > 0:
                try:
                    atr_val = float(atr_series.iloc[-1])
                except:
                    atr_val = current_price * 0.01  # 1% of price as fallback
            else:
                atr_val = current_price * 0.01
            
            metadata = {
                'strategy_version': '1.0',
                'fast_ma': current_fast,
                'slow_ma': current_slow,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'ma_separation': abs(current_fast - current_slow) / current_slow,
                'atr': atr_val  # For risk management
            }
            
            return self.create_signal(symbol, signal_action, current_price, confidence, metadata)
        
        return None
    
    def _prepare_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert list of dicts to DataFrame with proper types"""
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df