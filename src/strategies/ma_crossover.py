import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog
import asyncio

from .base import BaseStrategy
from .signal import Signal

logger = structlog.get_logger()

class MovingAverageCrossover(BaseStrategy):
    """
    Enhanced Moving Average Crossover Strategy with:
    - Longer MA periods to reduce whipsaws
    - Volume and momentum confirmation filters
    - Dynamic position sizing based on volatility (ATR)
    - Better exit rules with trailing stops and profit targets
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('ma_crossover', config)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on MA crossover using new architecture"""
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
    
    def _prepare_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert and prepare data for analysis"""
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df

    async def _analyze_symbol(self, symbol: str, data: List[Dict]) -> Optional[Signal]:
        """Enhanced analysis with multiple filters and dynamic position sizing"""
        # Get parameters from configuration
        fast_period = self.get_parameter_value('fast_period', 20)  # Increased from typical 10
        slow_period = self.get_parameter_value('slow_period', 50)  # Increased from typical 30
        
        # New confirmation parameters
        volume_confirmation = self.get_parameter_value('volume_confirmation', True)
        volume_period = self.get_parameter_value('volume_period', 20)
        volume_multiplier = self.get_parameter_value('volume_multiplier', 1.5)
        
        momentum_confirmation = self.get_parameter_value('momentum_confirmation', True)
        momentum_period = self.get_parameter_value('momentum_period', 14)
        min_momentum_threshold = self.get_parameter_value('min_momentum_threshold', 0.5)
        
        # Volatility and exit parameters
        atr_period = self.get_parameter_value('atr_period', 14)
        volatility_multiplier = self.get_parameter_value('volatility_multiplier', 2.0)
        trend_strength_threshold = self.get_parameter_value('trend_strength_threshold', 0.001)
        ma_separation_threshold = self.get_parameter_value('ma_separation_threshold', 0.002)
        
        # Check data sufficiency
        required_data = max(slow_period, volume_period, momentum_period, atr_period) + 5
        if len(data) < required_data:
            logger.warning("Insufficient data for enhanced analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=required_data)
            return None
        
        try:
            # Prepare data
            df = self._prepare_data(data)
            
            # Calculate all indicators
            indicators = self._calculate_enhanced_indicators(df, fast_period, slow_period, 
                                                           volume_period, momentum_period, atr_period)
            
            if len(df) < 3:  # Need at least 3 periods for trend analysis
                return None
            
            # Get current and previous values
            current_fast = float(indicators['sma_fast'].iloc[-1])
            current_slow = float(indicators['sma_slow'].iloc[-1])
            prev_fast = float(indicators['sma_fast'].iloc[-2])
            prev_slow = float(indicators['sma_slow'].iloc[-2])
            current_price = float(df['close'].iloc[-1])
            
            # Get additional indicators
            current_volume_ratio = float(indicators['volume_ratio'].iloc[-1])
            current_momentum = float(indicators['momentum'].iloc[-1])
            current_atr = float(indicators['atr'].iloc[-1])
            
            # Check for NaN values
            if any(pd.isna(val) for val in [current_fast, current_slow, prev_fast, prev_slow, 
                                          current_volume_ratio, current_momentum, current_atr]):
                return None
            
            # 1. Check for crossover (basic signal)
            signal_action = 'HOLD'
            base_confidence = 0.0
            crossover_strength = 0.0
            
            # Bullish crossover: fast MA crosses above slow MA
            if prev_fast <= prev_slow and current_fast > current_slow:
                signal_action = 'BUY'
                base_confidence = 0.6
                crossover_strength = (current_fast - current_slow) / current_slow
            
            # Bearish crossover: fast MA crosses below slow MA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                signal_action = 'SELL'
                base_confidence = 0.6
                crossover_strength = (current_slow - current_fast) / current_slow
            
            if signal_action == 'HOLD':
                return None
            
            # 2. Apply enhanced filters
            confidence_adjustments = []
            filter_results = {}
            
            # A. Trend Strength Filter
            ma_separation = abs(current_fast - current_slow) / current_price
            if ma_separation >= ma_separation_threshold:
                confidence_adjustments.append(0.1)
                filter_results['strong_trend'] = True
            else:
                confidence_adjustments.append(-0.1)
                filter_results['strong_trend'] = False
            
            # B. Volume Confirmation Filter
            if volume_confirmation:
                if current_volume_ratio >= volume_multiplier:
                    confidence_adjustments.append(0.15)
                    filter_results['volume_confirmed'] = True
                else:
                    confidence_adjustments.append(-0.15)
                    filter_results['volume_confirmed'] = False
                    # Reject weak volume signals
                    if current_volume_ratio < 0.8:
                        logger.debug("Signal rejected due to low volume", 
                                   symbol=symbol, volume_ratio=current_volume_ratio)
                        return None
            
            # C. Momentum Confirmation Filter
            if momentum_confirmation:
                momentum_aligned = (signal_action == 'BUY' and current_momentum > min_momentum_threshold) or \
                                 (signal_action == 'SELL' and current_momentum < -min_momentum_threshold)
                
                if momentum_aligned:
                    confidence_adjustments.append(0.1)
                    filter_results['momentum_confirmed'] = True
                else:
                    confidence_adjustments.append(-0.1)
                    filter_results['momentum_confirmed'] = False
                    # Reject momentum divergence
                    if abs(current_momentum) < min_momentum_threshold / 2:
                        logger.debug("Signal rejected due to weak momentum", 
                                   symbol=symbol, momentum=current_momentum)
                        return None
            
            # 3. Calculate final confidence
            final_confidence = base_confidence + sum(confidence_adjustments)
            final_confidence = max(0.1, min(1.0, final_confidence))
            
            # Minimum confidence threshold
            if final_confidence < 0.6:
                logger.debug("Signal rejected due to low confidence", 
                           symbol=symbol, confidence=final_confidence)
                return None
            
            # 4. Dynamic position sizing based on volatility
            volatility_adjusted_size = self._calculate_volatility_position_size(
                current_price, current_atr, volatility_multiplier
            )
            
            # 5. Calculate dynamic exit levels
            exit_levels = self._calculate_dynamic_exits(
                signal_action, current_price, current_atr
            )
            
            # 6. Create enhanced signal
            metadata = {
                'strategy_version': 'enhanced_v2',
                'fast_ma': current_fast,
                'slow_ma': current_slow,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'crossover_strength': crossover_strength,
                'ma_separation': ma_separation,
                'volume_ratio': current_volume_ratio,
                'momentum': current_momentum,
                'atr': current_atr,
                'volatility_adjusted_size': volatility_adjusted_size,
                'base_confidence': base_confidence,
                'confidence_adjustments': confidence_adjustments,
                'filter_results': filter_results,
                **exit_levels
            }
            
            # Override default position size with volatility-adjusted size
            signal = self.create_signal(symbol, signal_action, current_price, final_confidence, metadata)
            if signal:
                # Update quantity with volatility-adjusted size
                signal.quantity = Decimal(str(volatility_adjusted_size))
                
                logger.info("Enhanced MA crossover signal generated",
                           strategy=self.strategy_name,
                           symbol=symbol,
                           action=signal_action,
                           confidence=final_confidence,
                           fast_ma=current_fast,
                           slow_ma=current_slow,
                           volume_ratio=current_volume_ratio,
                           momentum=current_momentum,
                           volatility_size=volatility_adjusted_size,
                           filters_passed=filter_results)
            
            return signal
            
        except Exception as e:
            logger.error("Error in enhanced MA crossover analysis", symbol=symbol, error=str(e))
            return None
    
    def _calculate_enhanced_indicators(self, df: pd.DataFrame, fast_period: int, slow_period: int,
                                     volume_period: int, momentum_period: int, atr_period: int) -> Dict:
        """Calculate all indicators needed for enhanced analysis"""
        indicators = {}
        
        # Moving averages
        indicators['sma_fast'] = df['close'].rolling(window=fast_period).mean()
        indicators['sma_slow'] = df['close'].rolling(window=slow_period).mean()
        
        # Volume analysis
        indicators['volume_avg'] = df['volume'].rolling(window=volume_period).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_avg']
        
        # Momentum (rate of change)
        indicators['momentum'] = df['close'].pct_change(momentum_period) * 100
        
        # ATR (Average True Range) for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(window=atr_period).mean()
        
        return indicators
    
    def _calculate_volatility_position_size(self, current_price: float, atr: float, 
                                          volatility_multiplier: float) -> float:
        """
        Calculate position size based on volatility (ATR)
        Lower volatility = larger position size
        Higher volatility = smaller position size
        """
        base_position_size = self.get_parameter_value('position_size', 0.01)
        
        # Calculate volatility as percentage of price
        volatility_pct = (atr / current_price) * 100
        
        # Adjust position size inversely to volatility
        # If volatility is 2%, and multiplier is 2.0, then adjustment = 2.0/2.0 = 1.0 (no change)
        # If volatility is 4%, adjustment = 2.0/4.0 = 0.5 (half position)
        # If volatility is 1%, adjustment = 2.0/1.0 = 2.0 (double position, but capped)
        volatility_adjustment = volatility_multiplier / max(volatility_pct, 0.5)  # Min 0.5% to avoid huge positions
        volatility_adjustment = min(volatility_adjustment, 3.0)  # Cap at 3x
        volatility_adjustment = max(volatility_adjustment, 0.1)  # Min 0.1x
        
        adjusted_size = base_position_size * volatility_adjustment
        
        logger.debug("Volatility-based position sizing",
                    base_size=base_position_size,
                    volatility_pct=volatility_pct,
                    adjustment=volatility_adjustment,
                    final_size=adjusted_size)
        
        return adjusted_size
    
    def _calculate_dynamic_exits(self, signal_action: str, current_price: float, atr: float) -> Dict:
        """Calculate dynamic stop loss and profit target based on ATR"""
        trailing_stop_multiplier = self.get_parameter_value('trailing_stop_multiplier', 2.0)
        profit_target_multiplier = self.get_parameter_value('profit_target_multiplier', 3.0)
        
        if signal_action == 'BUY':
            stop_loss = current_price - (atr * trailing_stop_multiplier)
            profit_target = current_price + (atr * profit_target_multiplier)
        else:  # SELL
            stop_loss = current_price + (atr * trailing_stop_multiplier)
            profit_target = current_price - (atr * profit_target_multiplier)
        
        return {
            'dynamic_stop_loss': stop_loss,
            'dynamic_profit_target': profit_target,
            'atr_stop_distance': atr * trailing_stop_multiplier,
            'atr_profit_distance': atr * profit_target_multiplier,
            'risk_reward_ratio': profit_target_multiplier / trailing_stop_multiplier
        }