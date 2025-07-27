"""
Day Trading Strategy

A comprehensive day trading strategy that combines:
- EMA trend analysis (8/21/50 EMAs)
- RSI momentum confirmation
- MACD signal validation
- Volume profile analysis
- Support/resistance levels
- Risk management with dynamic stops

Designed for 5m-15m timeframes with intraday position management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class DayTradingStrategy(BaseStrategy):
    """
    Multi-indicator day trading strategy combining trend, momentum, and volume analysis
    """
    
    def __init__(self, config: Dict):
        # EMA periods for trend analysis - OPTIMIZED based on backtesting results
        self.fast_ema = config.get('fast_ema', 12)  # Increased from 8 - better trend confirmation
        self.medium_ema = config.get('medium_ema', 26)  # Increased from 21 - more stable reference
        self.slow_ema = config.get('slow_ema', 55)  # Increased from 50 - stronger trend filter
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_neutral_high = config.get('rsi_neutral_high', 60)
        self.rsi_neutral_low = config.get('rsi_neutral_low', 40)
        
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # Volume analysis - OPTIMIZED based on backtesting results
        self.volume_period = config.get('volume_period', 20)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)  # Increased from 1.2 - optimal balance
        self.min_volume_ratio = config.get('min_volume_ratio', 0.8)  # Minimum volume threshold
        
        # Support/Resistance  
        self.pivot_period = config.get('pivot_period', 10)
        self.support_resistance_threshold = config.get('support_resistance_threshold', 0.8)  # More realistic 0.8%
        
        # Signal scoring thresholds - OPTIMIZED based on backtesting results
        self.min_signal_score = config.get('min_signal_score', 0.8)  # Increased from 0.75 - optimal quality threshold
        self.strong_signal_score = config.get('strong_signal_score', 0.85)  # Maintained - higher bar for strong signals
        
        # Risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 1.5)
        self.take_profit_pct = config.get('take_profit_pct', 2.5)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
        self.max_daily_trades = config.get('max_daily_trades', 3)
        
        # Leverage settings
        self.leverage = config.get('leverage', 1.0)  # 1x = no leverage
        self.use_leverage = config.get('use_leverage', False)
        self.max_leverage = config.get('max_leverage', 10.0)  # Maximum allowed leverage
        self.leverage_risk_factor = config.get('leverage_risk_factor', 0.5)  # Reduce position size with leverage
        
        # Trading session parameters
        self.session_start = config.get('session_start', '09:30')
        self.session_end = config.get('session_end', '15:30')
        self.avoid_news_minutes = config.get('avoid_news_minutes', 30)
        
        # Strategy state
        self.daily_trades = 0
        self.last_trade_date = None
        
        # Call parent constructor after setting all parameters
        super().__init__(config)
        
        logger.info("Day trading strategy initialized", 
                   fast_ema=self.fast_ema,
                   medium_ema=self.medium_ema,
                   slow_ema=self.slow_ema,
                   max_daily_trades=self.max_daily_trades,
                   leverage=self.leverage,
                   use_leverage=self.use_leverage)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            # Check EMA periods are in ascending order
            if not (self.fast_ema < self.medium_ema < self.slow_ema):
                logger.error("EMA periods must be in ascending order")
                return False
            
            # Check RSI thresholds
            if not (0 < self.rsi_oversold < self.rsi_neutral_low < self.rsi_neutral_high < self.rsi_overbought < 100):
                logger.error("Invalid RSI threshold values")
                return False
            
            # Check MACD parameters
            if not (self.macd_fast < self.macd_slow):
                logger.error("MACD fast period must be less than slow period")
                return False
            
            # Check risk management parameters
            if self.stop_loss_pct <= 0 or self.take_profit_pct <= 0:
                logger.error("Stop loss and take profit must be positive")
                return False
            
            if self.max_daily_trades <= 0:
                logger.error("Max daily trades must be positive")
                return False
            
            # Check leverage parameters
            if self.use_leverage:
                if self.leverage <= 0 or self.leverage > self.max_leverage:
                    logger.error("Leverage must be between 0 and max_leverage", 
                               leverage=self.leverage, max_leverage=self.max_leverage)
                    return False
                
                if self.leverage_risk_factor <= 0 or self.leverage_risk_factor > 1:
                    logger.error("Leverage risk factor must be between 0 and 1")
                    return False
            
            logger.info("Day trading strategy parameters validated successfully")
            return True
            
        except Exception as e:
            logger.error("Parameter validation failed", error=str(e))
            return False
    
    async def generate_signals(self, market_data) -> List[Signal]:
        """Generate signals for all symbols"""
        signals = []
        
        for symbol, data in market_data.items():
            if symbol in self.symbols:
                # Convert data format if needed (backtest engine provides list of dicts)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Convert from list of dictionaries to DataFrame
                    df_data = pd.DataFrame(data)
                    if 'timestamp' in df_data.columns:
                        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='ms')
                        df_data.set_index('timestamp', inplace=True)
                    processed_data = df_data
                elif isinstance(data, pd.DataFrame):
                    # Data is already in DataFrame format
                    processed_data = data
                else:
                    # Skip if data format is not recognized
                    logger.warning("Unrecognized data format", symbol=symbol, data_type=type(data))
                    continue
                
                signal = await self.analyze_market_data(symbol, processed_data)
                if signal:
                    signals.append(signal)
        
        return signals

    async def analyze_market_data(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals
        """
        min_required = max([self.slow_ema, self.volume_period, self.pivot_period]) + 10
        if len(data) < min_required:
            logger.warning("Insufficient data for day trading analysis",
                         data_points=len(data),
                         required=min_required,
                         symbol=symbol)
            return None
        
        try:
            # Check if we're in trading session
            last_timestamp = data.index[-1]
            if hasattr(last_timestamp, 'time'):
                current_time = last_timestamp.time()
            else:
                current_time = pd.Timestamp(last_timestamp).time()
            
            if not self._is_trading_session(current_time):
                logger.debug("Outside trading session", current_time=current_time, 
                           session_start=self.session_start, session_end=self.session_end)
                return None
            
            # Check daily trade limit
            if hasattr(last_timestamp, 'date'):
                current_date = last_timestamp.date()
            else:
                current_date = pd.Timestamp(last_timestamp).date()
            if not self._check_daily_limit(current_date):
                return None
                
            # Calculate all indicators
            indicators = self._calculate_indicators(data)
            
            # Generate signal based on multiple confirmations
            signal = self._generate_signal(symbol, data, indicators)
            
            return signal
            
        except Exception as e:
            logger.error("Error in day trading analysis", symbol=symbol, error=str(e))
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators with proper error handling"""
        indicators = {}
        
        try:
            # Price data
            closes = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            # Validate data
            if len(closes) < self.slow_ema:
                raise ValueError(f"Insufficient data for EMA calculation: {len(closes)} < {self.slow_ema}")
            
            # EMAs for trend analysis
            indicators['ema_fast'] = closes.ewm(span=self.fast_ema, adjust=False).mean()
            indicators['ema_medium'] = closes.ewm(span=self.medium_ema, adjust=False).mean()
            indicators['ema_slow'] = closes.ewm(span=self.slow_ema, adjust=False).mean()
            
            # RSI for momentum
            indicators['rsi'] = self._calculate_rsi(closes, self.rsi_period)
            
            # MACD for trend confirmation
            macd_data = self._calculate_macd(closes)
            indicators['macd_line'] = macd_data['macd_line']
            indicators['macd_signal'] = macd_data['signal_line']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # Volume analysis
            indicators['volume_avg'] = volumes.rolling(window=self.volume_period, min_periods=1).mean()
            # Handle division by zero
            volume_avg = indicators['volume_avg']
            indicators['volume_ratio'] = volumes / volume_avg.replace(0, np.nan)
            
            # Support and resistance levels
            sr_levels = self._calculate_support_resistance(highs, lows, closes)
            indicators['support_level'] = sr_levels['support']
            indicators['resistance_level'] = sr_levels['resistance']
            
            # ATR for volatility-based stops
            indicators['atr'] = self._calculate_atr(highs, lows, closes, period=14)
            
            # Validate all indicators are calculated
            for key, value in indicators.items():
                if value is None:
                    raise ValueError(f"Failed to calculate indicator: {key}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with proper error handling"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Handle division by zero
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill any remaining NaN values with neutral RSI
            rsi = rsi.fillna(50)
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            # Return neutral RSI series
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicators with error handling"""
        try:
            ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            # Return zero series
            zero_series = pd.Series([0] * len(prices), index=prices.index)
            return {
                'macd_line': zero_series,
                'signal_line': zero_series,
                'histogram': zero_series
            }
    
    def _calculate_support_resistance(self, highs: pd.Series, lows: pd.Series, closes: pd.Series) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels"""
        try:
            recent_highs = highs.rolling(window=self.pivot_period, min_periods=1).max()
            recent_lows = lows.rolling(window=self.pivot_period, min_periods=1).min()
            
            # Use recent pivot points for S/R levels and ensure they're scalar
            resistance = float(recent_highs.iloc[-1])
            support = float(recent_lows.iloc[-1])
            
            # Validate the values
            if pd.isna(resistance) or pd.isna(support):
                current_price = float(closes.iloc[-1])
                resistance = current_price * 1.01  # 1% above current price
                support = current_price * 0.99     # 1% below current price
            
            return {
                'resistance': resistance,
                'support': support
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            # Fallback to current price +/- 1%
            current_price = float(closes.iloc[-1])
            return {
                'resistance': current_price * 1.01,
                'support': current_price * 0.99
            }
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range with error handling"""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            # Fill any NaN values with a reasonable default (0.5% of price)
            price_based_atr = closes * 0.005
            atr = atr.fillna(price_based_atr)
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            # Return 0.5% of price as fallback ATR
            return closes * 0.005
    
    def _calculate_leveraged_position_size(self, current_price: float, confidence: float) -> float:
        """Calculate position size adjusted for leverage and risk"""
        base_position_size = float(self.position_size)  # Convert Decimal to float
        
        if self.use_leverage and self.leverage > 1:
            # Reduce base position size when using leverage to manage risk
            adjusted_position_size = base_position_size * float(self.leverage_risk_factor)
            
            # Further adjust based on confidence - higher confidence allows larger positions
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
            final_position_size = adjusted_position_size * confidence_multiplier
            
            logger.debug("Leveraged position sizing", 
                        base_size=base_position_size,
                        leverage=self.leverage,
                        risk_factor=self.leverage_risk_factor,
                        confidence=confidence,
                        final_size=final_position_size)
            
            return final_position_size
        else:
            # No leverage - use base position size
            return base_position_size
    
    def _calculate_leveraged_stops(self, current_price: float, action: str, atr: float) -> Dict[str, float]:
        """Calculate stop loss and take profit levels adjusted for leverage"""
        if self.use_leverage and self.leverage > 1:
            # Tighter stops with leverage to manage risk
            leverage_adjustment = 1.0 / (1.0 + (self.leverage - 1) * 0.3)  # Reduce stops by 30% per leverage unit
            
            adjusted_stop_pct = self.stop_loss_pct * leverage_adjustment
            adjusted_profit_pct = self.take_profit_pct * leverage_adjustment
            
            # Also use ATR-based dynamic stops (more conservative with leverage)
            atr_stop_multiplier = 1.5 if self.leverage <= 3 else 1.0  # Tighter stops for higher leverage
            atr_based_stop = atr * atr_stop_multiplier
            
            if action == 'BUY':
                # Use the more conservative (closer) stop
                percentage_stop = current_price * (adjusted_stop_pct / 100)
                stop_loss = current_price - min(percentage_stop, atr_based_stop)
                take_profit = current_price + (current_price * adjusted_profit_pct / 100)
            else:  # SELL
                percentage_stop = current_price * (adjusted_stop_pct / 100)
                stop_loss = current_price + min(percentage_stop, atr_based_stop)
                take_profit = current_price - (current_price * adjusted_profit_pct / 100)
            
            logger.debug("Leveraged stops calculated", 
                        leverage=self.leverage,
                        original_stop_pct=self.stop_loss_pct,
                        adjusted_stop_pct=adjusted_stop_pct,
                        atr_stop=atr_based_stop,
                        final_stop=stop_loss,
                        take_profit=take_profit)
            
        else:
            # No leverage - use standard calculations
            if action == 'BUY':
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (current_price * self.take_profit_pct / 100)
            else:  # SELL
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (current_price * self.take_profit_pct / 100)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _generate_signal(self, symbol: str, data: pd.DataFrame, indicators: Dict) -> Optional[Signal]:
        """Generate trading signal using realistic scoring system"""
        
        try:
            current_price = float(data['close'].iloc[-1])
            
            # Get latest indicator values with proper error handling
            ema_fast = float(indicators['ema_fast'].iloc[-1])
            ema_medium = float(indicators['ema_medium'].iloc[-1])
            ema_slow = float(indicators['ema_slow'].iloc[-1])
            
            rsi = float(indicators['rsi'].iloc[-1])
            if pd.isna(rsi):
                return None
            
            macd_line = float(indicators['macd_line'].iloc[-1])
            macd_signal = float(indicators['macd_signal'].iloc[-1])
            macd_hist = float(indicators['macd_histogram'].iloc[-1])
            
            # ALWAYS use volume indicator as requested
            volume_ratio = float(indicators['volume_ratio'].iloc[-1])
            if pd.isna(volume_ratio):
                logger.debug("Volume ratio is NaN, skipping signal")
                return None
            if volume_ratio < self.min_volume_ratio:
                logger.debug("Insufficient volume for signal", volume_ratio=volume_ratio, min_required=self.min_volume_ratio)
                return None
            
            support = float(indicators['support_level'])
            resistance = float(indicators['resistance_level'])
            
            atr = float(indicators['atr'].iloc[-1])
            if pd.isna(atr) or atr <= 0:
                return None
                
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error accessing indicator values: {e}", symbol=symbol)
            return None
        
        # Calculate signal scores for different types
        bullish_score = self._calculate_bullish_score(current_price, ema_fast, ema_medium, ema_slow, 
                                                     rsi, macd_line, macd_signal, macd_hist,
                                                     volume_ratio, support, resistance)
        
        bearish_score = self._calculate_bearish_score(current_price, ema_fast, ema_medium, ema_slow,
                                                     rsi, macd_line, macd_signal, macd_hist, 
                                                     volume_ratio, support, resistance)
        
        
        # Debug logging for signal generation
        logger.debug("Signal scoring results", 
                    symbol=symbol,
                    bullish_score=bullish_score,
                    bearish_score=bearish_score,
                    min_signal_score=self.min_signal_score,
                    volume_ratio=volume_ratio,
                    current_price=current_price)
        
        # Determine signal type and strength
        if bullish_score >= self.min_signal_score and bullish_score > bearish_score:
            logger.info("Generating BUY signal", symbol=symbol, score=bullish_score)
            return self._create_signal(symbol, current_price, 'BUY', bullish_score, atr, 
                                     rsi, macd_hist, volume_ratio, support, resistance)
        
        elif bearish_score >= self.min_signal_score and bearish_score > bullish_score:
            logger.info("Generating SELL signal", symbol=symbol, score=bearish_score)
            return self._create_signal(symbol, current_price, 'SELL', bearish_score, atr,
                                     rsi, macd_hist, volume_ratio, support, resistance)
        
        logger.debug("No signal generated", symbol=symbol, reason="scores below threshold or equal")
        return None
    
    def _calculate_bullish_score(self, current_price: float, ema_fast: float, ema_medium: float, ema_slow: float,
                                rsi: float, macd_line: float, macd_signal: float, macd_hist: float,
                                volume_ratio: float, support: float, resistance: float) -> float:
        """Calculate bullish signal score using realistic criteria"""
        score = 0.0
        
        # 1. Trend Analysis (40% weight - most important)
        if ema_fast > ema_medium > ema_slow:
            score += 0.40  # Strong bullish alignment
        elif ema_fast > ema_medium:
            score += 0.25  # Partial bullish alignment
        elif current_price > ema_fast:
            score += 0.15  # Price above fast EMA
        
        # 2. Momentum Analysis (25% weight) - OPTIMIZED for better entries
        if 40 <= rsi <= 60:  # TIGHTENED: Only neutral RSI range, avoid overbought/oversold
            if rsi >= 50:
                score += 0.20  # Strong bullish momentum in neutral zone
            else:
                score += 0.15  # Good neutral momentum
        elif 30 <= rsi <= 70:  # Backup range but lower score
            if rsi >= 50:
                score += 0.10  # Reduced score for wider range
            else:
                score += 0.05  # Minimal score
        
        if macd_line > macd_signal:
            score += 0.10  # MACD bullish
        
        # 3. Volume Confirmation (20% weight - ALWAYS included)
        if volume_ratio >= self.volume_multiplier:  # 1.2x average
            score += 0.20  # Strong volume
        elif volume_ratio >= 1.0:
            score += 0.15  # Normal volume
        elif volume_ratio >= self.min_volume_ratio:  # 0.8x minimum
            score += 0.05  # Weak but acceptable volume
        
        # 4. Support/Resistance Context (10% weight)
        price_near_support = abs(current_price - support) / current_price <= (self.support_resistance_threshold / 100)
        price_near_resistance = abs(current_price - resistance) / current_price <= (self.support_resistance_threshold / 100)
        
        if price_near_support and not price_near_resistance:
            score += 0.10  # Good entry near support
        elif current_price > resistance and volume_ratio >= 1.1:
            score += 0.08  # Breakout above resistance with volume
        elif not price_near_resistance:
            score += 0.05  # Not near resistance
        
        # 5. Additional Factors (5% weight)
        if macd_hist > 0:
            score += 0.03  # MACD histogram positive
        if 45 <= rsi <= 55:  # OPTIMIZED: Prefer neutral RSI for additional scoring
            score += 0.03  # RSI in optimal neutral range (increased from 0.02)
        
        return min(score, 1.0)
    
    def _calculate_bearish_score(self, current_price: float, ema_fast: float, ema_medium: float, ema_slow: float,
                                rsi: float, macd_line: float, macd_signal: float, macd_hist: float,
                                volume_ratio: float, support: float, resistance: float) -> float:
        """Calculate bearish signal score using realistic criteria"""
        score = 0.0
        
        # 1. Trend Analysis (40% weight)
        if ema_fast < ema_medium < ema_slow:
            score += 0.40  # Strong bearish alignment
        elif ema_fast < ema_medium:
            score += 0.25  # Partial bearish alignment
        elif current_price < ema_fast:
            score += 0.15  # Price below fast EMA
        
        # 2. Momentum Analysis (25% weight) - OPTIMIZED for better entries
        if 40 <= rsi <= 60:  # TIGHTENED: Only neutral RSI range, avoid overbought/oversold
            if rsi <= 50:
                score += 0.20  # Strong bearish momentum in neutral zone
            else:
                score += 0.15  # Good neutral momentum
        elif 30 <= rsi <= 70:  # Backup range but lower score
            if rsi <= 50:
                score += 0.10  # Reduced score for wider range
            else:
                score += 0.05  # Minimal score
        
        if macd_line < macd_signal:
            score += 0.10  # MACD bearish
        
        # 3. Volume Confirmation (20% weight - ALWAYS included)
        if volume_ratio >= self.volume_multiplier:  # 1.2x average
            score += 0.20  # Strong volume
        elif volume_ratio >= 1.0:
            score += 0.15  # Normal volume
        elif volume_ratio >= self.min_volume_ratio:  # 0.8x minimum
            score += 0.05  # Weak but acceptable volume
        
        # 4. Support/Resistance Context (10% weight)
        price_near_support = abs(current_price - support) / current_price <= (self.support_resistance_threshold / 100)
        price_near_resistance = abs(current_price - resistance) / current_price <= (self.support_resistance_threshold / 100)
        
        if price_near_resistance and not price_near_support:
            score += 0.10  # Good entry near resistance
        elif current_price < support and volume_ratio >= 1.1:
            score += 0.08  # Breakdown below support with volume
        elif not price_near_support:
            score += 0.05  # Not near support
        
        # 5. Additional Factors (5% weight)
        if macd_hist < 0:
            score += 0.03  # MACD histogram negative
        if 45 <= rsi <= 55:  # OPTIMIZED: Prefer neutral RSI for additional scoring
            score += 0.03  # RSI in optimal neutral range (increased from 0.02)
        
        return min(score, 1.0)
    
    def _create_signal(self, symbol: str, current_price: float, action: str, confidence: float, atr: float,
                      rsi: float, macd_hist: float, volume_ratio: float, support: float, resistance: float) -> Signal:
        """Create a trading signal with proper metadata"""
        
        # Calculate leveraged position size and stops
        position_size = self._calculate_leveraged_position_size(current_price, confidence)
        stops = self._calculate_leveraged_stops(current_price, action, atr)
        
        # Determine signal type based on score
        signal_type = "strong" if confidence >= self.strong_signal_score else "normal"
        
        signal = Signal(
            symbol=symbol,
            action=action,
            price=current_price,
            quantity=position_size,
            confidence=confidence,
            metadata={
                'strategy': 'day_trading_v2',
                'signal_type': signal_type,
                'trend': action.lower(),
                'stop_loss': stops['stop_loss'],
                'take_profit': stops['take_profit'],
                'rsi': rsi,
                'macd_histogram': macd_hist,
                'volume_ratio': volume_ratio,
                'support_level': support,
                'resistance_level': resistance,
                'atr': atr,
                'leverage': self.leverage if self.use_leverage else 1.0,
                'leveraged_position': self.use_leverage,
                'base_position_size': self.position_size,
                'actual_position_size': position_size,
                'signal_score': confidence
            }
        )
        
        logger.info(f"Day trading {action} signal generated",
                   symbol=symbol,
                   price=current_price,
                   confidence=confidence,
                   signal_type=signal_type,
                   position_size=position_size,
                   volume_ratio=volume_ratio,
                   rsi=rsi)
        
        return signal
    
    def _calculate_signal_confidence(self, factors: Dict, action: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.0
        
        # Trend alignment (30%)
        if factors.get('trend_alignment'):
            confidence += 0.30
        
        # Momentum strength (25%)
        if factors.get('momentum_strength'):
            confidence += 0.25
        
        # Volume confirmation (20%)
        if factors.get('volume_support'):
            confidence += 0.20
        
        # Support/Resistance proximity (15%)
        if factors.get('support_level') or factors.get('resistance_level'):
            confidence += 0.15
        
        # RSI strength (5%)
        rsi = factors.get('rsi_level', 50)
        if action == 'BUY' and 40 <= rsi <= 60:
            confidence += 0.05
        elif action == 'SELL' and 40 <= rsi <= 60:
            confidence += 0.05
        
        # MACD momentum (5%)
        if factors.get('macd_momentum'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _is_trading_session(self, current_time) -> bool:
        """Check if current time is within trading session"""
        try:
            session_start = time.fromisoformat(self.session_start)
            session_end = time.fromisoformat(self.session_end)
            
            # Convert current_time to time object if needed
            if isinstance(current_time, time):
                time_obj = current_time
            else:
                time_obj = pd.Timestamp(current_time).time()
            
            is_in_session = session_start <= time_obj <= session_end
            logger.debug("Trading session check", 
                        current_time=time_obj, 
                        session_start=session_start, 
                        session_end=session_end,
                        in_session=is_in_session)
            return is_in_session
        except Exception as e:
            # If time parsing fails, assume we're in trading session (for tests)
            logger.debug("Trading session check failed, defaulting to True", error=str(e))
            return True
    
    def _check_daily_limit(self, current_date) -> bool:
        """Check if daily trade limit has been reached"""
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
        
        return self.daily_trades < self.max_daily_trades
    
    def on_trade_executed(self, trade_info: Dict):
        """Called when a trade is executed"""
        self.daily_trades += 1
        logger.info("Day trade executed", 
                   daily_trades=self.daily_trades,
                   max_trades=self.max_daily_trades)
    
    def get_risk_parameters(self) -> Dict:
        """Get risk management parameters"""
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'trailing_stop_pct': self.trailing_stop_pct,
            'max_daily_trades': self.max_daily_trades,
            'position_size': self.position_size,
            'max_position_size': self.position_size,  # For compatibility with tests
            'leverage': self.leverage,
            'use_leverage': self.use_leverage,
            'max_leverage': self.max_leverage,
            'leverage_risk_factor': self.leverage_risk_factor
        }
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'Day Trading Strategy V2',
            'description': 'Realistic day trading strategy using probability scoring system with multiple signal types: trend-following, breakout, and mean-reversion. Always includes volume confirmation.',
            'type': 'Multi-Signal Day Trading with Scoring',
            'timeframe': '5m-15m',
            'indicators': ['EMA', 'RSI', 'MACD', 'Volume', 'Support/Resistance'],
            'signal_types': ['Trend Following', 'Breakout', 'Support/Resistance Bounce'],
            'parameters': {
                'fast_ema': self.fast_ema,
                'medium_ema': self.medium_ema,
                'slow_ema': self.slow_ema,
                'rsi_period': self.rsi_period,
                'volume_multiplier': self.volume_multiplier,
                'min_volume_ratio': self.min_volume_ratio,
                'support_resistance_threshold': self.support_resistance_threshold,
                'min_signal_score': self.min_signal_score,
                'strong_signal_score': self.strong_signal_score,
                'max_daily_trades': self.max_daily_trades
            },
            'scoring_weights': {
                'trend_analysis': '40%',
                'momentum_analysis': '25%', 
                'volume_confirmation': '20%',
                'support_resistance': '10%',
                'additional_factors': '5%'
            },
            'risk_profile': 'Medium' if not self.use_leverage else 'High',
            'risk_management': 'Dynamic ATR-based stops with leverage support',
            'volume_requirement': 'Always required - minimum 0.8x average volume',
            'session_management': True,
            'daily_trade_limit': self.max_daily_trades,
            'leverage_enabled': self.use_leverage,
            'leverage_ratio': self.leverage if self.use_leverage else 1.0,
            'leverage_risk_management': 'Position size reduction and tighter stops'
        }