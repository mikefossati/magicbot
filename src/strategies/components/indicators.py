"""
Shared technical indicator calculations for trading strategies.

This module provides centralized, optimized implementations of common
technical indicators to eliminate code duplication across strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import structlog

logger = structlog.get_logger()

class IndicatorCalculator:
    """Centralized calculator for technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series (typically closing prices)
            period: RSI calculation period
            
        Returns:
            RSI values as pandas Series
        """
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
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices: Price series
            period: Moving average period
            
        Returns:
            SMA values as pandas Series
        """
        return prices.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices: Price series
            period: EMA period
            
        Returns:
            EMA values as pandas Series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicators.
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with macd_line, signal_line, and histogram
        """
        try:
            ema_fast = IndicatorCalculator.calculate_ema(prices, fast)
            ema_slow = IndicatorCalculator.calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = IndicatorCalculator.calculate_ema(macd_line, signal)
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
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        try:
            middle_band = IndicatorCalculator.calculate_sma(prices, period)
            std = prices.rolling(window=period, min_periods=1).std()
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            # Return price as middle band with minimal spread
            return {
                'upper': prices * 1.01,
                'middle': prices,
                'lower': prices * 0.99
            }
    
    @staticmethod
    def calculate_stochastic(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                           k_period: int = 14, d_period: int = 3, smooth_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            highs: High price series
            lows: Low price series
            closes: Close price series
            k_period: %K period
            d_period: %D smoothing period
            smooth_period: Additional smoothing period
            
        Returns:
            Dictionary with %K and %D lines
        """
        try:
            lowest_low = lows.rolling(window=k_period, min_periods=1).min()
            highest_high = highs.rolling(window=k_period, min_periods=1).max()
            
            # Handle division by zero
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, np.nan)
            
            k_percent = 100 * ((closes - lowest_low) / price_range)
            k_percent = k_percent.fillna(50)  # Neutral value for NaN
            
            # Smooth %K
            if smooth_period > 1:
                k_percent = k_percent.rolling(window=smooth_period, min_periods=1).mean()
            
            # Calculate %D
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            # Return neutral values
            neutral_series = pd.Series([50] * len(closes), index=closes.index)
            return {
                'k_percent': neutral_series,
                'd_percent': neutral_series
            }
    
    @staticmethod
    def calculate_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            highs: High price series
            lows: Low price series
            closes: Close price series
            period: ATR period
            
        Returns:
            ATR values as pandas Series
        """
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
    
    @staticmethod
    def calculate_volume_indicators(volumes: pd.Series, prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Args:
            volumes: Volume series
            prices: Price series
            period: Calculation period
            
        Returns:
            Dictionary with volume indicators
        """
        try:
            # Volume moving average
            volume_ma = volumes.rolling(window=period, min_periods=1).mean()
            
            # Volume ratio (current volume / average volume)
            volume_ratio = volumes / volume_ma.replace(0, np.nan)
            volume_ratio = volume_ratio.fillna(1.0)
            
            # On-Balance Volume (OBV)
            price_change = prices.diff()
            obv_change = np.where(price_change > 0, volumes, 
                                np.where(price_change < 0, -volumes, 0))
            obv = pd.Series(obv_change, index=volumes.index).cumsum()
            
            return {
                'volume_ma': volume_ma,
                'volume_ratio': volume_ratio,
                'obv': obv
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            # Return minimal indicators
            return {
                'volume_ma': volumes,
                'volume_ratio': pd.Series([1.0] * len(volumes), index=volumes.index),
                'obv': volumes.cumsum()
            }
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            prices: Price series
            period: Momentum period
            
        Returns:
            Momentum values as pandas Series
        """
        try:
            return ((prices / prices.shift(period)) - 1) * 100
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return pd.Series([0] * len(prices), index=prices.index)
    
    @staticmethod
    def calculate_support_resistance(highs: pd.Series, lows: pd.Series, closes: pd.Series, 
                                   period: int = 10) -> Dict[str, float]:
        """
        Calculate dynamic support and resistance levels.
        
        Args:
            highs: High price series
            lows: Low price series
            closes: Close price series
            period: Lookback period
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            recent_highs = highs.rolling(window=period, min_periods=1).max()
            recent_lows = lows.rolling(window=period, min_periods=1).min()
            
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
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all indicators needed for a strategy based on parameters.
        
        Args:
            data: OHLCV DataFrame
            params: Strategy parameters
            
        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {}
        
        try:
            closes = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            # RSI
            if 'rsi_period' in params:
                indicators['rsi'] = IndicatorCalculator.calculate_rsi(closes, params['rsi_period'])
            
            # Moving averages
            if 'fast_period' in params:
                indicators['sma_fast'] = IndicatorCalculator.calculate_sma(closes, params['fast_period'])
            if 'slow_period' in params:
                indicators['sma_slow'] = IndicatorCalculator.calculate_sma(closes, params['slow_period'])
            
            # EMAs for day trading
            if 'fast_ema' in params:
                indicators['ema_fast'] = IndicatorCalculator.calculate_ema(closes, params['fast_ema'])
            if 'medium_ema' in params:
                indicators['ema_medium'] = IndicatorCalculator.calculate_ema(closes, params['medium_ema'])
            if 'slow_ema' in params:
                indicators['ema_slow'] = IndicatorCalculator.calculate_ema(closes, params['slow_ema'])
            
            # MACD
            if all(param in params for param in ['macd_fast', 'macd_slow', 'macd_signal']):
                macd_data = IndicatorCalculator.calculate_macd(
                    closes, params['macd_fast'], params['macd_slow'], params['macd_signal']
                )
                indicators.update(macd_data)
            
            # Bollinger Bands
            if 'period' in params and 'std_dev' in params:
                bb_data = IndicatorCalculator.calculate_bollinger_bands(
                    closes, params['period'], params['std_dev']
                )
                indicators['bb_upper'] = bb_data['upper']
                indicators['bb_middle'] = bb_data['middle']
                indicators['bb_lower'] = bb_data['lower']
            
            # Stochastic
            if all(param in params for param in ['k_period', 'd_period']):
                stoch_data = IndicatorCalculator.calculate_stochastic(
                    highs, lows, closes, 
                    params['k_period'], params['d_period'], 
                    params.get('smooth_period', 3)
                )
                indicators['stoch_k'] = stoch_data['k_percent']
                indicators['stoch_d'] = stoch_data['d_percent']
            
            # ATR
            indicators['atr'] = IndicatorCalculator.calculate_atr(highs, lows, closes)
            
            # Volume indicators
            if 'volume_period' in params:
                volume_data = IndicatorCalculator.calculate_volume_indicators(
                    volumes, closes, params['volume_period']
                )
                indicators.update(volume_data)
            
            # Support/Resistance
            if 'pivot_period' in params:
                sr_data = IndicatorCalculator.calculate_support_resistance(
                    highs, lows, closes, params['pivot_period']
                )
                indicators['support_level'] = sr_data['support']
                indicators['resistance_level'] = sr_data['resistance']
            
            # Momentum
            if 'momentum_period' in params:
                indicators['momentum'] = IndicatorCalculator.calculate_momentum(
                    closes, params['momentum_period']
                )
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise