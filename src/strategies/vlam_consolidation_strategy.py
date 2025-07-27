"""
VLAM Consolidation Breakout Strategy

A sophisticated strategy that combines:
- Volatility and Liquidity Adjusted Momentum (VLAM) indicator
- Horizontal consolidation detection
- Spike and reversion entry methodology
- Dynamic stop loss and target management

Strategy Logic:
1. Identify clean horizontal consolidation zones
2. Wait for directional spike (breakout attempt)
3. Wait for VLAM signal confirming reversion back into consolidation
4. Enter on signal candle close with tight risk management
5. Target opposite side of consolidation range
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog

from .base import BaseStrategy, Signal

logger = structlog.get_logger()

class VLAMConsolidationStrategy(BaseStrategy):
    """
    VLAM Consolidation Breakout Strategy using Volatility and Liquidity Adjusted Momentum
    """
    
    def __init__(self, config: Dict):
        # VLAM Indicator Parameters
        self.vlam_period = config.get('vlam_period', 10)
        self.atr_period = config.get('atr_period', 10)
        self.volume_period = config.get('volume_period', 15)
        
        # Consolidation Detection Parameters
        self.consolidation_min_length = config.get('consolidation_min_length', 4)  # Minimum bars in consolidation
        self.consolidation_max_length = config.get('consolidation_max_length', 20)  # Maximum bars to look back
        self.consolidation_tolerance = config.get('consolidation_tolerance', 0.05)  # 5% tolerance for "horizontal" (optimized)
        self.min_touches = config.get('min_touches', 2)  # Minimum touches of support/resistance
        
        # Spike Detection Parameters
        self.spike_min_size = config.get('spike_min_size', 0.8)  # Minimum spike size as multiple of ATR (optimized)
        self.spike_volume_multiplier = config.get('spike_volume_multiplier', 1.5)  # Volume confirmation for spike (optimized)
        
        # Entry Signal Parameters
        self.vlam_signal_threshold = config.get('vlam_signal_threshold', 0.5)  # VLAM signal strength threshold (optimized)
        self.entry_timeout_bars = config.get('entry_timeout_bars', 12)  # Max bars to wait for entry after spike (optimized)
        
        # Risk Management Parameters
        self.stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 1.5)
        self.target_risk_reward = config.get('target_risk_reward', 3.0)  # Risk:Reward ratio (optimized)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)  # 2% max risk per trade
        
        # Position Management
        self.max_concurrent_positions = config.get('max_concurrent_positions', 2)
        self.position_timeout_hours = config.get('position_timeout_hours', 24)  # Max position hold time
        
        # Strategy State
        self.active_consolidations = {}  # Track consolidation zones per symbol
        self.spike_events = {}  # Track recent spike events
        self.pending_entries = {}  # Track pending entry setups
        
        # Call parent constructor
        super().__init__(config)
        
        logger.info("VLAM Consolidation strategy initialized", 
                   vlam_period=self.vlam_period,
                   consolidation_tolerance=self.consolidation_tolerance,
                   spike_min_size=self.spike_min_size,
                   max_risk=self.max_risk_per_trade)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        try:
            # Check VLAM parameters
            if self.vlam_period <= 0 or self.atr_period <= 0:
                logger.error("VLAM and ATR periods must be positive")
                return False
            
            # Check consolidation parameters
            if self.consolidation_min_length >= self.consolidation_max_length:
                logger.error("Consolidation min length must be less than max length")
                return False
            
            if not (0 < self.consolidation_tolerance < 0.1):
                logger.error("Consolidation tolerance must be between 0 and 10%")
                return False
            
            # Check spike parameters
            if self.spike_min_size <= 0:
                logger.error("Spike minimum size must be positive")
                return False
            
            # Check risk parameters
            if not (0 < self.max_risk_per_trade <= 0.1):
                logger.error("Max risk per trade must be between 0 and 10%")
                return False
            
            if self.target_risk_reward <= 1.0:
                logger.error("Target risk:reward ratio must be greater than 1.0")
                return False
            
            logger.info("VLAM Consolidation strategy parameters validated successfully")
            return True
            
        except Exception as e:
            logger.error("Parameter validation failed", error=str(e))
            return False
    
    async def generate_signals(self, market_data) -> List[Signal]:
        """Generate signals for all symbols"""
        signals = []
        
        for symbol, data in market_data.items():
            if symbol in self.symbols:
                # Convert data format if needed
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    df_data = pd.DataFrame(data)
                    if 'timestamp' in df_data.columns:
                        df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='ms')
                        df_data.set_index('timestamp', inplace=True)
                    processed_data = df_data
                elif isinstance(data, pd.DataFrame):
                    processed_data = data
                else:
                    logger.warning("Unrecognized data format", symbol=symbol, data_type=type(data))
                    continue
                
                signal = await self.analyze_market_data(symbol, processed_data)
                if signal:
                    signals.append(signal)
        
        return signals

    async def analyze_market_data(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals based on VLAM consolidation methodology
        """
        min_required = max([self.consolidation_max_length, self.atr_period, self.volume_period]) + 10
        if len(data) < min_required:
            logger.warning("Insufficient data for VLAM analysis",
                         data_points=len(data),
                         required=min_required,
                         symbol=symbol)
            return None
        
        try:
            # Calculate all indicators
            indicators = self._calculate_indicators(data)
            
            # Step 1: Identify consolidation zones
            consolidation = self._detect_consolidation(data, indicators)
            if not consolidation:
                logger.debug("No valid consolidation detected", symbol=symbol)
                return None
            
            # Step 2: Check for recent spike
            spike_event = self._detect_spike(data, indicators, consolidation)
            if not spike_event:
                logger.debug("No valid spike detected", symbol=symbol)
                return None
            
            # Step 3: Check for VLAM entry signal
            entry_signal = self._check_vlam_entry_signal(data, indicators, consolidation, spike_event)
            if not entry_signal:
                logger.debug("No VLAM entry signal", symbol=symbol)
                return None
            
            # Step 4: Generate trading signal
            signal = self._create_vlam_signal(symbol, data, indicators, consolidation, spike_event, entry_signal)
            
            return signal
            
        except Exception as e:
            logger.error("Error in VLAM analysis", symbol=symbol, error=str(e))
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate VLAM and supporting indicators"""
        indicators = {}
        
        try:
            # Price data
            opens = data['open']
            highs = data['high']
            lows = data['low']
            closes = data['close']
            volumes = data['volume']
            
            # Calculate ATR for volatility component
            indicators['atr'] = self._calculate_atr(highs, lows, closes, self.atr_period)
            
            # Calculate volume ratio for liquidity component
            volume_avg = volumes.rolling(window=self.volume_period, min_periods=1).mean()
            indicators['volume_ratio'] = volumes / volume_avg.replace(0, np.nan)
            
            # Calculate VLAM (Volatility and Liquidity Adjusted Momentum)
            indicators['vlam'] = self._calculate_vlam(opens, highs, lows, closes, volumes, indicators['atr'], indicators['volume_ratio'])
            
            # Additional technical indicators
            indicators['sma_20'] = closes.rolling(window=20, min_periods=1).mean()
            indicators['sma_50'] = closes.rolling(window=50, min_periods=1).mean()
            
            # Price action indicators
            indicators['high_low_ratio'] = (highs - lows) / closes
            indicators['body_ratio'] = abs(closes - opens) / (highs - lows).replace(0, np.nan)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def _calculate_vlam(self, opens: pd.Series, highs: pd.Series, lows: pd.Series, 
                       closes: pd.Series, volumes: pd.Series, atr: pd.Series, volume_ratio: pd.Series) -> pd.Series:
        """
        Calculate Volatility and Liquidity Adjusted Momentum using modified Heiken Ashi formula
        
        This indicator combines:
        - Heiken Ashi momentum calculation
        - ATR for volatility adjustment
        - Volume ratio for liquidity adjustment
        """
        try:
            # Initialize VLAM series
            vlam = pd.Series(index=closes.index, dtype=float)
            
            # Calculate Heiken Ashi components with volume and ATR adjustments
            ha_close = (opens + highs + lows + closes) / 4
            ha_open = pd.Series(index=closes.index, dtype=float)
            ha_high = pd.Series(index=closes.index, dtype=float)
            ha_low = pd.Series(index=closes.index, dtype=float)
            
            # Initialize first values
            ha_open.iloc[0] = (opens.iloc[0] + closes.iloc[0]) / 2
            
            for i in range(1, len(closes)):
                # Traditional Heiken Ashi calculation
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
                ha_high.iloc[i] = max(highs.iloc[i], ha_open.iloc[i], ha_close.iloc[i])
                ha_low.iloc[i] = min(lows.iloc[i], ha_open.iloc[i], ha_close.iloc[i])
                
                # VLAM calculation with volatility and liquidity adjustments
                momentum = ha_close.iloc[i] - ha_open.iloc[i]
                
                # Volatility adjustment (normalize by ATR)
                volatility_adj = momentum / max(atr.iloc[i], 0.0001)
                
                # Liquidity adjustment (weight by volume ratio)
                liquidity_weight = min(volume_ratio.iloc[i], 3.0)  # Cap at 3x for stability
                
                # Combined VLAM value
                vlam.iloc[i] = volatility_adj * liquidity_weight
            
            # Smooth the VLAM signal
            vlam_smoothed = vlam.rolling(window=3, min_periods=1).mean()
            
            # Normalize to -1 to +1 range using rolling statistics
            vlam_normalized = self._normalize_series(vlam_smoothed, window=self.vlam_period)
            
            return vlam_normalized.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating VLAM: {e}")
            return pd.Series([0] * len(closes), index=closes.index)
    
    def _normalize_series(self, series: pd.Series, window: int) -> pd.Series:
        """Normalize series to -1 to +1 range using rolling statistics"""
        try:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            
            # Z-score normalization
            z_score = (series - rolling_mean) / rolling_std.replace(0, 1)
            
            # Apply tanh for bounded normalization
            normalized = np.tanh(z_score)
            
            return normalized
        except:
            return series
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            # Fill any NaN values
            atr = atr.fillna(closes * 0.02)  # 2% of price as fallback
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return closes * 0.02
    
    def _detect_consolidation(self, data: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Detect horizontal consolidation zones
        
        Returns consolidation info or None if no valid consolidation found
        """
        try:
            highs = data['high']
            lows = data['low']
            closes = data['close']
            
            # Look for consolidation in recent bars
            for length in range(self.consolidation_min_length, min(self.consolidation_max_length, len(data))):
                recent_highs = highs.iloc[-length:]
                recent_lows = lows.iloc[-length:]
                recent_closes = closes.iloc[-length:]
                
                # Calculate consolidation range
                range_high = recent_highs.max()
                range_low = recent_lows.min()
                range_size = range_high - range_low
                range_mid = (range_high + range_low) / 2
                
                # Check if range is horizontal (low volatility)
                tolerance = range_mid * self.consolidation_tolerance
                if range_size > tolerance:
                    continue  # Range too wide, not a consolidation
                
                # Count touches of support and resistance levels
                support_touches = (recent_lows <= (range_low + tolerance * 0.5)).sum()
                resistance_touches = (recent_highs >= (range_high - tolerance * 0.5)).sum()
                
                # Check for minimum touches
                if (support_touches + resistance_touches) < self.min_touches:
                    continue
                
                # Check that price stayed mostly within range
                breakout_bars = ((recent_highs > range_high + tolerance) | 
                               (recent_lows < range_low - tolerance)).sum()
                if breakout_bars > length * 0.2:  # Max 20% breakout bars
                    continue
                
                # Valid consolidation found
                consolidation = {
                    'length': length,
                    'high': range_high,
                    'low': range_low,
                    'mid': range_mid,
                    'size': range_size,
                    'support_touches': support_touches,
                    'resistance_touches': resistance_touches,
                    'start_index': len(data) - length,
                    'end_index': len(data) - 1
                }
                
                logger.debug("Consolidation detected", 
                           length=length, 
                           range_size=range_size,
                           support_touches=support_touches,
                           resistance_touches=resistance_touches)
                
                return consolidation
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting consolidation: {e}")
            return None
    
    def _detect_spike(self, data: pd.DataFrame, indicators: Dict, consolidation: Dict) -> Optional[Dict]:
        """
        Detect directional spike around consolidation height
        
        Returns spike info or None if no valid spike found
        """
        try:
            highs = data['high']
            lows = data['low']
            closes = data['close']
            volumes = data['volume']
            atr = indicators['atr']
            volume_ratio = indicators['volume_ratio']
            
            # Look for spikes in recent bars (expand search to find spikes anywhere)
            # For now, search more broadly to find spikes that break consolidation
            search_start = max(0, len(data) - 15)  # Look at last 15 bars
            search_end = len(data)
            
            logger.debug(f"Searching for spikes from bar {search_start} to {search_end-1}")
            
            # Search through bars for spikes
            for i in range(search_start, search_end):
                current_bar = data.iloc[i]
                current_atr = atr.iloc[i]
                current_vol_ratio = volume_ratio.iloc[i]
                
                # Check for upward spike (high breaking above consolidation)
                upward_spike = (current_bar['high'] - consolidation['high']) / current_atr
                # Check for downward spike (low breaking below consolidation) 
                downward_spike = (consolidation['low'] - current_bar['low']) / current_atr
                
                logger.debug(f"Bar {i}: Up spike={upward_spike:.2f}, Down spike={downward_spike:.2f}, Vol={current_vol_ratio:.2f}")
                
                # Determine spike direction and strength
                spike_direction = None
                spike_strength = 0
                
                # Check if upward spike meets minimum size requirement
                if upward_spike >= self.spike_min_size:
                    spike_direction = 'up'
                    spike_strength = upward_spike
                    logger.debug(f"Potential upward spike detected: strength={spike_strength:.2f}")
                
                # Check if downward spike meets minimum size requirement
                elif downward_spike >= self.spike_min_size:
                    spike_direction = 'down'
                    spike_strength = downward_spike
                    logger.debug(f"Potential downward spike detected: strength={spike_strength:.2f}")
                
                if spike_direction:
                    # Check for volume confirmation
                    volume_confirmed = current_vol_ratio >= self.spike_volume_multiplier
                    logger.debug(f"Volume confirmed: {volume_confirmed} (ratio={current_vol_ratio:.2f}, required={self.spike_volume_multiplier})")
                    
                    # Additional validation: spike should originate from within or near consolidation
                    spike_valid = True  # Simplified for now - the break above/below is the key signal
                    
                    if spike_valid and volume_confirmed:
                        spike_event = {
                            'direction': spike_direction,
                            'strength': spike_strength,
                            'bar_index': i,
                            'high': current_bar['high'],
                            'low': current_bar['low'],
                            'close': current_bar['close'],
                            'volume_ratio': current_vol_ratio,
                            'bars_since': len(data) - 1 - i
                        }
                        
                        logger.debug("Spike detected", 
                                   direction=spike_direction,
                                   strength=spike_strength,
                                   bars_since=spike_event['bars_since'])
                        
                        return spike_event
                    else:
                        logger.debug(f"Spike rejected: valid={spike_valid}, vol_confirmed={volume_confirmed}")
            
            logger.debug("No valid spike found in search range")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting spike: {e}")
            return None
    
    def _check_vlam_entry_signal(self, data: pd.DataFrame, indicators: Dict, 
                                consolidation: Dict, spike_event: Dict) -> Optional[Dict]:
        """
        Check for VLAM entry signal confirming reversion back into consolidation
        """
        try:
            vlam = indicators['vlam']
            current_vlam = vlam.iloc[-1]
            previous_vlam = vlam.iloc[-2] if len(vlam) > 1 else 0
            
            # Check if we're within entry timeout period
            if spike_event['bars_since'] > self.entry_timeout_bars:
                logger.debug("Spike event too old for entry")
                return None
            
            # Determine expected VLAM signal direction based on spike
            # After upward spike, we want bearish VLAM signal (reversion down into consolidation)
            # After downward spike, we want bullish VLAM signal (reversion up into consolidation)
            
            expected_direction = 'bearish' if spike_event['direction'] == 'up' else 'bullish'
            
            # Check VLAM signal strength and direction
            signal_strength = abs(current_vlam)
            signal_direction = 'bullish' if current_vlam > 0 else 'bearish'
            
            # Confirm signal direction matches expectation
            if signal_direction != expected_direction:
                logger.debug("VLAM signal direction mismatch", 
                           expected=expected_direction, 
                           actual=signal_direction)
                return None
            
            # Check signal strength meets threshold
            if signal_strength < self.vlam_signal_threshold:
                logger.debug("VLAM signal too weak", 
                           strength=signal_strength, 
                           threshold=self.vlam_signal_threshold)
                return None
            
            # Check for signal momentum (increasing strength)
            signal_momentum = abs(current_vlam) > abs(previous_vlam)
            
            # Validate current price position for reversion trade
            current_price = data['close'].iloc[-1]
            
            if expected_direction == 'bearish':
                # After upward spike, we want to enter short as price reverts down
                entry_action = 'SELL'
                # Price should be within reasonable range of consolidation (allow reversion in progress)
                price_valid = current_price >= consolidation['low'] * 0.99  # Within 1% below consolidation
            else:
                # After downward spike, we want to enter long as price reverts up
                entry_action = 'BUY'
                # Price should be within reasonable range of consolidation (allow reversion in progress)
                price_valid = current_price <= consolidation['high'] * 1.01  # Within 1% above consolidation
            
            if not price_valid:
                logger.debug("Price position invalid for entry", 
                           price=current_price, 
                           consolidation_range=f"{consolidation['low']:.0f}-{consolidation['high']:.0f}",
                           action=entry_action)
                return None
            
            entry_signal = {
                'action': entry_action,
                'direction': expected_direction,
                'strength': signal_strength,
                'momentum': signal_momentum,
                'vlam_value': current_vlam,
                'price_valid': price_valid
            }
            
            logger.info("VLAM entry signal confirmed", 
                       action=entry_action,
                       strength=signal_strength,
                       vlam_value=current_vlam)
            
            return entry_signal
            
        except Exception as e:
            logger.error(f"Error checking VLAM entry signal: {e}")
            return None
    
    def _create_vlam_signal(self, symbol: str, data: pd.DataFrame, indicators: Dict,
                           consolidation: Dict, spike_event: Dict, entry_signal: Dict) -> Signal:
        """Create trading signal with proper risk management"""
        
        try:
            current_price = float(data['close'].iloc[-1])
            current_atr = float(indicators['atr'].iloc[-1])
            action = entry_signal['action']
            
            # Calculate stop loss based on current leg's high/low
            if action == 'SELL':
                # Stop loss at current leg's high (spike high)
                stop_loss = spike_event['high']
                # Target at consolidation low
                target = consolidation['low']
            else:  # BUY
                # Stop loss at current leg's low (spike low)
                stop_loss = spike_event['low']
                # Target at consolidation high
                target = consolidation['high']
            
            # Calculate risk and position size
            risk_amount = abs(current_price - stop_loss)
            target_amount = abs(target - current_price)
            
            # Ensure minimum risk:reward ratio
            if target_amount / risk_amount < self.target_risk_reward:
                # Adjust target to meet risk:reward requirement
                if action == 'SELL':
                    target = current_price - (risk_amount * self.target_risk_reward)
                else:
                    target = current_price + (risk_amount * self.target_risk_reward)
            
            # Calculate position size based on max risk per trade
            account_value = 10000  # This should come from account management
            max_risk_amount = account_value * self.max_risk_per_trade
            position_size = min(max_risk_amount / risk_amount, self.position_size)
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=position_size,
                confidence=entry_signal['strength'],
                metadata={
                    'strategy': 'vlam_consolidation',
                    'signal_type': 'consolidation_reversion',
                    'stop_loss': stop_loss,
                    'take_profit': target,
                    'risk_amount': risk_amount,
                    'target_amount': abs(target - current_price),
                    'risk_reward_ratio': abs(target - current_price) / risk_amount,
                    'consolidation': {
                        'high': consolidation['high'],
                        'low': consolidation['low'],
                        'length': consolidation['length']
                    },
                    'spike_event': {
                        'direction': spike_event['direction'],
                        'strength': spike_event['strength'],
                        'bars_since': spike_event['bars_since']
                    },
                    'vlam_signal': {
                        'value': entry_signal['vlam_value'],
                        'strength': entry_signal['strength'],
                        'direction': entry_signal['direction']
                    },
                    'atr': current_atr,
                    'position_timeout': self.position_timeout_hours
                }
            )
            
            logger.info(f"VLAM consolidation {action} signal created",
                       symbol=symbol,
                       price=current_price,
                       stop_loss=stop_loss,
                       target=target,
                       risk_reward=abs(target - current_price) / risk_amount,
                       confidence=entry_signal['strength'])
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating VLAM signal: {e}")
            raise
    
    def get_risk_parameters(self) -> Dict:
        """Get risk management parameters"""
        return {
            'max_risk_per_trade': self.max_risk_per_trade,
            'target_risk_reward': self.target_risk_reward,
            'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
            'max_concurrent_positions': self.max_concurrent_positions,
            'position_timeout_hours': self.position_timeout_hours,
            'position_size': self.position_size
        }
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'VLAM Consolidation Breakout Strategy',
            'description': 'Volatility and Liquidity Adjusted Momentum strategy for consolidation breakout reversions',
            'type': 'Mean Reversion / Consolidation Breakout',
            'timeframe': '5m-1h',
            'indicators': ['VLAM', 'ATR', 'Volume Ratio', 'Consolidation Detection'],
            'signal_types': ['Consolidation Reversion', 'Spike Fade'],
            'parameters': {
                'vlam_period': self.vlam_period,
                'consolidation_tolerance': self.consolidation_tolerance,
                'spike_min_size': self.spike_min_size,
                'vlam_signal_threshold': self.vlam_signal_threshold,
                'target_risk_reward': self.target_risk_reward,
                'max_risk_per_trade': self.max_risk_per_trade
            },
            'methodology': {
                'step_1': 'Identify horizontal consolidation zones with minimum touches',
                'step_2': 'Detect directional spikes with volume confirmation', 
                'step_3': 'Wait for VLAM signal confirming reversion into consolidation',
                'step_4': 'Enter on signal with stops at spike extreme, target opposite range'
            },
            'risk_profile': 'Medium-High',
            'risk_management': 'Dynamic stops based on spike levels, R:R minimum 2:1',
            'market_conditions': 'Best in ranging/consolidating markets with clear levels'
        }