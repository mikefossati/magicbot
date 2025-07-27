"""
Signal management components for trading strategies.

This module provides configurable signal generation, filtering,
and deduplication logic that can be shared across strategies.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import structlog
import pandas as pd

from ..signal import Signal

logger = structlog.get_logger()

@dataclass
class SignalRule:
    """Definition of a signal generation rule"""
    name: str
    weight: float
    threshold: float
    description: str = ""

class SignalManager:
    """Manages signal generation, scoring, and filtering"""
    
    def __init__(self, strategy_name: str, params: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.params = params
        self.last_signals = {}  # Track last signals to prevent duplicates
        
        # Signal deduplication settings
        self.enable_deduplication = params.get('enable_signal_deduplication', True)
        self.min_signal_interval = params.get('min_signal_interval_minutes', 0)
        
        logger.debug("Signal manager initialized", 
                    strategy=strategy_name,
                    deduplication=self.enable_deduplication)
    
    def should_generate_signal(self, symbol: str, action: str) -> bool:
        """
        Check if a signal should be generated based on deduplication rules.
        
        Args:
            symbol: Trading symbol
            action: Signal action (BUY/SELL/HOLD)
            
        Returns:
            True if signal should be generated
        """
        if not self.enable_deduplication:
            return action != 'HOLD'
        
        # Check for duplicate signals
        last_signal = self.last_signals.get(symbol)
        if action == 'HOLD' or (last_signal and last_signal == action):
            return False
        
        return True
    
    def record_signal(self, symbol: str, action: str) -> None:
        """Record a signal for deduplication tracking"""
        self.last_signals[symbol] = action
    
    def create_signal(self, symbol: str, action: str, price: float, confidence: float,
                     metadata: Optional[Dict[str, Any]] = None) -> Signal:
        """
        Create a trading signal with proper formatting.
        
        Args:
            symbol: Trading symbol
            action: Signal action
            price: Current price
            confidence: Signal confidence (0-1)
            metadata: Additional signal metadata
            
        Returns:
            Formatted Signal object
        """
        if metadata is None:
            metadata = {}
        
        # Add strategy information to metadata
        metadata.update({
            'strategy': self.strategy_name,
            'confidence_score': confidence,
            'signal_timestamp': None  # Will be set by the system
        })
        
        # Calculate position size based on confidence and strategy parameters
        base_position_size = self.params.get('position_size', 0.01)
        
        # Apply confidence scaling if enabled
        confidence_scaling = self.params.get('confidence_position_scaling', False)
        if confidence_scaling:
            min_scale = self.params.get('min_confidence_scale', 0.5)
            position_size = base_position_size * (min_scale + (confidence * (1 - min_scale)))
        else:
            position_size = base_position_size
        
        return Signal(
            symbol=symbol,
            action=action,
            quantity=Decimal(str(position_size)),
            confidence=confidence,
            price=Decimal(str(price)),
            metadata=metadata
        )
    
    def filter_signals_by_confidence(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on minimum confidence threshold.
        
        Args:
            signals: List of signals to filter
            
        Returns:
            Filtered list of signals
        """
        min_confidence = self.params.get('min_signal_confidence', 0.0)
        if min_confidence <= 0:
            return signals
        
        filtered_signals = [s for s in signals if s.confidence >= min_confidence]
        
        if len(filtered_signals) != len(signals):
            logger.debug("Filtered signals by confidence",
                        original_count=len(signals),
                        filtered_count=len(filtered_signals),
                        min_confidence=min_confidence)
        
        return filtered_signals
    
    def rank_signals_by_confidence(self, signals: List[Signal]) -> List[Signal]:
        """
        Rank signals by confidence and apply maximum signal limits.
        
        Args:
            signals: List of signals to rank
            
        Returns:
            Ranked and limited list of signals
        """
        if not signals:
            return signals
        
        # Sort by confidence (highest first)
        ranked_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
        
        # Apply maximum signal limit
        max_signals = self.params.get('max_concurrent_signals', len(ranked_signals))
        if max_signals < len(ranked_signals):
            ranked_signals = ranked_signals[:max_signals]
            logger.debug("Limited signals by max concurrent limit",
                        total_signals=len(signals),
                        max_allowed=max_signals)
        
        return ranked_signals

class BullishSignalScorer:
    """Calculates bullish signal scores using configurable rules"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
        # Default scoring weights (can be overridden in config)
        self.trend_weight = params.get('bullish_trend_weight', 0.40)
        self.momentum_weight = params.get('bullish_momentum_weight', 0.25)
        self.volume_weight = params.get('bullish_volume_weight', 0.20)
        self.support_resistance_weight = params.get('bullish_sr_weight', 0.10)
        self.additional_weight = params.get('bullish_additional_weight', 0.05)
    
    def calculate_score(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """
        Calculate bullish signal score.
        
        Args:
            current_price: Current market price
            indicators: Dictionary of calculated indicators
            
        Returns:
            Bullish score (0-1)
        """
        score = 0.0
        
        # Trend Analysis
        score += self._score_trend_bullish(current_price, indicators) * self.trend_weight
        
        # Momentum Analysis  
        score += self._score_momentum_bullish(indicators) * self.momentum_weight
        
        # Volume Confirmation
        score += self._score_volume_bullish(indicators) * self.volume_weight
        
        # Support/Resistance Context
        score += self._score_support_resistance_bullish(current_price, indicators) * self.support_resistance_weight
        
        # Additional Factors
        score += self._score_additional_bullish(indicators) * self.additional_weight
        
        return min(score, 1.0)
    
    def _score_trend_bullish(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """Score trend alignment for bullish signals"""
        score = 0.0
        
        # EMA alignment
        if all(key in indicators for key in ['ema_fast', 'ema_medium', 'ema_slow']):
            ema_fast = float(indicators['ema_fast'].iloc[-1])
            ema_medium = float(indicators['ema_medium'].iloc[-1])
            ema_slow = float(indicators['ema_slow'].iloc[-1])
            
            if ema_fast > ema_medium > ema_slow:
                score = 1.0  # Perfect bullish alignment
            elif ema_fast > ema_medium:
                score = 0.6  # Partial alignment
            elif current_price > ema_fast:
                score = 0.4  # Price above fast EMA
        
        # MA crossover
        elif all(key in indicators for key in ['sma_fast', 'sma_slow']):
            fast_ma = float(indicators['sma_fast'].iloc[-1])
            slow_ma = float(indicators['sma_slow'].iloc[-1])
            
            if fast_ma > slow_ma:
                score = 0.8
        
        return score
    
    def _score_momentum_bullish(self, indicators: Dict[str, Any]) -> float:
        """Score momentum for bullish signals"""
        score = 0.0
        
        # RSI scoring
        if 'rsi' in indicators:
            rsi = float(indicators['rsi'].iloc[-1])
            rsi_oversold = self.params.get('rsi_oversold', 30)
            rsi_overbought = self.params.get('rsi_overbought', 70)
            rsi_neutral_low = self.params.get('rsi_neutral_low', 40)
            rsi_neutral_high = self.params.get('rsi_neutral_high', 60)
            
            if rsi_neutral_low <= rsi <= rsi_neutral_high:
                if rsi >= 50:
                    score += 0.8  # Strong bullish momentum in neutral zone
                else:
                    score += 0.6  # Good neutral momentum
            elif rsi_oversold <= rsi <= rsi_overbought:
                if rsi >= 50:
                    score += 0.4  # Moderate bullish momentum
                else:
                    score += 0.2  # Weak momentum
        
        # MACD scoring
        if all(key in indicators for key in ['macd_line', 'signal_line']):
            macd_line = float(indicators['macd_line'].iloc[-1])
            macd_signal = float(indicators['signal_line'].iloc[-1])
            
            if macd_line > macd_signal:
                score += 0.2  # MACD bullish
        
        return min(score, 1.0)
    
    def _score_volume_bullish(self, indicators: Dict[str, Any]) -> float:
        """Score volume confirmation for bullish signals"""
        if 'volume_ratio' not in indicators:
            return 0.0
        
        volume_ratio = float(indicators['volume_ratio'].iloc[-1])
        volume_multiplier = self.params.get('volume_multiplier', 1.5)
        min_volume_ratio = self.params.get('min_volume_ratio', 0.8)
        
        if volume_ratio >= volume_multiplier:
            return 1.0  # Strong volume
        elif volume_ratio >= 1.0:
            return 0.75  # Normal volume
        elif volume_ratio >= min_volume_ratio:
            return 0.25  # Weak but acceptable volume
        else:
            return 0.0  # Insufficient volume
    
    def _score_support_resistance_bullish(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """Score support/resistance context for bullish signals"""
        if not all(key in indicators for key in ['support_level', 'resistance_level']):
            return 0.5  # Neutral score if S/R not available
        
        support = indicators['support_level']
        resistance = indicators['resistance_level']
        threshold_pct = self.params.get('support_resistance_threshold', 1.0) / 100
        
        price_near_support = abs(current_price - support) / current_price <= threshold_pct
        price_near_resistance = abs(current_price - resistance) / current_price <= threshold_pct
        
        if price_near_support and not price_near_resistance:
            return 1.0  # Good entry near support
        elif current_price > resistance:
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            if volume_ratio >= 1.1:
                return 0.8  # Breakout above resistance with volume
        elif not price_near_resistance:
            return 0.5  # Not near resistance
        
        return 0.0
    
    def _score_additional_bullish(self, indicators: Dict[str, Any]) -> float:
        """Score additional factors for bullish signals"""
        score = 0.0
        
        # MACD histogram
        if 'histogram' in indicators:
            macd_hist = float(indicators['histogram'].iloc[-1])
            if macd_hist > 0:
                score += 0.6
        
        # RSI in optimal range
        if 'rsi' in indicators:
            rsi = float(indicators['rsi'].iloc[-1])
            if 45 <= rsi <= 55:
                score += 0.4
        
        return min(score, 1.0)

class BearishSignalScorer:
    """Calculates bearish signal scores using configurable rules"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
        # Default scoring weights (can be overridden in config)
        self.trend_weight = params.get('bearish_trend_weight', 0.40)
        self.momentum_weight = params.get('bearish_momentum_weight', 0.25)
        self.volume_weight = params.get('bearish_volume_weight', 0.20)
        self.support_resistance_weight = params.get('bearish_sr_weight', 0.10)
        self.additional_weight = params.get('bearish_additional_weight', 0.05)
    
    def calculate_score(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """
        Calculate bearish signal score.
        
        Args:
            current_price: Current market price
            indicators: Dictionary of calculated indicators
            
        Returns:
            Bearish score (0-1)
        """
        score = 0.0
        
        # Trend Analysis
        score += self._score_trend_bearish(current_price, indicators) * self.trend_weight
        
        # Momentum Analysis
        score += self._score_momentum_bearish(indicators) * self.momentum_weight
        
        # Volume Confirmation
        score += self._score_volume_bearish(indicators) * self.volume_weight
        
        # Support/Resistance Context
        score += self._score_support_resistance_bearish(current_price, indicators) * self.support_resistance_weight
        
        # Additional Factors
        score += self._score_additional_bearish(indicators) * self.additional_weight
        
        return min(score, 1.0)
    
    def _score_trend_bearish(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """Score trend alignment for bearish signals"""
        score = 0.0
        
        # EMA alignment
        if all(key in indicators for key in ['ema_fast', 'ema_medium', 'ema_slow']):
            ema_fast = float(indicators['ema_fast'].iloc[-1])
            ema_medium = float(indicators['ema_medium'].iloc[-1])
            ema_slow = float(indicators['ema_slow'].iloc[-1])
            
            if ema_fast < ema_medium < ema_slow:
                score = 1.0  # Perfect bearish alignment
            elif ema_fast < ema_medium:
                score = 0.6  # Partial alignment
            elif current_price < ema_fast:
                score = 0.4  # Price below fast EMA
        
        # MA crossover
        elif all(key in indicators for key in ['sma_fast', 'sma_slow']):
            fast_ma = float(indicators['sma_fast'].iloc[-1])
            slow_ma = float(indicators['sma_slow'].iloc[-1])
            
            if fast_ma < slow_ma:
                score = 0.8
        
        return score
    
    def _score_momentum_bearish(self, indicators: Dict[str, Any]) -> float:
        """Score momentum for bearish signals"""
        score = 0.0
        
        # RSI scoring
        if 'rsi' in indicators:
            rsi = float(indicators['rsi'].iloc[-1])
            rsi_oversold = self.params.get('rsi_oversold', 30)
            rsi_overbought = self.params.get('rsi_overbought', 70)
            rsi_neutral_low = self.params.get('rsi_neutral_low', 40)
            rsi_neutral_high = self.params.get('rsi_neutral_high', 60)
            
            if rsi_neutral_low <= rsi <= rsi_neutral_high:
                if rsi <= 50:
                    score += 0.8  # Strong bearish momentum in neutral zone
                else:
                    score += 0.6  # Good neutral momentum
            elif rsi_oversold <= rsi <= rsi_overbought:
                if rsi <= 50:
                    score += 0.4  # Moderate bearish momentum
                else:
                    score += 0.2  # Weak momentum
        
        # MACD scoring
        if all(key in indicators for key in ['macd_line', 'signal_line']):
            macd_line = float(indicators['macd_line'].iloc[-1])
            macd_signal = float(indicators['signal_line'].iloc[-1])
            
            if macd_line < macd_signal:
                score += 0.2  # MACD bearish
        
        return min(score, 1.0)
    
    def _score_volume_bearish(self, indicators: Dict[str, Any]) -> float:
        """Score volume confirmation for bearish signals"""
        if 'volume_ratio' not in indicators:
            return 0.0
        
        volume_ratio = float(indicators['volume_ratio'].iloc[-1])
        volume_multiplier = self.params.get('volume_multiplier', 1.5)
        min_volume_ratio = self.params.get('min_volume_ratio', 0.8)
        
        if volume_ratio >= volume_multiplier:
            return 1.0  # Strong volume
        elif volume_ratio >= 1.0:
            return 0.75  # Normal volume
        elif volume_ratio >= min_volume_ratio:
            return 0.25  # Weak but acceptable volume
        else:
            return 0.0  # Insufficient volume
    
    def _score_support_resistance_bearish(self, current_price: float, indicators: Dict[str, Any]) -> float:
        """Score support/resistance context for bearish signals"""
        if not all(key in indicators for key in ['support_level', 'resistance_level']):
            return 0.5  # Neutral score if S/R not available
        
        support = indicators['support_level']
        resistance = indicators['resistance_level']
        threshold_pct = self.params.get('support_resistance_threshold', 1.0) / 100
        
        price_near_support = abs(current_price - support) / current_price <= threshold_pct
        price_near_resistance = abs(current_price - resistance) / current_price <= threshold_pct
        
        if price_near_resistance and not price_near_support:
            return 1.0  # Good entry near resistance
        elif current_price < support:
            volume_ratio = indicators.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            if volume_ratio >= 1.1:
                return 0.8  # Breakdown below support with volume
        elif not price_near_support:
            return 0.5  # Not near support
        
        return 0.0
    
    def _score_additional_bearish(self, indicators: Dict[str, Any]) -> float:
        """Score additional factors for bearish signals"""
        score = 0.0
        
        # MACD histogram
        if 'histogram' in indicators:
            macd_hist = float(indicators['histogram'].iloc[-1])
            if macd_hist < 0:
                score += 0.6
        
        # RSI in optimal range
        if 'rsi' in indicators:
            rsi = float(indicators['rsi'].iloc[-1])
            if 45 <= rsi <= 55:
                score += 0.4
        
        return min(score, 1.0)