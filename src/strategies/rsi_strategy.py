import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
import structlog

from .base import BaseStrategy
from .signal import Signal

logger = structlog.get_logger()

class RSIStrategy(BaseStrategy):
    """RSI (Relative Strength Index) Trading Strategy using new architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('rsi_strategy', config)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on RSI using new architecture"""
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
        """Analyze a single symbol for RSI signals using new architecture"""
        # Get parameters from configuration
        rsi_period = self.get_parameter_value('rsi_period')
        oversold_threshold = self.get_parameter_value('rsi_oversold')
        overbought_threshold = self.get_parameter_value('rsi_overbought')
        
        if len(data) < rsi_period + 1:
            logger.warning("Insufficient data for RSI analysis", 
                         symbol=symbol, 
                         data_points=len(data),
                         required=rsi_period + 1)
            return None
        
        try:
            # Prepare data
            df = pd.DataFrame(data)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            # Calculate indicators using shared component
            indicators = self.calculate_indicators(df)
            
            if len(df) < 2:
                return None
            
            # Get RSI values
            current_rsi = float(indicators['rsi'].iloc[-1])
            prev_rsi = float(indicators['rsi'].iloc[-2])
            current_price = float(df['close'].iloc[-1])
            
            if pd.isna(current_rsi) or pd.isna(prev_rsi):
                return None
            
            # Determine signal type and confidence
            signal_action = 'HOLD'
            confidence = 0.0
            
            # Buy signal: RSI is oversold and starting to recover
            if current_rsi <= oversold_threshold and current_rsi > prev_rsi:
                signal_action = 'BUY'
                confidence = 0.6
                
                # Increase confidence if RSI was deeply oversold
                if current_rsi < 20:
                    confidence += 0.2
                
                # Increase confidence if recovery is strong
                rsi_momentum = current_rsi - prev_rsi
                if rsi_momentum > 2:
                    confidence += 0.1
            
            # Sell signal: RSI is overbought and starting to decline
            elif current_rsi >= overbought_threshold and current_rsi < prev_rsi:
                signal_action = 'SELL'
                confidence = 0.6
                
                # Increase confidence if RSI was deeply overbought
                if current_rsi > 80:
                    confidence += 0.2
                
                # Increase confidence if decline is strong
                rsi_momentum = prev_rsi - current_rsi
                if rsi_momentum > 2:
                    confidence += 0.1
            
            # Create signal using shared signal manager
            if signal_action != 'HOLD':
                metadata = {
                    'rsi': current_rsi,
                    'prev_rsi': prev_rsi,
                    'rsi_period': rsi_period,
                    'oversold_threshold': oversold_threshold,
                    'overbought_threshold': overbought_threshold,
                    'rsi_momentum': current_rsi - prev_rsi if signal_action == 'BUY' else prev_rsi - current_rsi
                }
                
                signal = self.create_signal(symbol, signal_action, current_price, confidence, metadata)
                
                if signal:
                    logger.info("RSI signal generated",
                               strategy=self.strategy_name,
                               symbol=symbol,
                               action=signal_action,
                               confidence=confidence,
                               rsi=current_rsi,
                               prev_rsi=prev_rsi)
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error("Error in RSI analysis", symbol=symbol, error=str(e))
            return None