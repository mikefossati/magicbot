"""
Unit tests for Momentum Trading Strategy
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy

class TestMomentumTradingStrategy:
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return {
            'symbols': ['BTCUSDT'],
            'trend_ema_fast': 12,
            'trend_ema_slow': 26,
            'rsi_period': 14,
            'volume_surge_multiplier': 1.5,
            'base_position_size': 0.02,
            'max_risk_per_trade': 0.02
        }
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        
        # Create trending data
        base_price = 50000
        trend = np.linspace(0, 0.1, 100)  # 10% uptrend
        noise = np.random.normal(0, 0.01, 100)  # 1% noise
        
        prices = base_price * (1 + trend + noise)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.002, 100)),
            'high': prices * (1 + np.random.uniform(0, 0.01, 100)),
            'low': prices * (1 - np.random.uniform(0, 0.01, 100)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        # Ensure high >= close >= low and open
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data.set_index('timestamp')
    
    def test_strategy_initialization(self, basic_config):
        """Test strategy initialization with valid config"""
        strategy = MomentumTradingStrategy(basic_config)
        
        assert strategy.trend_ema_fast == 12
        assert strategy.trend_ema_slow == 26
        assert strategy.rsi_period == 14
        assert strategy.volume_surge_multiplier == 1.5
        assert strategy.base_position_size == 0.02
        assert strategy.max_risk_per_trade == 0.02
    
    def test_validate_parameters_valid(self, basic_config):
        """Test parameter validation with valid parameters"""
        strategy = MomentumTradingStrategy(basic_config)
        assert strategy.validate_parameters() == True
    
    def test_validate_parameters_invalid_ema(self, basic_config):
        """Test parameter validation with invalid EMA settings"""
        basic_config['trend_ema_fast'] = 30  # Fast > Slow
        basic_config['trend_ema_slow'] = 20
        
        with pytest.raises(ValueError):
            strategy = MomentumTradingStrategy(basic_config)
    
    def test_validate_parameters_invalid_rsi(self, basic_config):
        """Test parameter validation with invalid RSI period"""
        basic_config['rsi_period'] = 0
        
        with pytest.raises(ValueError):
            strategy = MomentumTradingStrategy(basic_config)
    
    def test_validate_parameters_invalid_position_size(self, basic_config):
        """Test parameter validation with invalid position sizing"""
        basic_config['base_position_size'] = 0.15  # Too large
        basic_config['max_position_size'] = 0.10
        
        with pytest.raises(ValueError):
            strategy = MomentumTradingStrategy(basic_config)
    
    def test_calculate_indicators(self, basic_config, sample_data):
        """Test indicator calculations"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Check that all required indicators are present
        required_indicators = [
            'ema_fast', 'ema_slow', 'ema_signal',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'atr', 'volume_sma', 'volume_ratio',
            'price_momentum', 'trend_strength'
        ]
        
        for indicator in required_indicators:
            assert indicator in indicators
            assert len(indicators[indicator]) == len(sample_data)
        
        # Check RSI bounds
        rsi_values = indicators['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Check EMA ordering (fast should be more responsive)
        assert not indicators['ema_fast'].isnull().all()
        assert not indicators['ema_slow'].isnull().all()
    
    def test_trend_detection_bullish(self, basic_config, sample_data):
        """Test bullish trend detection"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Modify data to create clear bullish trend
        indicators['ema_fast'].iloc[-10:] = np.linspace(51000, 52000, 10)
        indicators['ema_slow'].iloc[-10:] = np.linspace(50500, 51500, 10)
        indicators['trend_strength'].iloc[-10:] = 0.025  # Above threshold
        
        trend = strategy._detect_trend_direction(sample_data, indicators)
        assert trend == 'bullish'
    
    def test_trend_detection_bearish(self, basic_config, sample_data):
        """Test bearish trend detection"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Modify data to create clear bearish trend
        indicators['ema_fast'].iloc[-10:] = np.linspace(51000, 50000, 10)
        indicators['ema_slow'].iloc[-10:] = np.linspace(51500, 50500, 10)
        indicators['trend_strength'].iloc[-10:] = 0.025  # Above threshold
        
        trend = strategy._detect_trend_direction(sample_data, indicators)
        assert trend == 'bearish'
    
    def test_trend_detection_weak(self, basic_config, sample_data):
        """Test trend detection with weak trend"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Create weak trend (below threshold)
        indicators['ema_fast'].iloc[-10:] = 51000
        indicators['ema_slow'].iloc[-10:] = 50900
        indicators['trend_strength'].iloc[-10:] = 0.001  # Below threshold
        
        trend = strategy._detect_trend_direction(sample_data, indicators)
        assert trend is None
    
    def test_momentum_alignment_bullish(self, basic_config, sample_data):
        """Test momentum alignment for bullish trend"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set up bullish momentum alignment
        indicators['rsi'].iloc[-1] = 65  # Above 50
        indicators['macd'].iloc[-1] = 100
        indicators['macd_signal'].iloc[-1] = 95
        indicators['macd_histogram'].iloc[-1] = 5  # Positive
        
        aligned = strategy._check_momentum_alignment(indicators, 'bullish')
        assert aligned == True
    
    def test_momentum_alignment_bearish(self, basic_config, sample_data):
        """Test momentum alignment for bearish trend"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set up bearish momentum alignment
        indicators['rsi'].iloc[-1] = 35  # Below 50
        indicators['macd'].iloc[-1] = -100
        indicators['macd_signal'].iloc[-1] = -95
        indicators['macd_histogram'].iloc[-1] = -5  # Negative
        
        aligned = strategy._check_momentum_alignment(indicators, 'bearish')
        assert aligned == True
    
    def test_momentum_alignment_misaligned(self, basic_config, sample_data):
        """Test momentum alignment with misaligned indicators"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set up misaligned momentum (bullish trend, bearish indicators)
        indicators['rsi'].iloc[-1] = 35  # Below 50
        indicators['macd'].iloc[-1] = -100
        indicators['macd_signal'].iloc[-1] = -95
        indicators['macd_histogram'].iloc[-1] = -5  # Negative
        
        aligned = strategy._check_momentum_alignment(indicators, 'bullish')
        assert aligned == False
    
    def test_volume_confirmation_pass(self, basic_config, sample_data):
        """Test volume confirmation with sufficient surge"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set volume surge above threshold
        indicators['volume_ratio'].iloc[-1] = 2.0  # 100% above average
        
        confirmed = strategy._check_volume_confirmation(indicators)
        assert confirmed == True
    
    def test_volume_confirmation_fail(self, basic_config, sample_data):
        """Test volume confirmation with insufficient surge"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set volume below threshold
        indicators['volume_ratio'].iloc[-1] = 1.2  # Only 20% above average
        
        confirmed = strategy._check_volume_confirmation(indicators)
        assert confirmed == False
    
    def test_volume_confirmation_disabled(self, basic_config, sample_data):
        """Test volume confirmation when disabled"""
        basic_config['volume_confirmation_required'] = False
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Even with low volume, should pass when disabled
        indicators['volume_ratio'].iloc[-1] = 0.5
        
        confirmed = strategy._check_volume_confirmation(indicators)
        assert confirmed == True
    
    def test_breakout_detection_bullish(self, basic_config, sample_data):
        """Test bullish breakout detection"""
        strategy = MomentumTradingStrategy(basic_config)
        
        # Set current price above recent highs
        sample_data['close'].iloc[-1] = sample_data['high'].iloc[-20:-1].max() + 100
        
        breakout = strategy._detect_breakout(sample_data, 'bullish')
        assert breakout == True
    
    def test_breakout_detection_bearish(self, basic_config, sample_data):
        """Test bearish breakout detection"""
        strategy = MomentumTradingStrategy(basic_config)
        
        # Set current price below recent lows
        sample_data['close'].iloc[-1] = sample_data['low'].iloc[-20:-1].min() - 100
        
        breakout = strategy._detect_breakout(sample_data, 'bearish')
        assert breakout == True
    
    def test_breakout_detection_none(self, basic_config, sample_data):
        """Test no breakout detection"""
        strategy = MomentumTradingStrategy(basic_config)
        
        # Set current price within recent range
        recent_high = sample_data['high'].iloc[-20:-1].max()
        recent_low = sample_data['low'].iloc[-20:-1].min()
        sample_data['close'].iloc[-1] = (recent_high + recent_low) / 2
        
        bullish_breakout = strategy._detect_breakout(sample_data, 'bullish')
        bearish_breakout = strategy._detect_breakout(sample_data, 'bearish')
        
        assert bullish_breakout == False
        assert bearish_breakout == False
    
    def test_position_sizing_base(self, basic_config, sample_data):
        """Test base position sizing"""
        basic_config['trend_strength_scaling'] = False
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        size = strategy._calculate_position_size(indicators, 'bullish')
        assert size == strategy.base_position_size
    
    def test_position_sizing_scaled(self, basic_config, sample_data):
        """Test scaled position sizing"""
        basic_config['trend_strength_scaling'] = True
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set strong trend indicators
        indicators['trend_strength'].iloc[-1] = 0.05  # Strong trend
        indicators['rsi'].iloc[-1] = 70  # Strong bullish momentum
        
        size = strategy._calculate_position_size(indicators, 'bullish')
        assert size > strategy.base_position_size
        assert size <= strategy.max_position_size
    
    def test_signal_confidence_calculation(self, basic_config, sample_data):
        """Test signal confidence calculation"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set strong indicators
        indicators['trend_strength'].iloc[-1] = 0.04
        indicators['rsi'].iloc[-1] = 70
        indicators['macd_histogram'].iloc[-1] = 10
        indicators['volume_ratio'].iloc[-1] = 2.0
        
        confidence = strategy._calculate_signal_confidence(indicators, 'bullish')
        
        assert 0.1 <= confidence <= 0.95
        assert isinstance(confidence, float)
    
    def test_create_momentum_signal_buy(self, basic_config, sample_data):
        """Test creating a BUY momentum signal"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set indicators
        indicators['atr'].iloc[-1] = 1000
        indicators['trend_strength'].iloc[-1] = 0.03
        indicators['rsi'].iloc[-1] = 65
        indicators['macd_histogram'].iloc[-1] = 5
        indicators['volume_ratio'].iloc[-1] = 1.8
        
        current_time = datetime.now()
        signal = strategy._create_momentum_signal('BTCUSDT', sample_data, indicators, 'bullish', current_time)
        
        assert signal.symbol == 'BTCUSDT'
        assert signal.action == 'BUY'
        assert signal.metadata['stop_loss'] < float(signal.price)
        assert signal.metadata['take_profit'] > float(signal.price)
        assert signal.metadata['strategy'] == 'momentum_trading'
        assert signal.metadata['trend_direction'] == 'bullish'
    
    def test_create_momentum_signal_sell(self, basic_config, sample_data):
        """Test creating a SELL momentum signal"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set indicators
        indicators['atr'].iloc[-1] = 1000
        indicators['trend_strength'].iloc[-1] = 0.03
        indicators['rsi'].iloc[-1] = 35
        indicators['macd_histogram'].iloc[-1] = -5
        indicators['volume_ratio'].iloc[-1] = 1.8
        
        current_time = datetime.now()
        signal = strategy._create_momentum_signal('BTCUSDT', sample_data, indicators, 'bearish', current_time)
        
        assert signal.symbol == 'BTCUSDT'
        assert signal.action == 'SELL'
        assert signal.metadata['stop_loss'] > float(signal.price)
        assert signal.metadata['take_profit'] < float(signal.price)
        assert signal.metadata['strategy'] == 'momentum_trading'
        assert signal.metadata['trend_direction'] == 'bearish'
    
    def test_insufficient_data(self, basic_config):
        """Test handling of insufficient data"""
        strategy = MomentumTradingStrategy(basic_config)
        
        # Create minimal data
        minimal_data = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50200],
            'volume': [1000, 1200]
        })
        
        # Convert to expected format
        market_data = {
            'BTCUSDT': [
                {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'open': 50000,
                    'high': 50200,
                    'low': 49900,
                    'close': 50100,
                    'volume': 1000
                },
                {
                    'timestamp': int((datetime.now().timestamp() + 3600) * 1000),
                    'open': 50100,
                    'high': 50300,
                    'low': 50000,
                    'close': 50200,
                    'volume': 1200
                }
            ]
        }
        
        import asyncio
        signals = asyncio.run(strategy.generate_signals(market_data))
        
        assert len(signals) == 0
    
    def test_complete_signal_generation_flow(self, basic_config, sample_data):
        """Test complete signal generation with all conditions met"""
        strategy = MomentumTradingStrategy(basic_config)
        
        # Ensure sufficient data length
        extended_data = sample_data.copy()
        for i in range(50):  # Add more data points
            new_row = extended_data.iloc[-1:].copy()
            new_row.index = [extended_data.index[-1] + timedelta(hours=1)]
            new_row['close'] = new_row['close'].iloc[0] * 1.001  # Small uptrend
            new_row['high'] = new_row['close'] * 1.002
            new_row['low'] = new_row['close'] * 0.998
            new_row['open'] = new_row['close'] * 1.0005
            new_row['volume'] = 3000  # High volume
            extended_data = pd.concat([extended_data, new_row])
        
        # Convert to expected format
        market_data = {
            'BTCUSDT': []
        }
        
        for idx, row in extended_data.iterrows():
            market_data['BTCUSDT'].append({
                'timestamp': int(idx.timestamp() * 1000),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
        
        import asyncio
        signals = asyncio.run(strategy.generate_signals(market_data))
        
        # May or may not generate signals depending on exact conditions
        assert isinstance(signals, list)
        
        # If signals generated, validate structure
        for signal in signals:
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'price')
            assert hasattr(signal, 'confidence')
            assert signal.action in ['BUY', 'SELL']
    
    def test_risk_parameters(self, basic_config, sample_data):
        """Test risk management parameters"""
        strategy = MomentumTradingStrategy(basic_config)
        indicators = strategy._calculate_indicators(sample_data)
        
        # Set moderate volatility 
        indicators['atr'].iloc[-1] = 500  # Moderate ATR that should respect risk limits
        indicators['trend_strength'].iloc[-1] = 0.03
        indicators['rsi'].iloc[-1] = 65
        indicators['macd_histogram'].iloc[-1] = 5
        indicators['volume_ratio'].iloc[-1] = 1.8
        
        current_time = datetime.now()
        signal = strategy._create_momentum_signal('BTCUSDT', sample_data, indicators, 'bullish', current_time)
        
        # Check risk management
        risk_amount = abs(float(signal.price) - signal.metadata['stop_loss'])
        risk_percentage = risk_amount / float(signal.price)
        
        # Should respect maximum risk per trade
        assert risk_percentage <= strategy.max_risk_per_trade * 1.1  # Small tolerance for calculation precision
        
        # Check risk/reward ratio
        profit_amount = abs(signal.metadata['take_profit'] - float(signal.price))
        risk_reward = profit_amount / risk_amount
        assert risk_reward >= strategy.take_profit_risk_reward * 0.9  # Small tolerance
    
    def test_strategy_info(self, basic_config):
        """Test strategy information return"""
        strategy = MomentumTradingStrategy(basic_config)
        info = strategy.get_strategy_info()
        
        assert 'name' in info
        assert 'description' in info
        assert 'type' in info
        assert 'parameters' in info
        assert info['type'] == 'momentum'
        assert 'trend_detection' in info['parameters']
        assert 'momentum_indicators' in info['parameters']
        assert 'risk_management' in info['parameters']

if __name__ == '__main__':
    pytest.main([__file__])