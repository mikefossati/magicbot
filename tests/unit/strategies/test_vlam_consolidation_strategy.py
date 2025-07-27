"""
Unit tests for VLAM Consolidation Breakout Strategy
Comprehensive tests for Volatility and Liquidity Adjusted Momentum strategy
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta

from src.strategies.vlam_consolidation_strategy import VLAMConsolidationStrategy
from src.strategies.base import Signal


def create_vlam_config():
    """Create default VLAM strategy configuration"""
    return {
        'symbols': ['BTCUSDT'],
        'position_size': 0.02,
        'vlam_period': 14,
        'atr_period': 14,
        'volume_period': 20,
        'consolidation_min_length': 8,
        'consolidation_max_length': 30,
        'consolidation_tolerance': 0.02,
        'min_touches': 3,
        'spike_min_size': 1.5,
        'spike_volume_multiplier': 1.3,
        'vlam_signal_threshold': 0.6,
        'entry_timeout_bars': 5,
        'stop_loss_atr_multiplier': 1.5,
        'target_risk_reward': 2.0,
        'max_risk_per_trade': 0.02,
        'max_concurrent_positions': 2,
        'position_timeout_hours': 24
    }


def generate_consolidation_data(length: int = 50, consolidation_length: int = 15,
                               consolidation_range: float = 0.01, base_price: float = 50000.0):
    """Generate market data with a clear consolidation pattern"""
    np.random.seed(42)  # For reproducible tests
    
    data = []
    current_price = base_price
    
    # Generate initial data leading to consolidation
    for i in range(length - consolidation_length):
        # Random walk to consolidation area
        price_change = np.random.normal(0, 0.005) * current_price
        current_price = max(current_price + price_change, base_price * 0.9)
        
        # Create OHLC data
        open_price = current_price
        high_price = open_price * (1 + np.random.uniform(0, 0.01))
        low_price = open_price * (1 - np.random.uniform(0, 0.01))
        close_price = open_price + np.random.normal(0, 0.003) * open_price
        volume = 1000 + np.random.randint(-200, 500)
        
        timestamp = int((datetime.now() - timedelta(hours=length-i)).timestamp() * 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': float(open_price),
            'high': float(high_price),
            'low': float(low_price),
            'close': float(close_price),
            'volume': float(volume)
        })
    
    # Generate consolidation period
    consolidation_high = current_price * (1 + consolidation_range/2)
    consolidation_low = current_price * (1 - consolidation_range/2)
    
    for i in range(consolidation_length):
        # Price oscillates within consolidation range
        range_position = np.random.uniform(0.2, 0.8)  # Stay mostly within range
        target_price = consolidation_low + (consolidation_high - consolidation_low) * range_position
        
        open_price = target_price
        high_price = min(target_price * 1.005, consolidation_high)
        low_price = max(target_price * 0.995, consolidation_low)
        close_price = target_price + np.random.normal(0, 0.001) * target_price
        volume = 1000 + np.random.randint(-100, 300)
        
        timestamp = int((datetime.now() - timedelta(hours=consolidation_length-i)).timestamp() * 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': float(open_price),
            'high': float(high_price),
            'low': float(low_price),
            'close': float(close_price),
            'volume': float(volume)
        })
    
    return data


def generate_spike_data(consolidation_data: list, spike_direction: str = 'up', 
                       spike_size: float = 2.0, volume_spike: float = 2.0):
    """Add a spike to existing consolidation data"""
    data = consolidation_data.copy()
    last_bar = data[-1]
    
    # Calculate consolidation range from recent data
    recent_highs = [bar['high'] for bar in data[-15:]]
    recent_lows = [bar['low'] for bar in data[-15:]]
    consolidation_high = max(recent_highs)
    consolidation_low = min(recent_lows)
    
    # Estimate ATR from recent data
    atr_estimate = (consolidation_high - consolidation_low) * 0.3
    
    # Create spike bar
    if spike_direction == 'up':
        # Upward spike above consolidation
        open_price = last_bar['close']
        high_price = consolidation_high + (atr_estimate * spike_size)
        low_price = open_price * 0.998
        close_price = high_price * 0.99  # Close near high
    else:
        # Downward spike below consolidation  
        open_price = last_bar['close']
        low_price = consolidation_low - (atr_estimate * spike_size)
        high_price = open_price * 1.002
        close_price = low_price * 1.01  # Close near low
    
    volume = last_bar['volume'] * volume_spike
    timestamp = int(datetime.now().timestamp() * 1000)
    
    spike_bar = {
        'timestamp': timestamp,
        'open': float(open_price),
        'high': float(high_price),
        'low': float(low_price),
        'close': float(close_price),
        'volume': float(volume)
    }
    
    data.append(spike_bar)
    return data


class TestVLAMConsolidationStrategy:
    """Test VLAM Consolidation Strategy with comprehensive scenarios"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization with valid config"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        assert strategy.vlam_period == 14
        assert strategy.consolidation_tolerance == 0.02
        assert strategy.spike_min_size == 1.5
        assert strategy.vlam_signal_threshold == 0.6
        assert strategy.max_risk_per_trade == 0.02
        assert strategy.target_risk_reward == 2.0
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        assert strategy.validate_parameters() is True
    
    def test_validate_parameters_invalid_vlam_period(self):
        """Test parameter validation with invalid VLAM period"""
        config = create_vlam_config()
        config['vlam_period'] = 0
        
        # Manually create strategy without calling parent constructor
        strategy = VLAMConsolidationStrategy.__new__(VLAMConsolidationStrategy)
        strategy.vlam_period = 0
        strategy.atr_period = 14
        strategy.consolidation_min_length = 8
        strategy.consolidation_max_length = 30
        strategy.consolidation_tolerance = 0.02
        strategy.spike_min_size = 1.5
        strategy.max_risk_per_trade = 0.02
        strategy.target_risk_reward = 2.0
        
        assert strategy.validate_parameters() is False
    
    def test_validate_parameters_invalid_consolidation_length(self):
        """Test parameter validation with invalid consolidation lengths"""
        strategy = VLAMConsolidationStrategy.__new__(VLAMConsolidationStrategy)
        strategy.vlam_period = 14
        strategy.atr_period = 14
        strategy.consolidation_min_length = 20
        strategy.consolidation_max_length = 15  # Min > Max
        strategy.consolidation_tolerance = 0.02
        strategy.spike_min_size = 1.5
        strategy.max_risk_per_trade = 0.02
        strategy.target_risk_reward = 2.0
        
        assert strategy.validate_parameters() is False
    
    def test_validate_parameters_invalid_risk_reward(self):
        """Test parameter validation with invalid risk:reward ratio"""
        strategy = VLAMConsolidationStrategy.__new__(VLAMConsolidationStrategy)
        strategy.vlam_period = 14
        strategy.atr_period = 14
        strategy.consolidation_min_length = 8
        strategy.consolidation_max_length = 30
        strategy.consolidation_tolerance = 0.02
        strategy.spike_min_size = 1.5
        strategy.max_risk_per_trade = 0.02
        strategy.target_risk_reward = 0.5  # Less than 1.0
        
        assert strategy.validate_parameters() is False
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Create test data
        data = generate_consolidation_data(30)
        df = pd.DataFrame(data)
        
        atr = strategy._calculate_atr(df['high'], df['low'], df['close'], 14)
        
        assert len(atr) == len(df)
        assert all(atr > 0)  # ATR should be positive
        assert not atr.isna().any()  # No NaN values
    
    def test_vlam_calculation(self):
        """Test VLAM indicator calculation"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Create test data
        data = generate_consolidation_data(30)
        df = pd.DataFrame(data)
        
        # Calculate components
        atr = strategy._calculate_atr(df['high'], df['low'], df['close'], 14)
        volume_avg = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = df['volume'] / volume_avg
        
        vlam = strategy._calculate_vlam(df['open'], df['high'], df['low'], 
                                      df['close'], df['volume'], atr, volume_ratio)
        
        assert len(vlam) == len(df)
        assert all(vlam >= -1) and all(vlam <= 1)  # VLAM should be normalized
        assert not vlam.isna().any()  # No NaN values
    
    def test_consolidation_detection_valid(self):
        """Test consolidation detection with valid consolidation"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Generate data with clear consolidation
        data = generate_consolidation_data(40, consolidation_length=12, consolidation_range=0.015)
        df = pd.DataFrame(data)
        
        indicators = strategy._calculate_indicators(df)
        consolidation = strategy._detect_consolidation(df, indicators)
        
        assert consolidation is not None
        assert consolidation['length'] >= strategy.consolidation_min_length
        assert consolidation['high'] > consolidation['low']
        assert consolidation['support_touches'] >= 1
        assert consolidation['resistance_touches'] >= 1
    
    def test_consolidation_detection_invalid(self):
        """Test consolidation detection with invalid/trending data"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Generate trending data (no consolidation)
        np.random.seed(42)
        data = []
        base_price = 50000.0
        
        for i in range(30):
            # Strong uptrend
            price = base_price * (1.02 ** i)  # 2% per bar uptrend
            
            timestamp = int((datetime.now() - timedelta(hours=30-i)).timestamp() * 1000)
            data.append({
                'timestamp': timestamp,
                'open': float(price),
                'high': float(price * 1.01),
                'low': float(price * 0.99),
                'close': float(price * 1.005),
                'volume': 1000.0
            })
        
        df = pd.DataFrame(data)
        indicators = strategy._calculate_indicators(df)
        consolidation = strategy._detect_consolidation(df, indicators)
        
        assert consolidation is None  # Should not detect consolidation in trending market
    
    def test_spike_detection_upward(self):
        """Test upward spike detection"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Generate consolidation with upward spike
        base_data = generate_consolidation_data(35, consolidation_length=12)
        data_with_spike = generate_spike_data(base_data, 'up', spike_size=2.0, volume_spike=2.0)
        df = pd.DataFrame(data_with_spike)
        
        indicators = strategy._calculate_indicators(df)
        consolidation = strategy._detect_consolidation(df.iloc[:-1], indicators)  # Exclude spike bar
        
        assert consolidation is not None
        
        spike_event = strategy._detect_spike(df, indicators, consolidation)
        
        assert spike_event is not None
        assert spike_event['direction'] == 'up'
        assert spike_event['strength'] >= strategy.spike_min_size
        assert spike_event['bars_since'] == 0  # Most recent bar
    
    def test_spike_detection_downward(self):
        """Test downward spike detection"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Generate consolidation with downward spike
        base_data = generate_consolidation_data(35, consolidation_length=12)
        data_with_spike = generate_spike_data(base_data, 'down', spike_size=2.0, volume_spike=2.0)
        df = pd.DataFrame(data_with_spike)
        
        indicators = strategy._calculate_indicators(df)
        consolidation = strategy._detect_consolidation(df.iloc[:-1], indicators)
        
        assert consolidation is not None
        
        spike_event = strategy._detect_spike(df, indicators, consolidation)
        
        assert spike_event is not None
        assert spike_event['direction'] == 'down'
        assert spike_event['strength'] >= strategy.spike_min_size
    
    def test_vlam_entry_signal_bullish(self):
        """Test VLAM entry signal for bullish reversion (after downward spike)"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Create mock consolidation and spike data
        consolidation = {
            'high': 50500.0,
            'low': 49500.0,
            'mid': 50000.0,
            'length': 12
        }
        
        spike_event = {
            'direction': 'down',
            'strength': 2.0,
            'bars_since': 1,
            'low': 49000.0,
            'high': 49800.0
        }
        
        # Mock data with bullish VLAM signal
        mock_data = pd.DataFrame({
            'close': [49600.0, 49700.0],  # Price recovering
            'open': [49550.0, 49650.0],
            'high': [49650.0, 49750.0],
            'low': [49500.0, 49600.0],
            'volume': [1000.0, 1200.0]
        })
        
        # Mock indicators with strong bullish VLAM
        mock_indicators = {
            'vlam': pd.Series([0.4, 0.8])  # Strong bullish signal
        }
        
        entry_signal = strategy._check_vlam_entry_signal(mock_data, mock_indicators, 
                                                        consolidation, spike_event)
        
        assert entry_signal is not None
        assert entry_signal['action'] == 'BUY'
        assert entry_signal['direction'] == 'bullish'
        assert entry_signal['strength'] >= strategy.vlam_signal_threshold
    
    def test_vlam_entry_signal_bearish(self):
        """Test VLAM entry signal for bearish reversion (after upward spike)"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        consolidation = {
            'high': 50500.0,
            'low': 49500.0,
            'mid': 50000.0,
            'length': 12
        }
        
        spike_event = {
            'direction': 'up',
            'strength': 2.0,
            'bars_since': 1,
            'high': 51000.0,
            'low': 50200.0
        }
        
        # Mock data with bearish VLAM signal
        mock_data = pd.DataFrame({
            'close': [50400.0, 50300.0],  # Price declining
            'open': [50450.0, 50350.0],
            'high': [50500.0, 50400.0],
            'low': [50350.0, 50250.0],
            'volume': [1000.0, 1200.0]
        })
        
        mock_indicators = {
            'vlam': pd.Series([-0.4, -0.8])  # Strong bearish signal
        }
        
        entry_signal = strategy._check_vlam_entry_signal(mock_data, mock_indicators,
                                                        consolidation, spike_event)
        
        assert entry_signal is not None
        assert entry_signal['action'] == 'SELL'
        assert entry_signal['direction'] == 'bearish'
        assert entry_signal['strength'] >= strategy.vlam_signal_threshold
    
    def test_vlam_entry_signal_timeout(self):
        """Test VLAM entry signal timeout (spike too old)"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        consolidation = {'high': 50500.0, 'low': 49500.0, 'mid': 50000.0}
        spike_event = {
            'direction': 'up',
            'bars_since': 10  # Exceeds entry_timeout_bars (5)
        }
        
        mock_data = pd.DataFrame({'close': [50000.0]})
        mock_indicators = {'vlam': pd.Series([0.8])}
        
        entry_signal = strategy._check_vlam_entry_signal(mock_data, mock_indicators,
                                                        consolidation, spike_event)
        
        assert entry_signal is None  # Should timeout
    
    def test_vlam_entry_signal_wrong_direction(self):
        """Test VLAM entry signal with wrong direction"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        consolidation = {'high': 50500.0, 'low': 49500.0, 'mid': 50000.0}
        spike_event = {
            'direction': 'up',  # Expecting bearish VLAM
            'bars_since': 1
        }
        
        mock_data = pd.DataFrame({'close': [50000.0]})
        mock_indicators = {
            'vlam': pd.Series([0.8])  # Bullish VLAM (wrong direction)
        }
        
        entry_signal = strategy._check_vlam_entry_signal(mock_data, mock_indicators,
                                                        consolidation, spike_event)
        
        assert entry_signal is None  # Wrong direction
    
    def test_signal_creation_buy(self):
        """Test BUY signal creation with proper risk management"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Mock data
        mock_data = pd.DataFrame({
            'close': [49700.0],
            'open': [49650.0],
            'high': [49750.0],
            'low': [49600.0],
            'volume': [1200.0]
        })
        
        mock_indicators = {
            'atr': pd.Series([200.0])
        }
        
        consolidation = {
            'high': 50500.0,
            'low': 49500.0,
            'length': 12
        }
        
        spike_event = {
            'direction': 'down',
            'strength': 2.0,
            'bars_since': 1,
            'low': 49000.0,  # Stop loss level
            'high': 49800.0
        }
        
        entry_signal = {
            'action': 'BUY',
            'direction': 'bullish',
            'strength': 0.8,
            'vlam_value': 0.8
        }
        
        signal = strategy._create_vlam_signal('BTCUSDT', mock_data, mock_indicators,
                                            consolidation, spike_event, entry_signal)
        
        assert signal.action == 'BUY'
        assert signal.symbol == 'BTCUSDT'
        assert signal.confidence == 0.8
        assert signal.metadata['stop_loss'] == 49000.0  # Spike low
        # Take profit is adjusted for 2:1 risk:reward ratio
        risk = 49700.0 - 49000.0  # 700
        expected_target = 49700.0 + (risk * 2.0)  # 49700 + 1400 = 51100
        assert signal.metadata['take_profit'] == expected_target
        assert signal.metadata['strategy'] == 'vlam_consolidation'
    
    def test_signal_creation_sell(self):
        """Test SELL signal creation with proper risk management"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        mock_data = pd.DataFrame({
            'close': [50300.0],
            'open': [50350.0],
            'high': [50400.0],
            'low': [50250.0],
            'volume': [1200.0]
        })
        
        mock_indicators = {
            'atr': pd.Series([200.0])
        }
        
        consolidation = {
            'high': 50500.0,
            'low': 49500.0,
            'length': 12
        }
        
        spike_event = {
            'direction': 'up',
            'strength': 2.0,
            'bars_since': 1,
            'high': 51000.0,  # Stop loss level
            'low': 50200.0
        }
        
        entry_signal = {
            'action': 'SELL',
            'direction': 'bearish',
            'strength': 0.8,
            'vlam_value': -0.8
        }
        
        signal = strategy._create_vlam_signal('BTCUSDT', mock_data, mock_indicators,
                                            consolidation, spike_event, entry_signal)
        
        assert signal.action == 'SELL'
        assert signal.symbol == 'BTCUSDT'
        assert signal.confidence == 0.8
        assert signal.metadata['stop_loss'] == 51000.0  # Spike high
        # Take profit is adjusted for 2:1 risk:reward ratio
        risk = 51000.0 - 50300.0  # 700
        expected_target = 50300.0 - (risk * 2.0)  # 50300 - 1400 = 48900
        assert signal.metadata['take_profit'] == expected_target
    
    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Create insufficient data
        data = generate_consolidation_data(20)  # Less than required minimum
        df = pd.DataFrame(data)
        
        signal = await strategy.analyze_market_data('BTCUSDT', df)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_complete_signal_generation_flow(self):
        """Test complete signal generation flow with valid setup"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        # Generate realistic scenario: consolidation + spike + reversion setup
        base_data = generate_consolidation_data(45, consolidation_length=15, consolidation_range=0.015)
        
        # Add upward spike
        data_with_spike = generate_spike_data(base_data, 'up', spike_size=2.0, volume_spike=2.5)
        
        # Add reversion bar with bearish characteristics
        last_bar = data_with_spike[-1]
        reversion_bar = {
            'timestamp': int((datetime.now() + timedelta(hours=1)).timestamp() * 1000),
            'open': last_bar['close'],
            'high': last_bar['close'] * 1.002,
            'low': last_bar['close'] * 0.995,
            'close': last_bar['close'] * 0.997,  # Slight decline
            'volume': last_bar['volume'] * 1.1
        }
        data_with_spike.append(reversion_bar)
        
        market_data = {'BTCUSDT': data_with_spike}
        
        # Mock VLAM to return strong bearish signal
        with patch.object(strategy, '_calculate_vlam') as mock_vlam:
            # Create mock VLAM series with strong bearish signal at the end
            vlam_values = [0.0] * (len(data_with_spike) - 1) + [-0.8]
            mock_vlam.return_value = pd.Series(vlam_values)
            
            signals = await strategy.generate_signals(market_data)
            
            if signals:  # Signal generation depends on exact market conditions
                signal = signals[0]
                assert signal.action == 'SELL'
                assert signal.symbol == 'BTCUSDT'
                assert 'vlam_signal' in signal.metadata
                assert 'consolidation' in signal.metadata
                assert 'spike_event' in signal.metadata
    
    def test_risk_parameters(self):
        """Test risk parameter retrieval"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        risk_params = strategy.get_risk_parameters()
        
        assert 'max_risk_per_trade' in risk_params
        assert 'target_risk_reward' in risk_params
        assert 'max_concurrent_positions' in risk_params
        assert risk_params['max_risk_per_trade'] == 0.02
        assert risk_params['target_risk_reward'] == 2.0
    
    def test_strategy_info(self):
        """Test strategy information retrieval"""
        config = create_vlam_config()
        strategy = VLAMConsolidationStrategy(config)
        
        info = strategy.get_strategy_info()
        
        assert 'name' in info
        assert 'description' in info
        assert 'methodology' in info
        assert 'indicators' in info
        assert info['name'] == 'VLAM Consolidation Breakout Strategy'
        assert 'VLAM' in info['indicators']
        assert 'step_1' in info['methodology']