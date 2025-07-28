"""
Unit tests for Moving Average Crossover Strategy (New Architecture)
Tests strategy logic with schema validation, shared components, and new configuration system
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch
from decimal import Decimal

from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
from src.strategies.signal import Signal
from src.strategies.config.validator import ValidationError


@pytest.fixture
def simple_ma_config():
    """Valid configuration for SimpleMovingAverageCrossover (new architecture)"""
    return {
        'symbols': ['BTCUSDT'],
        'position_size': 0.02,
        'fast_period': 8,
        'slow_period': 21,
        'timeframes': ['5m'],
        'stop_loss_pct': 2.5,
        'take_profit_pct': 5.0,
        'atr_period': 14,
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 1.8,
        'trailing_stop_type': 'percentage'
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    timestamps = pd.date_range('2025-01-01', periods=50, freq='5min')
    prices = 3800 + np.cumsum(np.random.normal(0, 5, 50))  # Trending price data
    
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': price + np.random.uniform(-2, 2),
            'high': price + abs(np.random.normal(2, 1)),
            'low': price - abs(np.random.normal(2, 1)),
            'close': price,
            'volume': np.random.uniform(1000, 2000)
        })
    
    return {'BTCUSDT': data}


class TestSimpleMAKCrossoverStrategy:
    """Test Simple Moving Average Crossover Strategy (New Architecture)"""

    def test_strategy_initialization_valid_config(self, simple_ma_config):
        """Test strategy initialization with valid new architecture config"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Check that strategy uses schema-validated parameters
        assert strategy.strategy_name == 'ma_crossover_simple'
        assert strategy.get_parameter_value('fast_period') == 8
        assert strategy.get_parameter_value('slow_period') == 21
        assert strategy.get_parameter_value('position_size') == 0.02
        assert strategy.symbols == ['BTCUSDT']
        assert strategy.get_parameter_value('trailing_stop_enabled') == True
        assert strategy.get_parameter_value('trailing_stop_distance') == 1.8

    def test_strategy_initialization_invalid_config(self):
        """Test strategy initialization with invalid config (schema validation)"""
        # Missing required parameters
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02
            # Missing fast_period and slow_period
        }
        
        with pytest.raises(ValidationError):
            SimpleMovingAverageCrossover(invalid_config)

    def test_strategy_initialization_invalid_parameter_types(self):
        """Test strategy initialization with wrong parameter types"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 'not_a_number',  # Should be integer
            'slow_period': 21
        }
        
        with pytest.raises(ValidationError):
            SimpleMovingAverageCrossover(invalid_config)

    def test_parameter_access_through_schema(self, simple_ma_config):
        """Test that parameters are accessed through schema validation system"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Test parameter access method
        assert strategy.get_parameter_value('fast_period') == 8
        assert strategy.get_parameter_value('slow_period') == 21
        assert strategy.get_parameter_value('stop_loss_pct') == 2.5
        
        # Test default parameter handling
        assert strategy.get_parameter_value('lookback_periods') == 100  # Default value
        
        # Test non-existent parameter
        assert strategy.get_parameter_value('non_existent', 'default') == 'default'

    def test_shared_components_initialization(self, simple_ma_config):
        """Test that shared components are properly initialized"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Check shared components exist
        assert hasattr(strategy, 'indicator_calculator')
        assert hasattr(strategy, 'signal_manager')
        assert hasattr(strategy, 'risk_manager')
        
        # Check data requirements are calculated
        data_req = strategy.get_required_data()
        assert 'timeframes' in data_req
        assert 'lookback_periods' in data_req

    @pytest.mark.asyncio
    async def test_signal_generation_with_new_architecture(self, simple_ma_config, sample_market_data):
        """Test signal generation using new architecture patterns"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        signals = await strategy.generate_signals(sample_market_data)
        
        # Should return a list (may be empty due to strict filters)
        assert isinstance(signals, list)
        
        # If signals are generated, they should follow new architecture
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol == 'BTCUSDT'
            assert signal.action in ['BUY', 'SELL', 'HOLD']
            assert 0 <= signal.confidence <= 1.0
            assert hasattr(signal, 'metadata')
            
            # Check trailing stop metadata is included
            if signal.metadata:
                assert 'trailing_stop_enabled' in signal.metadata
                assert 'trailing_stop_distance' in signal.metadata
                assert 'trailing_stop_type' in signal.metadata

    @pytest.mark.asyncio
    async def test_trailing_stop_metadata_in_signals(self, simple_ma_config, sample_market_data):
        """Test that trailing stop parameters are included in signal metadata"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Generate signals (using a less restrictive config if needed)
        signals = await strategy.generate_signals(sample_market_data)
        
        # Check all signals have trailing stop metadata
        for signal in signals:
            if signal.metadata:
                assert signal.metadata.get('trailing_stop_enabled') == True
                assert signal.metadata.get('trailing_stop_distance') == 1.8
                assert signal.metadata.get('trailing_stop_type') == 'percentage'

    def test_indicator_calculation_through_shared_component(self, simple_ma_config):
        """Test that indicators are calculated through shared IndicatorCalculator"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Create sample DataFrame with all required columns
        data = pd.DataFrame({
            'close': [3800, 3805, 3810, 3815, 3820, 3825, 3830, 3835, 3840, 3845] + [3850] * 15,
            'high': [3810, 3815, 3820, 3825, 3830, 3835, 3840, 3845, 3850, 3855] + [3860] * 15,
            'low': [3790, 3795, 3800, 3805, 3810, 3815, 3820, 3825, 3830, 3835] + [3840] * 15,
            'volume': [1000] * 25
        })
        
        # Test indicator calculation
        indicators = strategy.calculate_indicators(data)
        
        # Should include required indicators
        assert 'sma_fast' in indicators
        assert 'sma_slow' in indicators
        
        # Indicators should be pandas Series
        assert isinstance(indicators['sma_fast'], pd.Series)
        assert isinstance(indicators['sma_slow'], pd.Series)

    @pytest.mark.asyncio 
    async def test_insufficient_data_handling(self, simple_ma_config):
        """Test strategy behavior with insufficient data"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Create insufficient data (less than required periods)
        insufficient_data = {
            'BTCUSDT': [
                {'timestamp': 1640995200000, 'open': 3800, 'high': 3810, 'low': 3790, 'close': 3805, 'volume': 1000},
                {'timestamp': 1640995500000, 'open': 3805, 'high': 3815, 'low': 3795, 'close': 3810, 'volume': 1100}
            ]
        }
        
        signals = await strategy.generate_signals(insufficient_data)
        
        # Should handle gracefully and return empty list
        assert signals == []

    @pytest.mark.asyncio
    async def test_error_handling_in_signal_generation(self, simple_ma_config):
        """Test error handling in signal generation"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # Test with malformed data
        malformed_data = {
            'BTCUSDT': [
                {'timestamp': 'invalid', 'close': 'not_a_number'}  # Invalid data
            ]
        }
        
        # Should not raise exception, should handle gracefully
        signals = await strategy.generate_signals(malformed_data)
        assert isinstance(signals, list)

    def test_strategy_info_method(self, simple_ma_config):
        """Test strategy info method returns correct information"""
        strategy = SimpleMovingAverageCrossover(simple_ma_config)
        
        # The strategy should have access to parameter info
        assert strategy.strategy_name == 'ma_crossover_simple'
        assert len(strategy.symbols) == 1
        assert strategy.symbols[0] == 'BTCUSDT'

