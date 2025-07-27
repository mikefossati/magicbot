"""
Unit tests for Moving Average Crossover Strategy
Tests strategy logic with mocked data, focusing on signal generation
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from decimal import Decimal

from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.base import Signal
from tests.unit.fixtures.historical_snapshots import get_historical_snapshot


class TestMAKCrossoverStrategy:
    """Test Moving Average Crossover Strategy"""

    def test_strategy_initialization(self, ma_crossover_config):
        """Test strategy initialization with valid config"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.position_size == Decimal('0.1')
        assert strategy.symbols == ['BTCUSDT']
        assert strategy.last_signals == {}

    def test_strategy_initialization_invalid_config(self):
        """Test strategy initialization with invalid config"""
        # Fast period >= slow period
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 30,
            'slow_period': 10,
            'position_size': 0.1
        }
        
        with pytest.raises(ValueError):
            MovingAverageCrossover(invalid_config)

    def test_validate_parameters_valid(self, ma_crossover_config):
        """Test parameter validation with valid parameters"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        assert strategy.validate_parameters() is True

    def test_validate_parameters_invalid_periods(self):
        """Test parameter validation with invalid periods"""
        # Test fast >= slow
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 30,
            'slow_period': 10,
            'position_size': 0.1
        }
        strategy = MovingAverageCrossover.__new__(MovingAverageCrossover)
        strategy.fast_period = 30
        strategy.slow_period = 10
        strategy.symbols = ['BTCUSDT']
        
        assert strategy.validate_parameters() is False
        
        # Test periods < 2
        strategy.fast_period = 1
        strategy.slow_period = 5
        assert strategy.validate_parameters() is False

    def test_validate_parameters_no_symbols(self):
        """Test parameter validation with no symbols"""
        strategy = MovingAverageCrossover.__new__(MovingAverageCrossover)
        strategy.fast_period = 10
        strategy.slow_period = 30
        strategy.symbols = []
        
        assert strategy.validate_parameters() is False

    def test_get_required_data(self, ma_crossover_config):
        """Test required data specifications"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        required_data = strategy.get_required_data()
        
        assert 'timeframes' in required_data
        assert 'lookback_periods' in required_data
        assert 'indicators' in required_data
        assert required_data['lookback_periods'] == 40  # slow_period + 10
        assert 'sma' in required_data['indicators']

    @pytest.mark.asyncio
    async def test_generate_signals_no_data(self, ma_crossover_config):
        """Test signal generation with no market data"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {}
        signals = await strategy.generate_signals(market_data)
        
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(self, ma_crossover_config):
        """Test signal generation with insufficient data"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        # Only 10 data points, need at least 30 for slow MA
        short_data = get_historical_snapshot('bullish_crossover')[:10]
        market_data = {'BTCUSDT': short_data}
        
        signals = await strategy.generate_signals(market_data)
        
        assert signals == []

    @pytest.mark.asyncio
    async def test_bullish_crossover_signal(self, ma_crossover_config):
        """Test bullish crossover signal generation"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        # Use bullish crossover data
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock a bullish crossover scenario
            mock_calc.return_value = {
                'current_fast': 51000.0,
                'current_slow': 50000.0,
                'prev_fast': 49900.0,  # Was below slow
                'prev_slow': 50000.0,
                'current_price': 51000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 1
            signal = signals[0]
            assert signal.action == 'BUY'
            assert signal.symbol == 'BTCUSDT'
            assert signal.confidence > 0.7
            assert signal.price == Decimal('51000.0')
            assert 'fast_ma' in signal.metadata
            assert 'slow_ma' in signal.metadata

    @pytest.mark.asyncio
    async def test_bearish_crossover_signal(self, ma_crossover_config):
        """Test bearish crossover signal generation"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bearish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock a bearish crossover scenario
            mock_calc.return_value = {
                'current_fast': 49000.0,
                'current_slow': 50000.0,
                'prev_fast': 50100.0,  # Was above slow
                'prev_slow': 50000.0,
                'current_price': 49000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 1
            signal = signals[0]
            assert signal.action == 'SELL'
            assert signal.symbol == 'BTCUSDT'
            assert signal.confidence > 0.7
            assert signal.price == Decimal('49000.0')

    @pytest.mark.asyncio
    async def test_no_crossover_no_signal(self, ma_crossover_config):
        """Test no signal when there's no crossover"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock no crossover scenario - fast MA consistently above slow MA
            mock_calc.return_value = {
                'current_fast': 51000.0,
                'current_slow': 50000.0,
                'prev_fast': 50500.0,  # Was also above slow
                'prev_slow': 50000.0,
                'current_price': 51000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            
            assert signals == []

    @pytest.mark.asyncio
    async def test_duplicate_signal_prevention(self, ma_crossover_config):
        """Test that duplicate signals are not generated"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock bullish crossover
            mock_calc.return_value = {
                'current_fast': 51000.0,
                'current_slow': 50000.0,
                'prev_fast': 49900.0,
                'prev_slow': 50000.0,
                'current_price': 51000.0
            }
            
            # First call should generate signal
            signals1 = await strategy.generate_signals(market_data)
            assert len(signals1) == 1
            assert signals1[0].action == 'BUY'
            
            # Second call with same conditions should not generate signal
            signals2 = await strategy.generate_signals(market_data)
            assert signals2 == []

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, ma_crossover_config):
        """Test confidence calculation based on crossover strength"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Strong crossover (large separation)
            mock_calc.return_value = {
                'current_fast': 52000.0,  # 4% above slow MA
                'current_slow': 50000.0,
                'prev_fast': 49900.0,
                'prev_slow': 50000.0,
                'current_price': 52000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            strong_confidence = signals[0].confidence
            
            # Reset last signals to allow new signal
            strategy.last_signals = {}
            
            # Weak crossover (small separation)
            mock_calc.return_value = {
                'current_fast': 50100.0,  # 0.2% above slow MA
                'current_slow': 50000.0,
                'prev_fast': 49900.0,
                'prev_slow': 50000.0,
                'current_price': 50100.0
            }
            
            signals = await strategy.generate_signals(market_data)
            weak_confidence = signals[0].confidence
            
            # Strong crossover should have higher confidence
            assert strong_confidence > weak_confidence

    @pytest.mark.asyncio
    async def test_nan_values_handling(self, ma_crossover_config):
        """Test handling of NaN values in calculations"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock NaN values
            mock_calc.return_value = {'error': 'nan_values'}
            
            signals = await strategy.generate_signals(market_data)
            
            assert signals == []

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, ma_crossover_config):
        """Test handling of insufficient data"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            mock_calc.return_value = {'error': 'insufficient_data'}
            
            signals = await strategy.generate_signals(market_data)
            
            assert signals == []

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        """Test signal generation for multiple symbols"""
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'fast_period': 10,
            'slow_period': 30,
            'position_size': 0.1
        }
        strategy = MovingAverageCrossover(config)
        
        market_data = {
            'BTCUSDT': get_historical_snapshot('bullish_crossover'),
            'ETHUSDT': get_historical_snapshot('eth_sample')
        }
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            # Mock bullish crossover for both symbols
            mock_calc.return_value = {
                'current_fast': 51000.0,
                'current_slow': 50000.0,
                'prev_fast': 49900.0,
                'prev_slow': 50000.0,
                'current_price': 51000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 2
            symbols = [signal.symbol for signal in signals]
            assert 'BTCUSDT' in symbols
            assert 'ETHUSDT' in symbols

    def test_moving_averages_calculation(self, ma_crossover_config):
        """Test the actual moving averages calculation logic"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        # Use real data for calculation test
        data = get_historical_snapshot('bullish_crossover')
        
        result = strategy._calculate_moving_averages(data)
        
        # Should return valid results
        assert 'current_fast' in result
        assert 'current_slow' in result
        assert 'prev_fast' in result
        assert 'prev_slow' in result
        assert 'current_price' in result
        
        # Values should be reasonable
        assert result['current_fast'] > 0
        assert result['current_slow'] > 0
        assert result['current_price'] > 0

    def test_moving_averages_calculation_insufficient_data(self, ma_crossover_config):
        """Test moving averages calculation with insufficient data"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        # Only 1 data point
        data = get_historical_snapshot('bullish_crossover')[:1]
        
        result = strategy._calculate_moving_averages(data)
        
        assert result == {'error': 'insufficient_data'}

    @pytest.mark.asyncio
    async def test_error_handling_in_generate_signals(self, ma_crossover_config):
        """Test error handling in generate_signals method"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_analyze_symbol', side_effect=Exception("Test error")):
            # Should handle exception gracefully and continue
            signals = await strategy.generate_signals(market_data)
            
            # Should return empty list on error
            assert signals == []

    @pytest.mark.asyncio
    async def test_signal_metadata(self, ma_crossover_config):
        """Test signal metadata content"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_moving_averages') as mock_calc:
            mock_calc.return_value = {
                'current_fast': 51000.0,
                'current_slow': 50000.0,
                'prev_fast': 49900.0,
                'prev_slow': 50000.0,
                'current_price': 51000.0
            }
            
            signals = await strategy.generate_signals(market_data)
            
            signal = signals[0]
            metadata = signal.metadata
            
            assert 'fast_ma' in metadata
            assert 'slow_ma' in metadata
            assert 'fast_period' in metadata
            assert 'slow_period' in metadata
            assert metadata['fast_period'] == 10
            assert metadata['slow_period'] == 30

    @pytest.mark.asyncio
    async def test_async_performance(self, ma_crossover_config):
        """Test that async operations don't block"""
        strategy = MovingAverageCrossover(ma_crossover_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        # Should complete quickly (async operations)
        import time
        start_time = time.time()
        
        signals = await strategy.generate_signals(market_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert execution_time < 1.0