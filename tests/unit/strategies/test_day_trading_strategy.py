"""
Unit tests for Day Trading Strategy
Comprehensive tests for day trading scenarios with realistic market conditions
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, time

from src.strategies.day_trading_strategy import DayTradingStrategy
from src.strategies.base import Signal
from tests.unit.fixtures.historical_snapshots import get_historical_snapshot


def create_mock_indicators(ema_fast=48800.0, ema_medium=48600.0, ema_slow=48400.0, 
                          rsi=65.0, macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
                          volume_ratio=2.5, support=48000.0, resistance=49000.0, atr=100.0):
    """Create mock indicators with proper pandas Series structure"""
    # Create a mock index with enough data points
    index = pd.date_range('2023-01-01', periods=100, freq='h')  # Fix deprecated 'H' 
    
    # Create series with the last value being our target value
    ema_fast_series = pd.Series([ema_fast] * 100, index=index)
    ema_medium_series = pd.Series([ema_medium] * 100, index=index)
    ema_slow_series = pd.Series([ema_slow] * 100, index=index)
    rsi_series = pd.Series([rsi] * 100, index=index)
    macd_line_series = pd.Series([macd_line] * 100, index=index)
    macd_signal_series = pd.Series([macd_signal] * 100, index=index)
    
    # Create MACD histogram with increasing trend (last > previous for momentum_bullish)
    macd_hist_values = [macd_hist - 1.0] * 99 + [macd_hist]  # Previous = 9.0, Current = 10.0
    macd_hist_series = pd.Series(macd_hist_values, index=index)
    
    volume_ratio_series = pd.Series([volume_ratio] * 100, index=index)
    atr_series = pd.Series([atr] * 100, index=index)
    
    return {
        'ema_fast': ema_fast_series,
        'ema_medium': ema_medium_series,
        'ema_slow': ema_slow_series,
        'rsi': rsi_series,
        'macd_line': macd_line_series,
        'macd_signal': macd_signal_series,
        'macd_histogram': macd_hist_series,
        'volume_ratio': volume_ratio_series,
        'support_level': support,  # These are scalars as per the code
        'resistance_level': resistance,
        'atr': atr_series
    }


def create_optimal_bullish_indicators(current_price=50000.0):
    """Create indicators optimized for new scoring system to generate strong BUY signal (score >= 0.8)"""
    return create_mock_indicators(
        # Trend: Full bullish alignment (+0.40)
        ema_fast=current_price + 100,   # 50100
        ema_medium=current_price + 50,  # 50050  
        ema_slow=current_price,         # 50000 (fast > medium > slow)
        
        # Momentum: Optimal RSI + MACD (+0.25) - OPTIMIZED for 40-60 range
        rsi=55.0,                       # Perfect neutral RSI for bullish momentum (+0.20)
        macd_line=50.0, 
        macd_signal=40.0,               # MACD bullish (+0.10)
        macd_hist=10.0,                 # Increasing histogram (+0.03)
        
        # Volume: Strong volume (+0.20)
        volume_ratio=1.5,               # Above 1.2x threshold = strong volume
        
        # Support/Resistance: Near support (+0.10)
        support=current_price - 50,     # 49950 (within 0.8% of 50000)
        resistance=current_price + 500, # 50500 (far from resistance)
        
        # ATR for stops
        atr=100.0
    )
    # Total possible score: 0.40 + 0.15 + 0.10 + 0.20 + 0.10 + 0.03 + 0.02 = 1.0


def create_optimal_bearish_indicators(current_price=50000.0):
    """Create indicators optimized for new scoring system to generate strong SELL signal (score >= 0.8)"""
    return create_mock_indicators(
        # Trend: Full bearish alignment (+0.40)
        ema_fast=current_price - 100,   # 49900
        ema_medium=current_price - 50,  # 49950
        ema_slow=current_price,         # 50000 (fast < medium < slow)
        
        # Momentum: Optimal RSI + MACD (+0.25) - OPTIMIZED for 40-60 range  
        rsi=45.0,                       # Perfect neutral RSI for bearish momentum (+0.20)
        macd_line=40.0,
        macd_signal=50.0,               # MACD bearish (+0.10)
        macd_hist=-10.0,                # Decreasing histogram (+0.03)
        
        # Volume: Strong volume (+0.20)
        volume_ratio=1.5,               # Above 1.2x threshold = strong volume
        
        # Support/Resistance: Near resistance (+0.10)
        support=current_price - 500,    # 49500 (far from support)
        resistance=current_price + 50,  # 50050 (within 0.8% of 50000)
        
        # ATR for stops
        atr=100.0
    )


def create_small_config_for_limited_data():
    """Create config with small periods that work with limited test data"""
    return {
        'fast_ema': 3,
        'medium_ema': 6, 
        'slow_ema': 10,        # Largest EMA period
        'volume_period': 8,    # Volume analysis period  
        'pivot_period': 5,     # Support/resistance period
        'rsi_period': 7,       # RSI period
        'session_start': '08:00',  # Extended trading hours for tests
        'session_end': '18:00'     # Test data goes to 17:00
        # Max required: max(10, 8, 5, 7) + 10 buffer = 20, works with 21+ data points
    }


class TestDayTradingStrategy:
    """Test Day Trading Strategy with comprehensive scenarios"""

    def test_strategy_initialization(self, day_trading_config):
        """Test strategy initialization with valid config"""
        strategy = DayTradingStrategy(day_trading_config)
        
        assert strategy.fast_ema == 8
        assert strategy.medium_ema == 21
        assert strategy.slow_ema == 50
        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30
        assert strategy.max_daily_trades == 3
        assert strategy.stop_loss_pct == 1.5
        assert strategy.take_profit_pct == 2.5
        # OPTIMIZED: Test new default values
        assert strategy.min_signal_score == 0.75  # Increased from 0.5
        assert strategy.strong_signal_score == 0.85  # Increased from 0.8

    def test_validate_parameters_valid(self, day_trading_config):
        """Test parameter validation with valid parameters"""
        strategy = DayTradingStrategy(day_trading_config)
        assert strategy.validate_parameters() is True

    def test_validate_parameters_invalid_emas(self):
        """Test parameter validation with invalid EMA periods"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 21,  # Should be < medium_ema
            'medium_ema': 8,
            'slow_ema': 50,
            'rsi_period': 14,
            'position_size': 0.02
        }
        strategy = DayTradingStrategy.__new__(DayTradingStrategy)
        strategy.fast_ema = 21
        strategy.medium_ema = 8
        strategy.slow_ema = 50
        strategy.rsi_period = 14
        strategy.symbols = ['BTCUSDT']
        
        assert strategy.validate_parameters() is False

    def test_validate_parameters_invalid_rsi(self):
        """Test parameter validation with invalid RSI thresholds"""
        strategy = DayTradingStrategy.__new__(DayTradingStrategy)
        strategy.fast_ema = 8
        strategy.medium_ema = 21
        strategy.slow_ema = 50
        strategy.rsi_period = 14
        strategy.rsi_overbought = 60  # Should be > rsi_oversold
        strategy.rsi_oversold = 70
        strategy.symbols = ['BTCUSDT']
        
        assert strategy.validate_parameters() is False

    @pytest.mark.asyncio
    async def test_morning_breakout_scenario(self, day_trading_config, day_trading_scenarios):
        """Test morning breakout scenario - should generate BUY signal"""
        # Use config optimized for limited test data
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        # Use morning breakout data
        market_data = {'BTCUSDT': day_trading_scenarios['morning_breakout']}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc, \
             patch.object(strategy, '_is_trading_session', return_value=True) as mock_session:
            # Use optimal conditions designed for new scoring system to guarantee BUY signal
            # Morning breakout data has final close price of 49350.0
            mock_calc.return_value = create_optimal_bullish_indicators(current_price=49350.0)
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 1
            signal = signals[0]
            assert signal.action == 'BUY'
            assert signal.confidence >= 0.75  # OPTIMIZED: Must meet new minimum threshold
            assert signal.metadata['stop_loss'] is not None
            assert signal.metadata['take_profit'] is not None

    @pytest.mark.asyncio
    async def test_trend_reversal_scenario(self, day_trading_config, day_trading_scenarios):
        """Test trend reversal at support - should generate BUY signal"""
        # Use config optimized for limited test data
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        market_data = {'BTCUSDT': day_trading_scenarios['trend_reversal']}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc, \
             patch.object(strategy, '_is_trading_session', return_value=True) as mock_session:
            # Mock trend reversal at support using optimal conditions
            # Current price for this scenario is 47300.0 based on test data
            mock_calc.return_value = create_optimal_bullish_indicators(current_price=47300.0)
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 1
            signal = signals[0]
            assert signal.action == 'BUY'
            assert 'support_level' in signal.metadata

    @pytest.mark.asyncio
    async def test_choppy_market_no_signal(self, day_trading_config, day_trading_scenarios):
        """Test choppy market - should NOT generate signals"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': day_trading_scenarios['choppy_market']}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            # Mock choppy/unclear conditions
            mock_calc.return_value = create_mock_indicators(
                ema_fast=47500.0, ema_medium=47480.0, ema_slow=47520.0,
                rsi=50.0, macd_line=2.0, macd_signal=1.5, macd_hist=0.5,
                volume_ratio=0.8, support=47000.0, resistance=48000.0, atr=60.0
            )
            
            signals = await strategy.generate_signals(market_data)
            
            # Should not generate signals in choppy conditions
            assert signals == []

    @pytest.mark.asyncio
    async def test_strong_trend_continuation(self, day_trading_config, day_trading_scenarios):
        """Test strong trend continuation - should generate BUY signal"""
        # Use config optimized for limited test data
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        market_data = {'BTCUSDT': day_trading_scenarios['strong_trend']}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc, \
             patch.object(strategy, '_is_trading_session', return_value=True) as mock_session:
            # Mock strong bullish trend using optimal conditions
            # Current price for this scenario is 48900.0 based on test data
            mock_calc.return_value = create_optimal_bullish_indicators(current_price=48900.0)
            
            signals = await strategy.generate_signals(market_data)
            
            assert len(signals) == 1
            signal = signals[0]
            assert signal.action == 'BUY'
            assert signal.confidence >= 0.85  # OPTIMIZED: Must meet new strong signal threshold

    @pytest.mark.asyncio
    async def test_false_breakout_avoidance(self, day_trading_config, day_trading_scenarios):
        """Test false breakout scenario - should avoid bad signals"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': day_trading_scenarios['false_breakouts']}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            # Mock false breakout conditions (conflicting signals)
            mock_calc.return_value = create_mock_indicators(
                ema_fast=47800.0, ema_medium=47750.0, ema_slow=47700.0,
                rsi=75.0, macd_line=20.0, macd_signal=25.0, macd_hist=-5.0,
                volume_ratio=0.6, support=47000.0, resistance=48000.0, atr=75.0
            )
            
            signals = await strategy.generate_signals(market_data)
            
            # Should avoid signal due to conflicting indicators
            assert signals == []

    @pytest.mark.asyncio
    async def test_overbought_sell_signal(self, day_trading_config):
        """Test sell signal in overbought conditions"""
        # Use config optimized for limited test data
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            with patch.object(strategy, '_is_trading_session', return_value=True):
                # Mock overbought sell conditions using optimal bearish indicators
                # Current price for this scenario is 51000.0 based on test data
                mock_calc.return_value = create_optimal_bearish_indicators(current_price=51000.0)
                
                signals = await strategy.generate_signals(market_data)
                
                assert len(signals) == 1
                signal = signals[0]
                assert signal.action == 'SELL'
                assert signal.confidence >= 0.75  # OPTIMIZED: Must meet new minimum threshold

    @pytest.mark.asyncio
    async def test_session_time_filtering(self, day_trading_config):
        """Test that signals are only generated during trading session"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(strategy, '_is_trading_session') as mock_session:
            with patch.object(strategy, '_calculate_indicators') as mock_calc:
                # Mock favorable conditions but outside session
                mock_session.return_value = False
                mock_calc.return_value = create_mock_indicators(
                    ema_fast=48800.0, ema_medium=48600.0, ema_slow=48400.0,
                    rsi=65.0, macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
                    volume_ratio=2.5, support=48000.0, resistance=49000.0, atr=100.0
                )
                
                signals = await strategy.generate_signals(market_data)
                
                # Should not generate signals outside trading session
                assert signals == []

    @pytest.mark.asyncio
    async def test_daily_trade_limit(self, day_trading_config):
        """Test daily trade limit enforcement"""
        strategy = DayTradingStrategy(day_trading_config)
        
        # Simulate max trades reached
        strategy.daily_trade_count = {'BTCUSDT': 3}  # Max is 3
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            mock_calc.return_value = create_mock_indicators(
                ema_fast=48800.0, ema_medium=48600.0, ema_slow=48400.0,
                rsi=65.0, macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
                volume_ratio=2.5, support=48000.0, resistance=49000.0, atr=100.0
            )
            
            signals = await strategy.generate_signals(market_data)
            
            # Should not generate signals when daily limit reached
            assert signals == []

    @pytest.mark.asyncio
    async def test_stop_loss_take_profit_calculation(self, day_trading_config):
        """Test stop loss and take profit calculation"""
        # Use smaller EMA periods to work with available test data (50 data points)
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('bullish_crossover')}  # Use dataset with enough data
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            with patch.object(strategy, '_is_trading_session', return_value=True):
                # Use optimal conditions to guarantee a BUY signal for this test
                # Current price for this scenario is 50200.0 based on test data
                mock_calc.return_value = create_optimal_bullish_indicators(current_price=50200.0)
                
                signals = await strategy.generate_signals(market_data)
                
                # Verify that a signal was generated for this test to be meaningful
                assert len(signals) > 0, "No signals generated - check test conditions"
                signal = signals[0]
                price = float(signal.price)
                stop_loss = float(signal.metadata['stop_loss'])
                take_profit = float(signal.metadata['take_profit'])
                atr = float(signal.metadata['atr'])
                
                # For BUY signal - strategy uses ATR-based stops, not percentage  
                expected_stop_loss = price - (atr * 2.0)  # ATR-based stop
                expected_take_profit = price + (price * 2.5 / 100)  # 2.5% above (percentage-based take profit)
                
                assert abs(stop_loss - expected_stop_loss) < 1.0
                assert abs(take_profit - expected_take_profit) < 1.0

    @pytest.mark.asyncio
    async def test_volume_confirmation(self, day_trading_config):
        """Test volume confirmation for signals"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            # Test low volume scenario
            mock_calc.return_value = create_mock_indicators(
                ema_fast=48800.0, ema_medium=48600.0, ema_slow=48400.0,
                rsi=65.0, macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
                volume_ratio=0.8, support=48000.0, resistance=49000.0, atr=100.0
            )
            
            signals = await strategy.generate_signals(market_data)
            
            # Should not generate signal without volume confirmation
            assert signals == []

    @pytest.mark.asyncio
    async def test_risk_parameters(self, day_trading_config):
        """Test risk parameter validation and application"""
        strategy = DayTradingStrategy(day_trading_config)
        
        risk_params = strategy.get_risk_parameters()
        
        assert 'max_position_size' in risk_params
        assert 'stop_loss_pct' in risk_params
        assert 'take_profit_pct' in risk_params
        assert 'max_daily_trades' in risk_params
        assert risk_params['stop_loss_pct'] == 1.5
        assert risk_params['take_profit_pct'] == 2.5

    @pytest.mark.asyncio
    async def test_strategy_info(self, day_trading_config):
        """Test strategy information and metadata"""
        strategy = DayTradingStrategy(day_trading_config)
        
        info = strategy.get_strategy_info()
        
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'risk_profile' in info
        assert info['name'] == 'Day Trading Strategy V2'

    def test_indicator_calculations(self, day_trading_config):
        """Test the actual indicator calculations"""
        strategy = DayTradingStrategy(day_trading_config)
        
        # Use real data for calculation test (bullish_crossover has more data points)
        raw_data = get_historical_snapshot('bullish_crossover')
        
        # Convert to DataFrame format expected by the strategy
        df_data = pd.DataFrame(raw_data)
        if 'timestamp' in df_data.columns:
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='ms')
            df_data.set_index('timestamp', inplace=True)
        
        # This would normally call the mocked method, but test the real calculation
        # if the method exists and is testable
        if hasattr(strategy, '_calculate_indicators'):
            result = strategy._calculate_indicators(df_data)
            
            # Should return valid results
            assert isinstance(result, dict)
            assert len(result) > 0
            
            # Check that expected indicators are present
            expected_indicators = ['ema_fast', 'ema_medium', 'ema_slow', 'rsi', 'macd_line', 'atr']
            for indicator in expected_indicators:
                assert indicator in result

    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self):
        """Test processing multiple symbols concurrently"""
        # Use small config to work with limited test data
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'position_size': 0.02
        }
        config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(config)
        
        market_data = {
            'BTCUSDT': get_historical_snapshot('morning_breakout'),
            'ETHUSDT': get_historical_snapshot('eth_sample')
        }
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            with patch.object(strategy, '_is_trading_session', return_value=True):
                # Use optimal conditions to guarantee signals for both symbols
                mock_calc.return_value = create_optimal_bullish_indicators(current_price=48800.0)
                
                signals = await strategy.generate_signals(market_data)
                
                # Should process both symbols
                symbols = [signal.symbol for signal in signals]
                assert 'BTCUSDT' in symbols
                assert 'ETHUSDT' in symbols

    @pytest.mark.asyncio
    async def test_performance_under_load(self, day_trading_config):
        """Test strategy performance with realistic load"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        # Test multiple rapid calls (simulating real-time trading)
        import time
        start_time = time.time()
        
        tasks = []
        for _ in range(10):
            task = strategy.generate_signals(market_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle multiple concurrent calls efficiently
        assert execution_time < 2.0  # All 10 calls in under 2 seconds
        assert len(results) == 10

    # NEW TESTS FOR OPTIMIZATION CHANGES

    @pytest.mark.asyncio
    async def test_optimized_signal_scoring_thresholds(self, day_trading_config):
        """Test that optimized signal scoring prevents weak signals"""
        strategy = DayTradingStrategy(day_trading_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            with patch.object(strategy, '_is_trading_session', return_value=True):
                # Mock weak signal conditions (would have passed old 0.5 threshold but not new 0.75)
                mock_calc.return_value = create_mock_indicators(
                    ema_fast=48600.0, ema_medium=48500.0, ema_slow=48400.0,  # Partial alignment (+0.25)
                    rsi=65.0,                                                  # Good RSI (+0.10)
                    macd_line=45.0, macd_signal=40.0,                         # MACD bullish (+0.10)
                    volume_ratio=1.0,                                         # Normal volume (+0.15)
                    support=48000.0, resistance=49000.0,                      # Not near levels (+0.05)
                    atr=100.0
                )
                # Total score: 0.25 + 0.10 + 0.10 + 0.15 + 0.05 = 0.65 (below 0.75 threshold)
                
                signals = await strategy.generate_signals(market_data)
                
                # Should NOT generate signal due to optimized higher threshold
                assert signals == []

    @pytest.mark.asyncio
    async def test_optimized_rsi_range_preference(self, day_trading_config):
        """Test that optimized RSI logic prefers 40-60 range over extremes"""
        # Use smaller config for limited test data
        test_config = day_trading_config.copy()
        test_config.update(create_small_config_for_limited_data())
        strategy = DayTradingStrategy(test_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(strategy, '_calculate_indicators') as mock_calc:
            with patch.object(strategy, '_is_trading_session', return_value=True):
                # Test optimal RSI in 40-60 range with perfect conditions
                optimal_indicators = create_optimal_bullish_indicators(current_price=49350.0)
                optimal_indicators['rsi'] = pd.Series([55.0] * 100, index=optimal_indicators['rsi'].index)  # Optimal RSI
                mock_calc.return_value = optimal_indicators
                
                signals_optimal = await strategy.generate_signals(market_data)
                
                # Test extreme RSI (overbought) with same other conditions
                extreme_indicators = create_optimal_bullish_indicators(current_price=49350.0) 
                extreme_indicators['rsi'] = pd.Series([75.0] * 100, index=extreme_indicators['rsi'].index)  # Extreme RSI
                mock_calc.return_value = extreme_indicators
                
                signals_extreme = await strategy.generate_signals(market_data)
                
                # Both should generate signals but optimal should have higher confidence
                assert len(signals_optimal) == 1
                optimal_confidence = signals_optimal[0].confidence
                
                if len(signals_extreme) == 1:
                    extreme_confidence = signals_extreme[0].confidence
                    assert optimal_confidence > extreme_confidence, f"Optimal RSI confidence {optimal_confidence} should be higher than extreme RSI confidence {extreme_confidence}"
                else:
                    # If extreme RSI doesn't generate signal, that's also acceptable (stricter threshold working)
                    assert optimal_confidence >= 0.75

    @pytest.mark.asyncio
    async def test_optimized_position_sizing(self):
        """Test that optimized position sizing reduces transaction cost impact"""
        # Test with optimized config (3% position size)
        optimized_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.03,  # Optimized 3%
            'min_signal_score': 0.75,  # Optimized threshold
            'strong_signal_score': 0.85
        }
        optimized_config.update(create_small_config_for_limited_data())
        
        # Test with old config (10% position size)
        old_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.1,   # Old 10%
            'min_signal_score': 0.5,   # Old threshold
            'strong_signal_score': 0.8
        }
        old_config.update(create_small_config_for_limited_data())
        
        optimized_strategy = DayTradingStrategy(optimized_config)
        old_strategy = DayTradingStrategy(old_config)
        
        market_data = {'BTCUSDT': get_historical_snapshot('morning_breakout')}
        
        with patch.object(optimized_strategy, '_calculate_indicators') as mock_calc_opt:
            with patch.object(old_strategy, '_calculate_indicators') as mock_calc_old:
                with patch.object(optimized_strategy, '_is_trading_session', return_value=True):
                    with patch.object(old_strategy, '_is_trading_session', return_value=True):
                        # Use optimal conditions for both
                        optimal_indicators = create_optimal_bullish_indicators(49350.0)
                        mock_calc_opt.return_value = optimal_indicators
                        mock_calc_old.return_value = optimal_indicators
                        
                        signals_opt = await optimized_strategy.generate_signals(market_data)
                        signals_old = await old_strategy.generate_signals(market_data)
                        
                        # Both should generate signals, but optimized has smaller position
                        assert len(signals_opt) == 1
                        assert len(signals_old) == 1
                        
                        # Optimized position should be smaller
                        assert signals_opt[0].quantity < signals_old[0].quantity
                        assert abs(signals_opt[0].quantity - 0.03) < 0.01  # About 3%
                        assert abs(signals_old[0].quantity - 0.1) < 0.01   # About 10%

    def test_bullish_score_calculation_optimization(self, day_trading_config):
        """Test that optimized bullish scoring gives higher scores to neutral RSI"""
        strategy = DayTradingStrategy(day_trading_config)
        
        # Test neutral RSI (optimal range 40-60)
        neutral_score = strategy._calculate_bullish_score(
            current_price=50000.0,
            ema_fast=50100.0, ema_medium=50050.0, ema_slow=50000.0,  # Perfect alignment
            rsi=55.0,  # Optimal neutral RSI
            macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
            volume_ratio=1.5, support=49950.0, resistance=50500.0
        )
        
        # Test extreme RSI (outside preferred range)
        extreme_score = strategy._calculate_bullish_score(
            current_price=50000.0,
            ema_fast=50100.0, ema_medium=50050.0, ema_slow=50000.0,  # Same alignment
            rsi=75.0,  # Overbought RSI
            macd_line=50.0, macd_signal=40.0, macd_hist=10.0,
            volume_ratio=1.5, support=49950.0, resistance=50500.0
        )
        
        # Test that neutral RSI gets higher RSI score component
        # Both should be high scores due to perfect trend alignment, but neutral should be higher
        assert neutral_score > extreme_score, f"Neutral RSI score {neutral_score} should be higher than extreme RSI score {extreme_score}"
        assert neutral_score >= 0.75  # Should meet new minimum threshold
        
        # Test the specific RSI scoring difference
        # Neutral RSI (55) should get: 0.20 (40-60 range with >= 50) + 0.03 (45-55 optimal range)
        # Extreme RSI (75) should get: 0.10 (30-70 backup range with >= 50) + 0.0 (not in 45-55 range)
        # Additional scoring differences may apply, just verify the direction is correct
        actual_diff = neutral_score - extreme_score
        assert abs(actual_diff) >= 0.10, f"Expected significant difference, got only {actual_diff}"

    def test_bearish_score_calculation_optimization(self, day_trading_config):
        """Test that optimized bearish scoring gives higher scores to neutral RSI"""
        strategy = DayTradingStrategy(day_trading_config)
        
        # Test neutral RSI (optimal range 40-60)
        neutral_score = strategy._calculate_bearish_score(
            current_price=50000.0,
            ema_fast=49900.0, ema_medium=49950.0, ema_slow=50000.0,  # Perfect bearish alignment
            rsi=45.0,  # Optimal neutral RSI
            macd_line=40.0, macd_signal=50.0, macd_hist=-10.0,
            volume_ratio=1.5, support=49500.0, resistance=50050.0
        )
        
        # Test extreme RSI
        extreme_score = strategy._calculate_bearish_score(
            current_price=50000.0,
            ema_fast=49900.0, ema_medium=49950.0, ema_slow=50000.0,  # Same alignment
            rsi=25.0,  # Oversold RSI
            macd_line=40.0, macd_signal=50.0, macd_hist=-10.0,
            volume_ratio=1.5, support=49500.0, resistance=50050.0
        )
        
        # Test that neutral RSI gets higher RSI score component
        # Both should be high scores due to perfect trend alignment, but neutral should be higher
        assert neutral_score > extreme_score, f"Neutral RSI score {neutral_score} should be higher than extreme RSI score {extreme_score}"
        assert neutral_score >= 0.75  # Should meet new minimum threshold
        
        # Test the specific RSI scoring difference
        # Neutral RSI (45) should get: 0.15 (40-60 range with <= 50) + 0.03 (45-55 optimal range)
        # Extreme RSI (25) should get: 0.10 (30-70 backup range with <= 50) + 0.0 (not in 45-55 range)
        # Additional scoring differences may apply, just verify the direction is correct
        actual_diff = neutral_score - extreme_score
        assert abs(actual_diff) >= 0.08, f"Expected significant difference, got only {actual_diff}"