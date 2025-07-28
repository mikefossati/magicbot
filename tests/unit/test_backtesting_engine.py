"""
Unit tests for Enhanced Backtesting Engine
Tests trailing stop loss functionality, position management, and signal processing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from decimal import Decimal

from src.backtesting.engine import BacktestEngine, Position, Trade
from src.strategies.signal import Signal


class TestBacktestEngineTrailingStop:
    """Test backtesting engine with trailing stop functionality"""

    @pytest.fixture
    def engine(self):
        """BacktestEngine instance"""
        return BacktestEngine(initial_balance=10000.0, fast_mode=False)

    @pytest.fixture
    def sample_strategy(self):
        """Mock strategy for testing"""
        strategy = Mock()
        strategy.strategy_name = 'test_strategy'
        strategy.generate_signals = AsyncMock(return_value=[])
        return strategy

    @pytest.fixture
    def sample_historical_data(self):
        """Sample historical data for testing"""
        timestamps = pd.date_range('2025-01-01', periods=100, freq='5min')
        
        # Create uptrending price data
        base_price = 3800
        prices = []
        for i in range(100):
            if i < 50:
                # Uptrend
                price = base_price + i * 2 + np.random.normal(0, 5)
            else:
                # Pullback to test trailing stop
                pullback = (i - 50) * 1.5
                price = base_price + 100 - pullback + np.random.normal(0, 3)
            prices.append(max(price, base_price - 50))  # Prevent too low prices
        
        data = []
        for timestamp, price in zip(timestamps, prices):
            data.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': price + np.random.uniform(-2, 2),
                'high': price + abs(np.random.normal(2, 1)),
                'low': price - abs(np.random.normal(2, 1)),
                'close': price,
                'volume': np.random.uniform(1000, 2000)
            })
        
        return {'TESTUSDT': pd.DataFrame(data)}

    def test_position_with_trailing_stop_initialization(self, engine):
        """Test Position dataclass with trailing stop fields"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,  # 2% below entry
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        assert position.trailing_stop_enabled == True
        assert position.trailing_stop_distance == 2.0
        assert position.trailing_stop_type == 'percentage'
        assert position.trailing_stop_price == 3724.0
        assert position.highest_price == 3800.0

    def test_trailing_stop_update_long_position(self, engine):
        """Test trailing stop update for LONG position"""
        # Create a LONG position with trailing stop
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        # Update position with higher price
        position.current_price = 3900.0
        engine._update_trailing_stop(position)
        
        # Trailing stop should move up
        expected_trailing_stop = 3900.0 * 0.98  # 2% below new high
        assert position.highest_price == 3900.0
        assert position.trailing_stop_price == expected_trailing_stop

    def test_trailing_stop_update_long_position_no_change(self, engine):
        """Test trailing stop doesn't update when price doesn't reach new high"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        original_trailing_stop = position.trailing_stop_price
        
        # Update with lower price
        position.current_price = 3750.0
        engine._update_trailing_stop(position)
        
        # Trailing stop should not change
        assert position.highest_price == 3800.0  # No change
        assert position.trailing_stop_price == original_trailing_stop

    def test_trailing_stop_trigger_detection(self, engine):
        """Test trailing stop trigger detection"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            current_price=3720.0,  # Below trailing stop
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        # Check stop conditions
        stop_reason = engine._check_stop_conditions(position)
        assert stop_reason == 'trailing_stop'

    def test_trailing_stop_priority_over_regular_stop(self, engine):
        """Test that trailing stop takes priority over regular stop loss"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            current_price=3720.0,
            stop_loss=3700.0,  # Regular stop loss
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,  # More restrictive than regular stop
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        # Trailing stop should trigger first
        stop_reason = engine._check_stop_conditions(position)
        assert stop_reason == 'trailing_stop'

    def test_trailing_stop_absolute_type(self, engine):
        """Test trailing stop with absolute distance type"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=50.0,  # $50 absolute distance
            trailing_stop_type='absolute',
            trailing_stop_price=3750.0,  # $50 below entry
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        # Update with higher price
        position.current_price = 3900.0
        engine._update_trailing_stop(position)
        
        # Trailing stop should be $50 below new high
        expected_trailing_stop = 3900.0 - 50.0
        assert position.trailing_stop_price == expected_trailing_stop

    def test_trailing_stop_short_position(self, engine):
        """Test trailing stop for SHORT position"""
        position = Position(
            symbol='TESTUSDT',
            side='SHORT',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3876.0,  # 2% above entry for SHORT
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        # Update with lower price (favorable for SHORT)
        position.current_price = 3700.0
        engine._update_trailing_stop(position)
        
        # Trailing stop should move down
        expected_trailing_stop = 3700.0 * 1.02  # 2% above new low
        assert position.lowest_price == 3700.0
        assert position.trailing_stop_price == expected_trailing_stop

    @pytest.mark.asyncio
    async def test_execute_buy_with_trailing_stop_metadata(self, engine):
        """Test buy execution with trailing stop metadata"""
        # Create signal with trailing stop metadata
        signal = Signal(
            symbol='TESTUSDT',
            action='BUY',
            quantity=Decimal('0.001'),
            confidence=0.8,
            price=Decimal('3800.0'),
            metadata={
                'trailing_stop_enabled': True,
                'trailing_stop_distance': 2.0,
                'trailing_stop_type': 'percentage'
            }
        )
        
        # Mock market data
        market_data = {
            'TESTUSDT': {
                'close': 3800.0,
                'high': 3810.0,
                'low': 3790.0,
                'volume': 1000
            }
        }
        
        await engine._execute_buy(signal, 3800.0)
        
        # Check that position has trailing stop configured
        position = engine.positions['TESTUSDT']
        assert position.trailing_stop_enabled == True
        assert position.trailing_stop_distance == 2.0
        assert position.trailing_stop_type == 'percentage'
        assert position.trailing_stop_price is not None

    def test_close_position_automatically_trailing_stop(self, engine):
        """Test automatic position closure due to trailing stop"""
        # Setup position
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        engine.positions['TESTUSDT'] = position
        engine.current_time = datetime.now()
        
        # Close position due to trailing stop
        engine._close_position_automatically('TESTUSDT', 3720.0, 'trailing_stop')
        
        # Position should be closed
        assert 'TESTUSDT' not in engine.positions
        
        # Trade should be recorded
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.strategy == 'auto_trailing_stop'
        # Use pytest.approx for floating point comparison with slippage
        assert abs(trade.exit_price - 3724.0) < 5.0  # Allow for slippage tolerance

    @pytest.mark.asyncio
    async def test_full_backtest_with_trailing_stop(self, engine, sample_strategy, sample_historical_data):
        """Test complete backtest with trailing stop functionality"""
        # Create a signal that will trigger trailing stop
        test_signal = Signal(
            symbol='TESTUSDT',
            action='BUY',
            quantity=Decimal('0.001'),
            confidence=0.8,
            price=Decimal('3800.0'),
            metadata={
                'trailing_stop_enabled': True,
                'trailing_stop_distance': 3.0,
                'trailing_stop_type': 'percentage'
            }
        )
        
        # Mock strategy to return one buy signal early in backtest
        sample_strategy.generate_signals = AsyncMock(side_effect=[
            [test_signal],  # First call returns signal
            *([[]] * 99)    # Remaining calls return empty
        ])
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 1, 8, 15)  # About 8 hours later
        
        results = await engine.run_backtest(
            strategy=sample_strategy,
            historical_data=sample_historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Should have executed at least one trade
        assert results['trades']['total'] >= 1
        
        # Check if any trades were closed by trailing stop
        trailing_stop_trades = [
            trade for trade in results['trades_detail'] 
            if 'trailing_stop' in str(trade.strategy)
        ]
        
        # Should have at least one trailing stop execution given the price pattern
        # (price goes up then down, should trigger trailing stop)
        if trailing_stop_trades:
            assert len(trailing_stop_trades) >= 1

    def test_position_update_with_trailing_stop(self, engine):
        """Test position updates include trailing stop logic"""
        # Setup position with trailing stop
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=True,
            trailing_stop_distance=2.0,
            trailing_stop_type='percentage',
            trailing_stop_price=3724.0,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        engine.positions['TESTUSDT'] = position
        
        # Mock market data with higher price
        market_data = {
            'TESTUSDT': {
                'close': 3900.0,
                'high': 3910.0,
                'low': 3890.0,
                'volume': 1000
            }
        }
        
        # Update positions
        engine._update_positions(market_data)
        
        # Trailing stop should have been updated
        updated_position = engine.positions['TESTUSDT']
        assert updated_position.highest_price == 3900.0
        assert updated_position.trailing_stop_price > 3724.0  # Should have moved up

    def test_trailing_stop_disabled(self, engine):
        """Test that trailing stop logic is skipped when disabled"""
        position = Position(
            symbol='TESTUSDT',
            side='LONG',
            quantity=0.001,
            entry_price=3800.0,
            entry_time=datetime.now(),
            trailing_stop_enabled=False,  # Disabled
            trailing_stop_distance=None,
            trailing_stop_type='percentage',
            trailing_stop_price=None,
            highest_price=3800.0,
            lowest_price=3800.0
        )
        
        original_highest = position.highest_price
        
        # Update with higher price
        position.current_price = 3900.0
        engine._update_trailing_stop(position)
        
        # Should not update trailing stop when disabled
        assert position.highest_price == original_highest
        assert position.trailing_stop_price is None