"""
Test suite for the new strategy architecture.

This module tests the centralized configuration system, shared components,
and ensures all strategies work with the new architecture.
"""

import pytest
import yaml
from decimal import Decimal
from typing import Dict, Any
import pandas as pd
import numpy as np

from src.strategies.config import ConfigLoader, ConfigValidator, ValidationError
from src.strategies.registry import create_strategy, validate_strategy_config
from src.strategies.components import IndicatorCalculator, SignalManager, RiskManager

class TestParameterValidation:
    """Test parameter validation and configuration loading"""
    
    def test_ma_crossover_validation_success(self):
        """Test valid MA crossover configuration"""
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'position_size': 0.01,
            'fast_period': 10,
            'slow_period': 30
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover', config)
        assert is_valid, f"Validation errors: {errors}"
    
    def test_ma_crossover_validation_failure(self):
        """Test invalid MA crossover configuration"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'fast_period': 30,  # Invalid: fast >= slow
            'slow_period': 10
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover', config)
        assert not is_valid
        assert any('fast_period must be less than slow_period' in error for error in errors)
    
    def test_rsi_strategy_validation_success(self):
        """Test valid RSI strategy configuration"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('rsi_strategy', config)
        assert is_valid, f"Validation errors: {errors}"
    
    def test_missing_required_parameters(self):
        """Test validation with missing required parameters"""
        config = {
            'position_size': 0.01,
            # Missing symbols
            'fast_period': 10,
            'slow_period': 30
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover', config)
        assert not is_valid
        assert any('Missing required parameter: symbols' in error for error in errors)
    
    def test_config_loader_applies_defaults(self):
        """Test that ConfigLoader applies default values"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'fast_period': 10,
            'slow_period': 30
        }
        
        processed_config = ConfigLoader.load_strategy_params('ma_crossover', config)
        
        # Should have defaults applied
        assert 'timeframes' in processed_config
        assert processed_config['timeframes'] == ['1h']  # Default value
        assert 'stop_loss_pct' in processed_config
        assert processed_config['stop_loss_pct'] == 2.0  # Default value

class TestIndicatorCalculator:
    """Test shared indicator calculations"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.2),
            'low': prices - np.abs(np.random.randn(100) * 0.2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.calculator = IndicatorCalculator()
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.calculator.calculate_rsi(self.data['close'], period=14)
        
        assert len(rsi) == len(self.data)
        assert not rsi.isna().all()  # Should have some valid values
        assert (rsi <= 100).all() and (rsi >= 0).all()  # RSI should be 0-100
    
    def test_moving_averages(self):
        """Test moving average calculations"""
        sma = self.calculator.calculate_sma(self.data['close'], period=10)
        ema = self.calculator.calculate_ema(self.data['close'], period=10)
        
        assert len(sma) == len(self.data)
        assert len(ema) == len(self.data)
        
        # EMA should respond faster than SMA
        assert not sma.equals(ema)
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_data = self.calculator.calculate_macd(self.data['close'], fast=12, slow=26, signal=9)
        
        required_keys = ['macd_line', 'signal_line', 'histogram']
        for key in required_keys:
            assert key in macd_data
            assert len(macd_data[key]) == len(self.data)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb_data = self.calculator.calculate_bollinger_bands(self.data['close'], period=20, std_dev=2.0)
        
        required_keys = ['upper', 'middle', 'lower']
        for key in required_keys:
            assert key in bb_data
            assert len(bb_data[key]) == len(self.data)
        
        # Upper should be above middle, middle above lower (exclude NaN values)
        valid_idx = ~bb_data['upper'].isna()
        assert (bb_data['upper'][valid_idx] >= bb_data['middle'][valid_idx]).all()
        assert (bb_data['middle'][valid_idx] >= bb_data['lower'][valid_idx]).all()

class TestStrategyCreation:
    """Test strategy creation with new architecture"""
    
    def test_ma_crossover_creation(self):
        """Test MA crossover strategy creation"""
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'position_size': 0.01,
            'fast_period': 10,
            'slow_period': 30,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0
        }
        
        strategy = create_strategy('ma_crossover', config)
        
        assert strategy.strategy_name == 'ma_crossover'
        assert strategy.symbols == ['BTCUSDT', 'ETHUSDT']
        assert strategy.position_size == Decimal('0.01')
        
        # Test parameter access
        assert strategy.get_parameter_value('fast_period') == 10
        assert strategy.get_parameter_value('slow_period') == 30
        
        # Test components initialization
        assert strategy.indicator_calculator is not None
        assert strategy.signal_manager is not None
        assert strategy.risk_manager is not None
    
    def test_rsi_strategy_creation(self):
        """Test RSI strategy creation"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        strategy = create_strategy('rsi_strategy', config)
        
        assert strategy.strategy_name == 'rsi_strategy'
        assert strategy.get_parameter_value('rsi_period') == 14
        assert strategy.get_parameter_value('rsi_oversold') == 30
    
    def test_invalid_strategy_creation(self):
        """Test creation with invalid configuration"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'fast_period': 30,  # Invalid
            'slow_period': 10   # Invalid
        }
        
        with pytest.raises(ValidationError):
            create_strategy('ma_crossover', config)
    
    def test_unknown_strategy(self):
        """Test creation of unknown strategy"""
        config = {'symbols': ['BTCUSDT'], 'position_size': 0.01}
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy('unknown_strategy', config)

class TestSignalManager:
    """Test signal management functionality"""
    
    def setup_method(self):
        """Setup signal manager"""
        params = {
            'position_size': 0.01,
            'enable_signal_deduplication': True,
            'min_signal_confidence': 0.6
        }
        self.signal_manager = SignalManager('test_strategy', params)
    
    def test_signal_deduplication(self):
        """Test signal deduplication logic"""
        # First signal should be allowed
        assert self.signal_manager.should_generate_signal('BTCUSDT', 'BUY')
        
        # Record the signal
        self.signal_manager.record_signal('BTCUSDT', 'BUY')
        
        # Same signal should be blocked
        assert not self.signal_manager.should_generate_signal('BTCUSDT', 'BUY')
        
        # Different action should be allowed
        assert self.signal_manager.should_generate_signal('BTCUSDT', 'SELL')
    
    def test_signal_creation(self):
        """Test signal creation"""
        signal = self.signal_manager.create_signal(
            symbol='BTCUSDT',
            action='BUY',
            price=50000.0,
            confidence=0.8,
            metadata={'test': 'data'}
        )
        
        assert signal.symbol == 'BTCUSDT'
        assert signal.action == 'BUY'
        assert signal.confidence == 0.8
        assert signal.metadata['strategy'] == 'test_strategy'
        assert signal.metadata['test'] == 'data'

class TestRiskManager:
    """Test risk management functionality"""
    
    def setup_method(self):
        """Setup risk manager"""
        params = {
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'position_size': 0.01,
            'max_position_size': 0.1,
            'use_leverage': False,
            'leverage': 1.0
        }
        self.risk_manager = RiskManager('test_strategy', params)
    
    def test_position_sizing(self):
        """Test position size calculation"""
        position_size = self.risk_manager.calculate_position_size(
            current_price=50000.0,
            confidence=0.8
        )
        
        assert position_size == 0.01  # Base position size
    
    def test_stop_levels_calculation(self):
        """Test stop loss and take profit calculation"""
        stops = self.risk_manager.calculate_stop_levels(
            current_price=50000.0,
            action='BUY'
        )
        
        assert 'stop_loss' in stops
        assert 'take_profit' in stops
        
        # For BUY: stop_loss < current_price < take_profit
        assert stops['stop_loss'] < 50000.0
        assert stops['take_profit'] > 50000.0
    
    def test_risk_validation(self):
        """Test risk parameter validation"""
        is_valid, errors = self.risk_manager.validate_risk_parameters()
        assert is_valid, f"Risk validation errors: {errors}"

class TestDataRequirements:
    """Test data requirements calculation"""
    
    def test_ma_crossover_data_requirements(self):
        """Test MA crossover data requirements"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'fast_period': 10,
            'slow_period': 30
        }
        
        processed_config = ConfigLoader.load_strategy_params('ma_crossover', config)
        data_req = ConfigLoader.get_data_requirements('ma_crossover', processed_config)
        
        assert 'timeframes' in data_req
        assert 'lookback_periods' in data_req
        assert 'indicators' in data_req
        
        # Should need at least slow_period + buffer
        assert data_req['lookback_periods'] >= 30
        
        # Should require SMA indicators
        assert 'sma' in data_req['indicators']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])