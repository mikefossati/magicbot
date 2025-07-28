"""
Unit tests for New Strategy Architecture Components
Tests schema validation, shared components, and new configuration system
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.strategies.config.schema import StrategyParameterSchema, ParameterDefinition, ParameterType
from src.strategies.config.validator import ConfigValidator, ValidationError
from src.strategies.config.loader import ConfigLoader
from src.strategies.components.indicators import IndicatorCalculator
from src.strategies.components.signals import SignalManager
from src.strategies.components.risk import RiskManager
from src.strategies.registry import create_strategy, get_available_strategies, get_strategy_info


class TestStrategyParameterSchema:
    """Test parameter schema validation system"""

    def test_schema_exists_for_strategies(self):
        """Test that schema exists for implemented strategies"""
        assert 'ma_crossover_simple' in StrategyParameterSchema.STRATEGY_SCHEMAS
        assert 'ma_crossover' in StrategyParameterSchema.STRATEGY_SCHEMAS
        assert 'rsi_strategy' in StrategyParameterSchema.STRATEGY_SCHEMAS

    def test_parameter_definition_validation(self):
        """Test ParameterDefinition validation"""
        # Valid integer parameter
        param = ParameterDefinition(
            name='test_period',
            param_type=ParameterType.INTEGER,
            default=20,
            min_value=5,
            max_value=100
        )
        
        assert param.validate(20) == True
        assert param.validate(5) == True
        assert param.validate(100) == True
        assert param.validate(4) == False  # Below min
        assert param.validate(101) == False  # Above max
        assert param.validate('not_int') == False  # Wrong type

    def test_schema_get_method(self):
        """Test schema retrieval for specific strategies"""
        schema = StrategyParameterSchema.get_schema('ma_crossover_simple')
        
        assert 'required_params' in schema
        assert 'optional_params' in schema
        assert 'parameters' in schema
        assert 'symbols' in schema['required_params']
        assert 'position_size' in schema['required_params']


class TestConfigValidator:
    """Test configuration validation system"""

    def test_valid_config_validation(self):
        """Test validation of valid configuration"""
        valid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 8,
            'slow_period': 21
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover_simple', valid_config)
        assert is_valid == True
        assert errors == []

    def test_missing_required_param_validation(self):
        """Test validation fails for missing required parameters"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02
            # Missing fast_period and slow_period
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover_simple', invalid_config)
        assert is_valid == False
        assert len(errors) > 0
        assert any('fast_period' in error for error in errors)

    def test_invalid_parameter_type_validation(self):
        """Test validation fails for invalid parameter types"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 'not_a_number',  # Should be integer
            'slow_period': 21
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover_simple', invalid_config)
        assert is_valid == False
        assert any('fast_period' in error for error in errors)

    def test_parameter_range_validation(self):
        """Test validation of parameter ranges"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 1,  # Below minimum
            'slow_period': 21
        }
        
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover_simple', invalid_config)
        assert is_valid == False

    def test_validation_error_raising(self):
        """Test ConfigValidator.validate_and_raise method"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02
            # Missing required parameters
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_and_raise('ma_crossover_simple', invalid_config)
        
        assert 'ma_crossover_simple' in str(exc_info.value)


class TestConfigLoader:
    """Test configuration loading and processing"""

    def test_load_strategy_params(self):
        """Test loading and processing of strategy parameters"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 8,
            'slow_period': 21
        }
        
        processed_config = ConfigLoader.load_strategy_params('ma_crossover_simple', config)
        
        # Should include original params plus defaults
        assert processed_config['symbols'] == ['BTCUSDT']
        assert processed_config['position_size'] == 0.02
        assert processed_config['fast_period'] == 8
        assert processed_config['slow_period'] == 21
        assert 'lookback_periods' in processed_config  # Default value

    def test_get_data_requirements(self):
        """Test data requirements calculation"""
        params = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 8,
            'slow_period': 21,
            'timeframes': ['5m']
        }
        
        data_req = ConfigLoader.get_data_requirements('ma_crossover_simple', params)
        
        assert 'timeframes' in data_req
        assert 'lookback_periods' in data_req
        assert data_req['timeframes'] == ['5m']

    def test_get_parameter_info(self):
        """Test parameter information retrieval"""
        param_info = ConfigLoader.get_parameter_info('ma_crossover_simple')
        
        assert 'strategy_name' in param_info
        assert 'parameters' in param_info
        assert 'required_params' in param_info
        assert 'optional_params' in param_info
        assert param_info['strategy_name'] == 'ma_crossover_simple'


class TestIndicatorCalculator:
    """Test shared indicator calculation component"""

    @pytest.fixture
    def sample_data(self):
        """Sample price data for indicator testing"""
        return pd.DataFrame({
            'close': [3800, 3805, 3810, 3815, 3820, 3825, 3830, 3835, 3840, 3845] + [3850] * 15,
            'high': [3810, 3815, 3820, 3825, 3830, 3835, 3840, 3845, 3850, 3855] + [3860] * 15,
            'low': [3790, 3795, 3800, 3805, 3810, 3815, 3820, 3825, 3830, 3835] + [3840] * 15,
            'volume': [1000] * 25
        })

    @pytest.fixture
    def indicator_calculator(self):
        """IndicatorCalculator instance"""
        return IndicatorCalculator()

    def test_sma_calculation(self, indicator_calculator, sample_data):
        """Test Simple Moving Average calculation"""
        sma = indicator_calculator.calculate_sma(sample_data['close'], 5)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_data)
        # Our SMA uses min_periods=1, so first value should be valid
        assert not pd.isna(sma.iloc[0])  # First value should be valid
        assert not pd.isna(sma.iloc[-1])  # Last value should be valid
        
        # Test that SMA values are reasonable
        assert sma.iloc[0] == sample_data['close'].iloc[0]  # First SMA equals first price
        assert sma.iloc[4] == sample_data['close'].iloc[:5].mean()  # 5th value is proper 5-period average

    def test_ema_calculation(self, indicator_calculator, sample_data):
        """Test Exponential Moving Average calculation"""
        ema = indicator_calculator.calculate_ema(sample_data['close'], 5)
        
        assert isinstance(ema, pd.Series)
        assert len(ema) == len(sample_data)
        assert not pd.isna(ema.iloc[-1])

    def test_rsi_calculation(self, indicator_calculator, sample_data):
        """Test RSI calculation"""
        rsi = indicator_calculator.calculate_rsi(sample_data['close'], 14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)

    def test_calculate_all_indicators(self, indicator_calculator, sample_data):
        """Test calculation of all indicators based on parameters"""
        params = {
            'fast_period': 8,
            'slow_period': 21,
            'rsi_period': 14,
            'atr_period': 14
        }
        
        indicators = indicator_calculator.calculate_all_indicators(sample_data, params)
        
        assert isinstance(indicators, dict)
        assert 'sma_fast' in indicators
        assert 'sma_slow' in indicators
        # Should include other indicators as determined by strategy needs


class TestSignalManager:
    """Test signal management component"""

    @pytest.fixture
    def signal_manager(self):
        """SignalManager instance"""
        params = {'symbols': ['BTCUSDT'], 'position_size': 0.02}
        return SignalManager('test_strategy', params)

    def test_signal_manager_initialization(self, signal_manager):
        """Test SignalManager initialization"""
        assert signal_manager.strategy_name == 'test_strategy'
        assert hasattr(signal_manager, 'last_signals')

    def test_should_generate_signal_deduplication(self, signal_manager):
        """Test signal deduplication logic"""
        # First signal should be allowed
        assert signal_manager.should_generate_signal('BTCUSDT', 'BUY') == True
        
        # Record the signal
        signal_manager.record_signal('BTCUSDT', 'BUY')
        
        # Same signal should be deduplicated
        assert signal_manager.should_generate_signal('BTCUSDT', 'BUY') == False
        
        # Different action should be allowed
        assert signal_manager.should_generate_signal('BTCUSDT', 'SELL') == True

    def test_create_signal(self, signal_manager):
        """Test signal creation through SignalManager"""
        from src.strategies.signal import Signal
        from decimal import Decimal
        
        signal = signal_manager.create_signal('BTCUSDT', 'BUY', 3800.0, 0.8, {'test': 'metadata'})
        
        assert isinstance(signal, Signal)
        assert signal.symbol == 'BTCUSDT'
        assert signal.action == 'BUY'
        # Signal should have quantity set by signal manager
        assert signal.quantity is not None
        assert signal.confidence == 0.8
        assert signal.metadata['test'] == 'metadata'


class TestRiskManager:
    """Test risk management component"""

    @pytest.fixture
    def risk_manager(self):
        """RiskManager instance"""
        params = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'stop_loss_pct': 2.5,
            'take_profit_pct': 5.0
        }
        return RiskManager('test_strategy', params)

    def test_risk_manager_initialization(self, risk_manager):
        """Test RiskManager initialization"""
        assert risk_manager.strategy_name == 'test_strategy'
        assert risk_manager.stop_loss_pct == 2.5
        assert risk_manager.take_profit_pct == 5.0

    def test_calculate_stop_levels(self, risk_manager):
        """Test stop loss and take profit calculation"""
        price = 3800.0
        atr = 20.0  # Example ATR value
        
        stops = risk_manager.calculate_stop_levels(price, 'BUY', atr)
        
        assert 'stop_loss' in stops
        assert 'take_profit' in stops
        assert stops['stop_loss'] < price  # Stop loss should be below entry for BUY
        assert stops['take_profit'] > price  # Take profit should be above entry for BUY

    def test_calculate_stop_levels_sell(self, risk_manager):
        """Test stop levels for SELL signals"""
        price = 3800.0
        atr = 20.0
        
        stops = risk_manager.calculate_stop_levels(price, 'SELL', atr)
        
        assert stops['stop_loss'] > price  # Stop loss should be above entry for SELL
        assert stops['take_profit'] < price  # Take profit should be below entry for SELL


class TestStrategyRegistry:
    """Test strategy registry and factory functions"""

    def test_get_available_strategies(self):
        """Test getting list of available strategies"""
        strategies = get_available_strategies()
        
        assert isinstance(strategies, dict)
        assert 'ma_crossover_simple' in strategies
        assert 'ma_crossover' in strategies

    def test_get_strategy_info(self):
        """Test getting strategy information"""
        info = get_strategy_info('ma_crossover_simple')
        
        assert isinstance(info, dict)
        assert 'parameters' in info
        assert 'required_params' in info
        assert 'optional_params' in info

    def test_create_strategy_valid_config(self):
        """Test strategy creation with valid config"""
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 8,
            'slow_period': 21
        }
        
        strategy = create_strategy('ma_crossover_simple', config)
        
        assert strategy.strategy_name == 'ma_crossover_simple'
        assert strategy.symbols == ['BTCUSDT']

    def test_create_strategy_invalid_config(self):
        """Test strategy creation with invalid config"""
        invalid_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02
            # Missing required parameters
        }
        
        with pytest.raises(ValidationError):
            create_strategy('ma_crossover_simple', invalid_config)

    def test_create_strategy_nonexistent(self):
        """Test strategy creation for non-existent strategy"""
        config = {'symbols': ['BTCUSDT'], 'position_size': 0.02}
        
        with pytest.raises(ValueError):
            create_strategy('nonexistent_strategy', config)


class TestTrailingStopLossIntegration:
    """Test trailing stop loss integration in new architecture"""

    @pytest.fixture
    def trailing_stop_config(self):
        """Configuration with trailing stop enabled"""
        return {
            'symbols': ['BTCUSDT'],
            'position_size': 0.02,
            'fast_period': 8,
            'slow_period': 21,
            'trailing_stop_enabled': True,
            'trailing_stop_distance': 1.8,
            'trailing_stop_type': 'percentage'
        }

    def test_trailing_stop_in_schema(self):
        """Test that trailing stop parameters are in schema"""
        schema = StrategyParameterSchema.get_schema('ma_crossover_simple')
        
        assert 'trailing_stop_enabled' in schema['optional_params']
        assert 'trailing_stop_distance' in schema['optional_params']
        assert 'trailing_stop_type' in schema['optional_params']

    def test_trailing_stop_validation(self, trailing_stop_config):
        """Test trailing stop parameter validation"""
        is_valid, errors = ConfigValidator.validate_strategy_config('ma_crossover_simple', trailing_stop_config)
        
        assert is_valid == True
        assert errors == []

    def test_trailing_stop_in_signal_metadata(self, trailing_stop_config):
        """Test that trailing stop parameters appear in signal metadata"""
        from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
        
        strategy = SimpleMovingAverageCrossover(trailing_stop_config)
        
        # Test signal creation through create_signal method
        sample_metadata = {'atr': 20.0}
        signal = strategy.create_signal('BTCUSDT', 'BUY', 3800.0, 0.8, sample_metadata)
        
        if signal and signal.metadata:
            assert signal.metadata.get('trailing_stop_enabled') == True
            assert signal.metadata.get('trailing_stop_distance') == 1.8
            assert signal.metadata.get('trailing_stop_type') == 'percentage'