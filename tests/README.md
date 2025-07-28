# Trading Strategy Test Suite

Comprehensive unit and integration tests for the new trading strategy architecture in the Magicbot system. The test suite validates schema-based configuration, shared components, and trailing stop functionality.

## 🏗️ Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── run_tests.py                    # Test runner script
├── unit/                           # Fast tests with mocked data
│   ├── fixtures/
│   │   └── historical_snapshots.py # Real historical market data
│   ├── strategies/
│   │   ├── test_new_architecture.py # New strategy architecture tests
│   │   ├── test_ma_crossover.py    # MA crossover strategy (new architecture)
│   │   └── legacy/                 # Deprecated legacy tests
│   │       ├── test_momentum_trading_strategy.py
│   │       └── test_vlam_consolidation_strategy.py
│   └── test_backtesting_engine.py  # Enhanced backtesting with trailing stops
├── integration/                    # Tests with real Binance testnet
│   ├── test_strategy_integration.py # End-to-end strategy tests
│   └── test_exchange_integration.py # Exchange API tests
└── performance/                    # Performance and latency tests
    └── test_latency.py            # Low latency requirements
```

## 🚀 Quick Start

### Run Unit Tests (Fast)
```bash
# Run all unit tests (excluding legacy)
python tests/run_tests.py --mode unit

# Run with coverage
python tests/run_tests.py --mode unit --coverage

# Run specific strategy (new architecture)
python tests/run_tests.py --strategy ma_crossover_simple

# Run new architecture tests only
python -m pytest tests/unit/strategies/test_new_architecture.py tests/unit/strategies/test_ma_crossover.py tests/unit/test_backtesting_engine.py
```

### Run Integration Tests (Requires API Keys)
```bash
# Set up environment using .env file
cp .env.example .env
# Edit .env file with your Binance testnet credentials

# Run integration tests
python tests/run_tests.py --mode integration

# Run performance tests
python tests/run_tests.py --mode performance
```

### Run Complete Suite
```bash
# Run everything
python tests/run_tests.py --mode all

# Skip integration tests (for development)
python tests/run_tests.py --mode all --skip-integration
```

## 📋 Test Categories

### Unit Tests (Fast - < 500ms total)
- **New Architecture Testing**: Schema validation, shared components
- **Strategy Logic Testing**: Signal generation with new configuration system
- **Parameter Validation**: Schema-based validation, type checking
- **Risk Management**: Stop loss, take profit, trailing stop calculations
- **Shared Components**: IndicatorCalculator, SignalManager, RiskManager
- **Mock Data Scenarios**: Bull/bear markets, volatile conditions
- **Concurrent Processing**: Multiple symbols, async performance
- **Backtesting Engine**: Enhanced engine with trailing stop functionality

### Integration Tests (Manual - Real API)
- **Real Market Data**: Live Binance testnet integration
- **End-to-End Workflows**: Data fetch → analysis → signal generation
- **Low Latency Validation**: < 100ms average, < 200ms max
- **Connection Stability**: Extended use, error recovery
- **Memory Usage**: No memory leaks over time

### New Architecture Scenarios
- **Schema Validation**: Parameter type checking, range validation
- **Configuration Loading**: YAML-based configuration processing
- **Shared Component Integration**: Indicator calculation, risk management
- **Trailing Stop Logic**: Dynamic stop adjustments, position management
- **Legacy Strategy Migration**: Testing compatibility and deprecation

## 🧪 Test Data Strategy

### Historical Snapshots
Uses real historical market data snapshots for deterministic testing:

```python
# Available scenarios for new architecture testing
'bullish_crossover'    # MA crossover upward trend (new architecture)
'bearish_crossover'    # MA crossover downward trend (new architecture)
'volatile_market'      # High volatility, choppy conditions
'morning_breakout'     # Day trading breakout pattern
'eth_sample'           # ETH data for multi-symbol tests

# Sample data format (OHLCV with timestamps)
sample_data = {
    'timestamp': int,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float
}
```

### New Architecture Testing Strategy
- **Schema-Based Testing**: Configuration validation before strategy creation
- **Shared Component Testing**: IndicatorCalculator, SignalManager, RiskManager
- **Deep Mocking**: Exchange APIs, network calls mocked
- **Strategy Logic**: Fully tested with real calculation logic using shared components
- **Indicator Math**: Pandas operations tested with real data through IndicatorCalculator
- **Edge Cases**: NaN handling, insufficient data, errors
- **Trailing Stop Testing**: Position management, dynamic stop adjustments

## 📊 Performance Requirements

### Latency Targets
- **Average Signal Generation**: < 100ms
- **Maximum Latency**: < 200ms  
- **Multi-Symbol Processing**: < 500ms for 2 symbols
- **Memory Growth**: < 10MB over 50 iterations

### Test Execution Speed
- **Unit Tests**: Complete suite < 5 seconds
- **Integration Tests**: < 60 seconds
- **Performance Tests**: < 30 seconds

## 🔧 Configuration

### Environment Setup (.env file)
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET=your_testnet_secret

# Optional settings
LOG_LEVEL=INFO
TEST_TIMEOUT=30
DEBUG=true
```

### pytest Configuration
See `pytest.ini` for:
- Async test support
- Custom markers (unit, integration, performance)
- Coverage configuration
- Warning filters

## 🧭 Running Specific Tests

### By Strategy
```bash
# New architecture strategies
python tests/run_tests.py --strategy ma_crossover_simple
python -m pytest tests/unit/strategies/test_ma_crossover.py

# Legacy strategies (deprecated)
python -m pytest tests/unit/strategies/legacy/
```

### By Scenario
```bash
# New architecture specific tests
python -m pytest -k "new_architecture"
python -m pytest -k "trailing_stop" 
python -m pytest -k "schema"

# General scenarios
python -m pytest -k "crossover"
python -m pytest -k "latency"
```

### By Marker
```bash
python -m pytest -m unit           # Fast unit tests only
python -m pytest -m integration    # Real API tests only  
python -m pytest -m performance    # Latency tests only
```

## 📈 Coverage Reports

Generate detailed coverage reports:

```bash
# Run tests with coverage
python tests/run_tests.py --mode unit --coverage

# View HTML report
open htmlcov/index.html

# View terminal report
python -m pytest --cov=src --cov-report=term-missing
```

## 🐛 Debugging Tests

### Verbose Output
```bash
python tests/run_tests.py --mode unit --verbose
python -m pytest tests/unit/strategies/test_ma_crossover.py -v -s
```

### Single Test
```bash
# New architecture tests
python -m pytest tests/unit/strategies/test_new_architecture.py::TestStrategyParameterSchema::test_schema_exists_for_strategies -v -s
python -m pytest tests/unit/strategies/test_ma_crossover.py::TestSimpleMAKCrossoverStrategy::test_strategy_initialization_valid_config -v -s
```

### Debug with pdb
```bash
python -m pytest --pdb tests/unit/strategies/test_ma_crossover.py
```

## 🚨 Common Issues

### Missing API Keys
```
❌ Missing API credentials. Set BINANCE_API_KEY and BINANCE_SECRET environment variables
```
Solution: Set up testnet API keys in environment

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
Solution: Run tests from project root directory

### Async Test Issues
```
RuntimeWarning: coroutine was never awaited
```
Solution: Use `@pytest.mark.asyncio` decorator

## 🏛️ New Strategy Architecture Testing

The test suite has been completely refactored to support the new schema-based strategy architecture. Key improvements include:

### Architecture Components Tested
- **StrategyParameterSchema**: Parameter definitions, type validation, range checking
- **ConfigValidator**: Configuration validation, error handling
- **ConfigLoader**: Parameter loading, default value application
- **IndicatorCalculator**: Shared technical indicator calculations
- **SignalManager**: Signal creation, deduplication, filtering
- **RiskManager**: Stop loss, take profit, trailing stop calculations
- **Strategy Registry**: Strategy creation, factory patterns

### Trailing Stop Integration
Comprehensive testing of enhanced trailing stop functionality:
- Position management with trailing stops
- Dynamic stop price adjustments
- Multiple trailing stop types (percentage, absolute)
- Integration with backtesting engine

### Legacy Test Management
Legacy tests for old architecture strategies are preserved in `tests/unit/strategies/legacy/` but excluded from main test runs to maintain fast execution times.

## 🎯 Test Development Guidelines

### Writing New Architecture Tests
1. **Schema-First Testing**: Test parameter schemas before implementation
2. **Use Shared Components**: Test through IndicatorCalculator, SignalManager, etc.
3. **Mock External Dependencies**: Exchange APIs, network calls
4. **Test Strategy Logic Thoroughly**: Focus on signal generation with new architecture
5. **Include Edge Cases**: NaN values, insufficient data, invalid configurations
6. **Performance Aware**: Keep unit tests fast (< 100ms each)

### Test Naming Convention
```python
# New architecture tests
def test_strategy_initialization_valid_config()     # Schema validation
def test_shared_components_initialization()         # Component integration
def test_trailing_stop_metadata_in_signals()        # Trailing stop features
def test_insufficient_data_handling()               # Edge case
def test_parameter_access_through_schema()          # Schema usage
```

### Assertion Guidelines
```python
# Good - Specific assertions for new architecture
assert strategy.strategy_name == 'ma_crossover_simple'
assert strategy.get_parameter_value('fast_period') == 8
assert signal.metadata['trailing_stop_enabled'] == True
assert isinstance(indicators['sma_fast'], pd.Series)
assert 0.7 <= signal.confidence <= 1.0

# Avoid - Vague assertions  
assert signal is not None
assert len(signals) > 0
assert strategy.params is not None
```

## 📚 Additional Resources

- [Strategy Architecture Documentation](../CLAUDE.md) - Complete guide to new architecture
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Binance Testnet](https://testnet.binance.vision/)

## 📊 Test Results Summary

Current test suite status:
- ✅ **52 tests passing** (New architecture + Enhanced backtesting)
- 🏛️ **29 new architecture tests** - Full schema validation and shared component coverage  
- 🔄 **11 MA crossover tests** - Updated for new architecture with trailing stops
- 🎯 **12 backtesting tests** - Enhanced engine with trailing stop functionality
- 📁 **49 legacy tests** - Preserved in `legacy/` folder for migration reference

The new architecture provides comprehensive test coverage ensuring reliability and maintainability of the trading strategy system.