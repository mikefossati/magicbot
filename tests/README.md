# Trading Strategy Test Suite

Comprehensive unit and integration tests for all trading strategies in the Magicbot system.

## 🏗️ Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── run_tests.py                    # Test runner script
├── unit/                           # Fast tests with mocked data
│   ├── fixtures/
│   │   └── historical_snapshots.py # Real historical market data
│   └── strategies/
│       ├── test_ma_crossover.py    # MA crossover strategy tests
│       ├── test_day_trading_strategy.py # Day trading scenarios
│       ├── test_macd_strategy.py   # MACD strategy tests
│       └── test_stochastic_strategy.py # Stochastic tests
├── integration/                    # Tests with real Binance testnet
│   ├── test_strategy_integration.py # End-to-end strategy tests
│   └── test_exchange_integration.py # Exchange API tests
└── performance/                    # Performance and latency tests
    └── test_latency.py            # Low latency requirements
```

## 🚀 Quick Start

### Run Unit Tests (Fast)
```bash
# Run all unit tests
python tests/run_tests.py --mode unit

# Run with coverage
python tests/run_tests.py --mode unit --coverage

# Run specific strategy
python tests/run_tests.py --strategy ma_crossover
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
- **Strategy Logic Testing**: Signal generation, crossover detection
- **Parameter Validation**: Invalid configs, edge cases  
- **Risk Management**: Stop loss, take profit calculations
- **Mock Data Scenarios**: Bull/bear markets, volatile conditions
- **Concurrent Processing**: Multiple symbols, async performance

### Integration Tests (Manual - Real API)
- **Real Market Data**: Live Binance testnet integration
- **End-to-End Workflows**: Data fetch → analysis → signal generation
- **Low Latency Validation**: < 100ms average, < 200ms max
- **Connection Stability**: Extended use, error recovery
- **Memory Usage**: No memory leaks over time

### Day Trading Scenarios
- **Morning Breakout**: Volume confirmation, trend following
- **Trend Reversal**: Support/resistance levels
- **Choppy Markets**: Signal avoidance in unclear conditions
- **False Breakouts**: Conflicting indicator handling
- **Session Filtering**: Trading hours enforcement

## 🧪 Test Data Strategy

### Historical Snapshots
Uses real historical market data snapshots for deterministic testing:

```python
# Available scenarios
'bullish_crossover'    # MA crossover upward trend
'bearish_crossover'    # MA crossover downward trend  
'volatile_market'      # High volatility, choppy conditions
'morning_breakout'     # Day trading breakout pattern
'eth_sample'           # ETH data for multi-symbol tests
```

### Mock Strategy
- **Deep Mocking**: Exchange APIs, network calls mocked
- **Strategy Logic**: Fully tested with real calculation logic
- **Indicator Math**: Pandas operations tested with real data
- **Edge Cases**: NaN handling, insufficient data, errors

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
python tests/run_tests.py --strategy ma_crossover
python tests/run_tests.py --strategy day_trading_strategy
```

### By Scenario
```bash
python tests/run_tests.py --mode day-trading
python -m pytest -k "morning_breakout"
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
python -m pytest tests/unit/strategies/test_ma_crossover.py::TestMAKCrossoverStrategy::test_bullish_crossover_signal -v -s
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

## 🎯 Test Development Guidelines

### Writing New Tests
1. **Use Historical Snapshots**: Prefer real data over synthetic
2. **Mock External Dependencies**: Exchange APIs, network calls
3. **Test Strategy Logic Thoroughly**: Focus on signal generation
4. **Include Edge Cases**: NaN values, insufficient data
5. **Performance Aware**: Keep unit tests fast (< 100ms each)

### Test Naming Convention
```python
def test_bullish_crossover_signal()        # What it tests
def test_insufficient_data_handling()      # Edge case
def test_concurrent_symbol_processing()    # Performance aspect
```

### Assertion Guidelines
```python
# Good - Specific assertions
assert signal.action == 'BUY'
assert 0.7 <= signal.confidence <= 1.0
assert signal.stop_loss < signal.price

# Avoid - Vague assertions  
assert signal is not None
assert len(signals) > 0
```

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Binance Testnet](https://testnet.binance.vision/)
- [Strategy Implementation Guide](../docs/STRATEGY_GUIDE.md)