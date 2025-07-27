# Trading Strategy Architecture Documentation

This document describes the architectural patterns and conventions for implementing trading strategies in the magicbot system. All new strategies must follow this architecture to ensure consistency, maintainability, and proper integration with the trading engine.

## Architecture Overview

The trading strategy system uses a centralized, configuration-driven architecture where:

1. **YAML configuration is the single source of truth** for all parameters
2. **Shared components** eliminate code duplication across strategies  
3. **Schema-based validation** ensures configuration correctness
4. **Standardized interfaces** enable seamless integration with backtesting and optimization

## Core Components

### 1. Parameter Schema (`src/strategies/config/schema.py`)

All strategy parameters must be defined in the centralized schema before implementation:

```python
# Add your strategy to STRATEGY_SCHEMAS
'my_new_strategy': {
    'required_params': ['symbols', 'position_size', 'my_param'],
    'optional_params': ['timeframes', 'stop_loss_pct', 'take_profit_pct'],
    'custom_params': {
        'my_param': ParameterDefinition(
            name='my_param',
            param_type=ParameterType.INTEGER,
            default=20,
            min_value=5,
            max_value=100,
            description='My custom parameter description'
        )
    }
}
```

### 2. Shared Components

#### IndicatorCalculator (`src/strategies/components/indicators.py`)
Provides centralized technical indicator calculations:
- `calculate_rsi(prices, period)`
- `calculate_sma(prices, period)` 
- `calculate_ema(prices, period)`
- `calculate_macd(prices, fast, slow, signal)`
- `calculate_bollinger_bands(prices, period, std_dev)`
- `calculate_all_indicators(data, params)` - Auto-calculates based on strategy parameters

#### SignalManager (`src/strategies/components/signals.py`)
Handles signal generation and filtering:
- Signal deduplication to prevent duplicate trades
- Confidence-based position sizing
- Configurable signal filtering rules

#### RiskManager (`src/strategies/components/risk.py`)
Manages risk and position sizing:
- Stop loss and take profit calculation
- Position sizing with confidence scaling
- Leverage support with risk adjustments
- Portfolio risk validation

## Creating a New Strategy

### Step 1: Define Schema

Add your strategy parameters to `src/strategies/config/schema.py`:

```python
'trend_following_strategy': {
    'required_params': ['symbols', 'position_size'],
    'optional_params': ['timeframes', 'stop_loss_pct', 'take_profit_pct', 'volume_confirmation'],
    'custom_params': {
        'trend_period': ParameterDefinition(
            name='trend_period',
            param_type=ParameterType.INTEGER,
            default=20,
            min_value=5,
            max_value=200,
            description='Period for trend calculation'
        ),
        'momentum_threshold': ParameterDefinition(
            name='momentum_threshold',
            param_type=ParameterType.FLOAT,
            default=0.02,
            min_value=0.001,
            max_value=0.1,
            description='Minimum momentum threshold for signals'
        )
    }
}
```

### Step 2: Implement Strategy Class

Create your strategy file `src/strategies/trend_following_strategy.py`:

```python
import pandas as pd
from typing import Dict, List, Any, Optional

from .base import BaseStrategy
from .signal import Signal

class TrendFollowingStrategy(BaseStrategy):
    """Trend Following Strategy using new architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        # Must call super with strategy name matching schema
        super().__init__('trend_following_strategy', config)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on trend analysis"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            try:
                signal = await self._analyze_symbol(symbol, market_data[symbol])
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
        
        return signals
    
    async def _analyze_symbol(self, symbol: str, data: List[Dict]) -> Optional[Signal]:
        """Analyze a single symbol using new architecture patterns"""
        # Get parameters from validated configuration
        trend_period = self.get_parameter_value('trend_period')
        momentum_threshold = self.get_parameter_value('momentum_threshold')
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Check minimum data requirements
        if len(df) < trend_period + 1:
            return None
        
        # Calculate indicators using shared component
        indicators = self.calculate_indicators(df)
        
        # Your strategy logic here
        current_price = float(df['close'].iloc[-1])
        
        # Example: Simple trend following
        trend_ma = indicators.get('sma_fast', df['close'].rolling(trend_period).mean())
        current_ma = float(trend_ma.iloc[-1])
        prev_ma = float(trend_ma.iloc[-2])
        
        momentum = (current_ma - prev_ma) / prev_ma
        
        signal_action = 'HOLD'
        confidence = 0.0
        
        if momentum > momentum_threshold:
            signal_action = 'BUY'
            confidence = min(0.9, 0.5 + abs(momentum) * 10)
        elif momentum < -momentum_threshold:
            signal_action = 'SELL'
            confidence = min(0.9, 0.5 + abs(momentum) * 10)
        
        # Create signal using shared signal manager
        if signal_action != 'HOLD':
            metadata = {
                'trend_period': trend_period,
                'momentum': momentum,
                'momentum_threshold': momentum_threshold,
                'current_ma': current_ma,
                'atr': float(indicators.get('atr', 0))
            }
            
            return self.create_signal(symbol, signal_action, current_price, confidence, metadata)
        
        return None
    
    def _prepare_data(self, data: List[Dict]) -> pd.DataFrame:
        """Convert list of dicts to DataFrame with proper types"""
        df = pd.DataFrame(data)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df
```

### Step 3: Register Strategy

Add your strategy to `src/strategies/registry.py`:

```python
from .trend_following_strategy import TrendFollowingStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    # ... existing strategies
    'trend_following_strategy': TrendFollowingStrategy,
}
```

And to `src/strategies/__init__.py`:

```python
from .trend_following_strategy import TrendFollowingStrategy

__all__ = [
    # ... existing exports
    'TrendFollowingStrategy',
]
```

### Step 4: Add Configuration

Add strategy configuration to `config/development.yaml`:

```yaml
strategies:
  enabled:
    - trend_following_strategy  # Add to enabled list
    
  trend_following_strategy:
    symbols: ["BTCUSDT", "ETHUSDT"]
    position_size: 0.02
    trend_period: 20
    momentum_threshold: 0.015
    timeframes: ["1h"]
    stop_loss_pct: 2.0
    take_profit_pct: 4.0
    volume_confirmation: true
```

## Architecture Rules and Conventions

### Required Patterns

1. **Constructor Pattern**:
   ```python
   def __init__(self, config: Dict[str, Any]):
       super().__init__('strategy_name', config)  # Must match schema key
   ```

2. **Parameter Access**:
   ```python
   # CORRECT: Use schema-validated parameters
   period = self.get_parameter_value('my_period')
   
   # WRONG: Direct config access with defaults
   period = config.get('my_period', 20)  # NO HARDCODED DEFAULTS
   ```

3. **Indicator Calculation**:
   ```python
   # CORRECT: Use shared component
   indicators = self.calculate_indicators(df)
   rsi = indicators['rsi']
   
   # WRONG: Duplicate calculation code
   def _calculate_rsi(self, prices):  # Don't duplicate
   ```

4. **Signal Creation**:
   ```python
   # CORRECT: Use signal manager
   return self.create_signal(symbol, action, price, confidence, metadata)
   
   # WRONG: Direct Signal instantiation
   return Signal(symbol=symbol, action=action, ...)  # Bypasses risk management
   ```

### Schema Validation Rules

1. **All parameters must be defined in schema** - No parameters can be used without schema definition
2. **Required parameters must be provided** - Strategies will fail to initialize without them
3. **Type validation is enforced** - INTEGER parameters cannot accept strings
4. **Range validation applies** - Values outside min/max ranges are rejected
5. **Strategy-specific validation** - Custom validation rules in `ConfigValidator`

### Data Requirements

The system automatically calculates data requirements based on your parameters:

```python
# Automatically determined from schema:
data_requirements = {
    'timeframes': ['1h'],  # From timeframes parameter
    'lookback_periods': 50,  # Max(period parameters) + buffer
    'indicators': ['sma', 'rsi']  # Based on strategy type
}
```

## Testing New Strategies

### Unit Test Template

Create `tests/test_my_strategy.py`:

```python
import pytest
from src.strategies.registry import create_strategy

class TestMyStrategy:
    def test_strategy_creation(self):
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'trend_period': 20,
            'momentum_threshold': 0.02
        }
        
        strategy = create_strategy('trend_following_strategy', config)
        assert strategy.strategy_name == 'trend_following_strategy'
        assert strategy.get_parameter_value('trend_period') == 20
    
    def test_invalid_config(self):
        config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01,
            'trend_period': -5  # Invalid: below min_value
        }
        
        with pytest.raises(ValidationError):
            create_strategy('trend_following_strategy', config)
```

### Integration Testing

```python
# Test with backtesting engine
from src.backtesting.engine import BacktestEngine

async def test_strategy_backtest():
    strategy = create_strategy('trend_following_strategy', config)
    engine = BacktestEngine(initial_balance=10000.0)
    
    results = await engine.run_backtest(
        strategy=strategy,
        historical_data=test_data,
        start_date=start_date,
        end_date=end_date
    )
    
    assert results['total_trades'] >= 0
    assert 'sharpe_ratio' in results['metrics']
```

## Optimization Integration

For parameter optimization, use the tuple factory pattern:

```python
# In optimization code
strategy_factory = ('trend_following_strategy', {
    'symbols': ['BTCUSDT'],
    'position_size': 0.01,
    'timeframes': ['1h']
})

# Optimizer will merge optimization parameters with base config
best_result = await optimizer.optimize(
    strategy_factory=strategy_factory,
    parameter_space=param_space,
    historical_data=data,
    start_date=start,
    end_date=end
)
```

## Common Patterns and Best Practices

### Error Handling
```python
try:
    signal = await self._analyze_symbol(symbol, data)
    if signal:
        signals.append(signal)
except Exception as e:
    self.logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
    # Continue processing other symbols
```

### Logging
```python
import structlog
logger = structlog.get_logger()

# Use structured logging
logger.info("Signal generated",
           strategy=self.strategy_name,
           symbol=symbol,
           action=action,
           confidence=confidence)
```

### Data Validation
```python
# Always validate data before analysis
if len(data) < self.get_parameter_value('min_required_periods'):
    logger.warning("Insufficient data", symbol=symbol, data_points=len(data))
    return None

# Check for NaN values
if pd.isna(current_value):
    return None
```

### Metadata Best Practices
```python
metadata = {
    'strategy_version': '1.0',
    'signal_timestamp': current_timestamp,
    'primary_indicator': indicator_value,
    'secondary_confirmations': [confirm1, confirm2],
    'risk_metrics': {
        'volatility': volatility,
        'atr': atr_value
    }
}
```

## Performance Considerations

1. **Use shared components** - Don't implement your own indicator calculations
2. **Minimize data copying** - Work with DataFrames efficiently  
3. **Async where possible** - Use `await asyncio.to_thread()` for CPU-intensive operations
4. **Cache expensive calculations** - Store intermediate results when appropriate
5. **Validate early** - Check data requirements before heavy computation

## Migration from Legacy Strategies

If updating an existing strategy to the new architecture:

1. **Extract hardcoded defaults** → Move to schema definitions
2. **Remove duplicate calculations** → Use shared IndicatorCalculator
3. **Update constructor signature** → Use new BaseStrategy pattern
4. **Replace direct Signal creation** → Use SignalManager
5. **Add proper validation** → Define schema validation rules
6. **Update tests** → Use new configuration patterns

## Troubleshooting

### Common Issues

1. **ValidationError on strategy creation**
   - Check schema definition matches strategy name
   - Verify all required parameters are provided
   - Check parameter types and ranges

2. **ImportError with circular imports**
   - Import Signal from `src.strategies.signal`
   - Import BaseStrategy from `src.strategies.base`

3. **Missing indicators in calculate_indicators()**
   - Add required indicators to schema's `_get_required_indicators()`
   - Ensure parameter names match indicator calculation expectations

4. **Signals not generated**
   - Check `should_generate_signal()` deduplication logic
   - Verify confidence meets minimum thresholds
   - Ensure metadata includes required risk management data

### Debugging Tools

```python
# Check strategy configuration
strategy = create_strategy('my_strategy', config)
print(strategy.get_strategy_info())

# Validate configuration without creating strategy
from src.strategies.registry import validate_strategy_config
is_valid = validate_strategy_config('my_strategy', config)

# Check parameter schema
from src.strategies.registry import get_strategy_info
info = get_strategy_info('my_strategy')
print(info['parameters'])
```

This architecture ensures consistency, maintainability, and proper integration across the entire trading system. All new strategies must follow these patterns to maintain system integrity and enable seamless operation with backtesting, optimization, and live trading components.