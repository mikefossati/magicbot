# 🎛️ Strategy Parameters Quick Reference

This guide provides detailed explanations of all strategy parameters and how to optimize them for different market conditions.

## 📊 Day Trading Strategy Parameters

### **EMA (Exponential Moving Average) Settings**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `fast_ema` | 8 | 5-15 | Fast trend detection | Lower = more signals, more noise |
| `medium_ema` | 21 | 15-30 | Medium-term trend | Standard Fibonacci number |
| `slow_ema` | 50 | 30-100 | Long-term trend filter | Higher = fewer, stronger signals |

**Market Conditions:**
- **Trending Markets**: Use 8/21/50 (default)
- **Choppy Markets**: Use 10/25/60 (slower)
- **High Volatility**: Use 5/15/40 (faster)

### **RSI (Relative Strength Index) Settings**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `rsi_period` | 14 | 7-21 | RSI calculation period | 14 is standard, don't change unless needed |
| `rsi_overbought` | 70 | 65-85 | Sell signal threshold | Higher = fewer sell signals |
| `rsi_oversold` | 30 | 15-35 | Buy signal threshold | Lower = fewer buy signals |
| `rsi_neutral_high` | 60 | 55-65 | Upper neutral zone | Creates buffer zone |
| `rsi_neutral_low` | 40 | 35-45 | Lower neutral zone | Prevents whipsaws |

**Market Conditions:**
- **Bull Market**: 75/25 thresholds
- **Bear Market**: 65/35 thresholds  
- **Sideways**: 70/30 thresholds (default)

### **MACD Settings**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `macd_fast` | 12 | 8-15 | Fast EMA period | Standard setting, rarely changed |
| `macd_slow` | 26 | 20-35 | Slow EMA period | Standard setting, rarely changed |
| `macd_signal` | 9 | 7-12 | Signal line smoothing | Lower = more sensitive |

**Note**: MACD parameters are standardized. Only adjust for specific timeframes.

### **Volume Analysis**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `volume_period` | 20 | 10-30 | Volume MA period | Match with EMA periods |
| `volume_multiplier` | 1.5 | 1.2-2.5 | Volume confirmation threshold | Higher = stronger signals only |

**Market Conditions:**
- **Low Volume Markets**: 1.2-1.3 multiplier
- **High Volume Markets**: 1.8-2.5 multiplier

### **Support/Resistance**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `pivot_period` | 10 | 5-20 | S/R calculation period | Lower = more reactive levels |
| `support_resistance_threshold` | 0.2 | 0.1-0.5 | Price proximity to S/R (%) | Lower = must be very close |

### **Risk Management**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `stop_loss_pct` | 1.5 | 0.5-3.0 | Stop loss percentage | Higher for volatile pairs |
| `take_profit_pct` | 2.5 | 1.0-5.0 | Take profit percentage | Aim for 1.5:1 risk/reward |
| `trailing_stop_pct` | 1.0 | 0.5-2.0 | Trailing stop percentage | Let winners run |
| `max_daily_trades` | 3 | 1-10 | Daily trade limit | Prevent overtrading |

### **Position Sizing**

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `position_size` | 0.02 | 0.01-0.05 | Position size (% of capital) | Start small, scale up |

### **Leverage Settings** (Advanced)

| Parameter | Default | Range | Description | Optimization Tips |
|-----------|---------|-------|-------------|-------------------|
| `use_leverage` | false | true/false | Enable leverage trading | Start without leverage |
| `leverage` | 3.0 | 1.0-10.0 | Leverage multiplier | 2x-3x recommended |
| `max_leverage` | 10.0 | 1.0-20.0 | Maximum allowed leverage | Safety limit |
| `leverage_risk_factor` | 0.6 | 0.3-0.8 | Position reduction factor | Higher leverage = smaller positions |

---

## 📈 Other Strategy Parameters

### **RSI Strategy**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `rsi_period` | 14 | 10-21 | RSI calculation period |
| `oversold` | 30 | 20-35 | Buy signal threshold |
| `overbought` | 70 | 65-80 | Sell signal threshold |
| `position_size` | 0.01 | 0.005-0.03 | Position size |

### **Bollinger Bands Strategy**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `period` | 20 | 15-25 | Moving average period |
| `std_dev` | 2.0 | 1.5-2.5 | Standard deviation multiplier |
| `mean_reversion_threshold` | 0.02 | 0.01-0.05 | Mean reversion trigger |

### **Breakout Strategy**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lookback_period` | 20 | 10-30 | Breakout detection period |
| `breakout_threshold` | 1.02 | 1.01-1.05 | Breakout confirmation level |
| `volume_confirmation` | true | true/false | Require volume confirmation |
| `min_volatility` | 0.005 | 0.001-0.01 | Minimum volatility filter |

### **MACD Strategy**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `fast_period` | 12 | 8-15 | MACD fast EMA |
| `slow_period` | 26 | 20-35 | MACD slow EMA |
| `signal_period` | 9 | 7-12 | Signal line period |
| `histogram_threshold` | 0.0 | -0.01-0.01 | Histogram cross threshold |

### **Momentum Strategy**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `momentum_period` | 14 | 10-20 | Momentum calculation period |
| `sma_period` | 20 | 15-30 | Simple moving average period |
| `momentum_threshold` | 5.0 | 2.0-10.0 | Momentum signal threshold |
| `volume_confirmation` | true | true/false | Volume confirmation |

---

## 🎯 Parameter Optimization Guide

### **Step 1: Baseline Testing**
Start with default parameters and run backtests:
```bash
python scripts/test_day_trading_strategy.py
```

### **Step 2: Single Parameter Optimization**
Change one parameter at a time:

```yaml
# Test different EMA periods
day_trading_strategy:
  fast_ema: 5    # Test: 5, 8, 10, 12
  medium_ema: 21 # Keep constant
  slow_ema: 50   # Keep constant
```

### **Step 3: Risk Parameter Adjustment**
Optimize risk settings based on volatility:

```yaml
# For high volatility pairs (e.g., small altcoins)
day_trading_strategy:
  stop_loss_pct: 2.5     # Wider stops
  take_profit_pct: 4.0   # Higher targets
  position_size: 0.015   # Smaller positions

# For low volatility pairs (e.g., BTC/ETH)
day_trading_strategy:
  stop_loss_pct: 1.0     # Tighter stops
  take_profit_pct: 2.0   # Lower targets
  position_size: 0.025   # Larger positions
```

### **Step 4: Market Condition Adaptation**

#### **Bull Market Settings**
```yaml
day_trading_strategy:
  fast_ema: 8
  medium_ema: 21
  slow_ema: 50
  rsi_overbought: 75     # Let trends run longer
  rsi_oversold: 25
  take_profit_pct: 3.5   # Higher targets in trends
```

#### **Bear Market Settings**
```yaml
day_trading_strategy:
  fast_ema: 10           # Slower signals
  medium_ema: 25
  slow_ema: 60
  rsi_overbought: 65     # Earlier exits
  rsi_oversold: 35
  stop_loss_pct: 2.0     # Wider stops for volatility
```

#### **Sideways Market Settings**
```yaml
day_trading_strategy:
  fast_ema: 6            # Faster for range trading
  medium_ema: 18
  slow_ema: 45
  rsi_overbought: 70     # Standard levels
  rsi_oversold: 30
  volume_multiplier: 1.8 # Higher volume confirmation
```

---

## 📊 Performance Optimization Matrix

### **Signal Frequency vs Quality**

| Setting | Signals/Day | Win Rate | Profit Factor | Best For |
|---------|-------------|----------|---------------|----------|
| Conservative | 1-2 | 70%+ | 2.0+ | Beginners |
| Balanced | 2-4 | 60%+ | 1.5+ | Most traders |
| Aggressive | 4-8 | 50%+ | 1.2+ | Experienced |

### **Conservative Settings**
```yaml
day_trading_strategy:
  fast_ema: 10
  rsi_overbought: 75
  rsi_oversold: 25
  volume_multiplier: 2.0
  max_daily_trades: 2
```

### **Balanced Settings** (Default)
```yaml
day_trading_strategy:
  fast_ema: 8
  rsi_overbought: 70
  rsi_oversold: 30
  volume_multiplier: 1.5
  max_daily_trades: 3
```

### **Aggressive Settings**
```yaml
day_trading_strategy:
  fast_ema: 6
  rsi_overbought: 65
  rsi_oversold: 35
  volume_multiplier: 1.2
  max_daily_trades: 5
```

---

## 🔍 Parameter Testing Tools

### **Backtesting Scripts**
```bash
# Test parameter variations
python scripts/optimize_parameters.py --strategy day_trading --param fast_ema --range 5,15

# Compare different configurations
python scripts/compare_configurations.py
```

### **Parameter Sensitivity Analysis**
```bash
# Analyze how sensitive the strategy is to parameter changes
python scripts/parameter_sensitivity.py
```

---

## ⚠️ Optimization Warnings

### **Common Mistakes**
1. **Over-Optimization**: Parameters that work perfectly on historical data may fail in live trading
2. **Curve Fitting**: Optimizing too many parameters simultaneously
3. **Not Testing Out-of-Sample**: Always test on unseen data
4. **Ignoring Transaction Costs**: Include realistic fees and slippage

### **Best Practices**
1. **Test on Multiple Time Periods**: Bull, bear, and sideways markets
2. **Use Walk-Forward Analysis**: Test on rolling periods
3. **Keep It Simple**: Fewer parameters = more robust strategy
4. **Regular Re-optimization**: Market conditions change over time

---

## 📈 Quick Parameter Reference Card

**Print this for quick reference:**

```
DAY TRADING STRATEGY - QUICK SETTINGS

Trending Market:     EMA 8/21/50, RSI 75/25, Vol 1.5x
Sideways Market:     EMA 6/18/45, RSI 70/30, Vol 1.8x
Volatile Market:     EMA 10/25/60, RSI 65/35, Vol 2.0x

Risk Levels:
Conservative:        1.5% stop, 2.5% target, 2% position
Balanced:           1.5% stop, 2.5% target, 2% position  
Aggressive:         2.0% stop, 3.5% target, 2.5% position

Leverage (Advanced):
Beginner:           No leverage (1x)
Intermediate:       2x-3x with 0.7 risk factor
Advanced:           3x-5x with 0.5 risk factor
```