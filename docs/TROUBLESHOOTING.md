# 🔧 MagicBot Troubleshooting Guide

This guide helps you diagnose and fix common issues when using the MagicBot trading application.

## 📋 Quick Diagnostics

### **Is Everything Working?** Run This First:

```bash
# Quick system check
python scripts/system_check.py

# Test exchange connection
python -c "
import asyncio
from src.exchanges.binance_exchange import BinanceExchange

async def test():
    exchange = BinanceExchange()
    await exchange.connect()
    print('✅ Exchange connected successfully')
    await exchange.disconnect()

asyncio.run(test())
"
```

---

## 🚨 Common Issues and Solutions

### **1. "Insufficient data for day trading analysis"**

**Symptoms:**
```
[WARNING] Insufficient data for day trading analysis
data_points=25 required=60 symbol=BTCUSDT
```

**Root Cause:** Strategy needs more historical data than available

**Solutions:**

#### **Quick Fix - Reduce Data Requirements:**
```python
# In your strategy configuration
day_trading_strategy:
  fast_ema: 8      # Keep as is
  medium_ema: 15    # Reduce from 21
  slow_ema: 30      # Reduce from 50
```

#### **Better Fix - Fetch More Data:**
```python
# In your backtest script, increase the date range
start_date = end_date - timedelta(days=14)  # Increase from 7 days
```

#### **Best Fix - Implement Warm-up Period:**
```python
# Add to your strategy
def needs_warmup_period(self) -> int:
    return max(self.slow_ema, self.volume_period, self.pivot_period) + 10
```

---

### **2. API Connection Issues**

**Symptoms:**
```
[ERROR] Failed to connect to Binance API
ConnectionError: Unable to connect to binance.com
```

**Diagnostic Steps:**

#### **Step 1: Check API Keys**
```bash
echo "API Key: $BINANCE_API_KEY"
echo "Secret Key: ${BINANCE_SECRET_KEY:0:10}..."  # Show first 10 chars only
```

#### **Step 2: Test Network Connection**
```bash
# Test if you can reach Binance
ping api.binance.com
curl -I https://api.binance.com/api/v3/ping
```

#### **Step 3: Verify API Permissions**
```python
# Test API permissions
python scripts/test_api_permissions.py
```

**Common Solutions:**

#### **Wrong API Keys:**
```bash
# Double-check your .env file or environment variables
export BINANCE_API_KEY="your_correct_api_key"
export BINANCE_SECRET_KEY="your_correct_secret_key"
```

#### **IP Restriction:**
- Log into Binance
- Go to API Management
- Check if IP restriction is enabled
- Add your current IP address

#### **API Permissions:**
Ensure your API key has these permissions enabled:
- ✅ Enable Reading
- ✅ Enable Spot & Margin Trading (for live trading)
- ❌ Enable Withdrawals (NOT recommended)

#### **Rate Limiting:**
```python
# Add rate limiting to your requests
import time

class RateLimitedExchange:
    def __init__(self):
        self.last_request = 0
        
    async def make_request(self):
        # Ensure at least 100ms between requests
        time_since_last = time.time() - self.last_request
        if time_since_last < 0.1:
            await asyncio.sleep(0.1 - time_since_last)
        
        # Make your request here
        self.last_request = time.time()
```

---

### **3. No Trades Being Executed**

**Symptoms:**
```
[INFO] Backtest completed total_trades=0
Strategy running but no signals generated
```

**Diagnostic Approach:**

#### **Step 1: Check Signal Generation**
```bash
# Test signal generation in isolation
python scripts/debug_signal_generation.py
```

#### **Step 2: Lower Signal Thresholds**
```yaml
# Make strategy more sensitive (temporarily for testing)
day_trading_strategy:
  rsi_overbought: 65     # Lower from 70
  rsi_oversold: 35       # Higher from 30
  volume_multiplier: 1.2 # Lower from 1.5
```

#### **Step 3: Check Market Conditions**
```python
# Verify you're not trading outside market hours
day_trading_strategy:
  session_start: "00:00"  # Trade 24/7 for crypto
  session_end: "23:59"
```

#### **Step 4: Review Risk Limits**
```yaml
# Check if risk limits are too restrictive
risk_management:
  max_position_size: 0.1    # Increase if too small
  max_daily_loss: 0.1       # Increase if hitting limit
```

**Common Causes & Solutions:**

#### **Too Conservative Parameters:**
```yaml
# Before (too conservative)
day_trading_strategy:
  rsi_overbought: 80
  rsi_oversold: 20
  volume_multiplier: 2.5

# After (more balanced)
day_trading_strategy:
  rsi_overbought: 70
  rsi_oversold: 30
  volume_multiplier: 1.5
```

#### **Wrong Symbol Configuration:**
```yaml
# Make sure symbols are correct
day_trading_strategy:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Not BTC/USDT
```

#### **Session Time Issues:**
```yaml
# For 24/7 crypto trading
day_trading_strategy:
  session_start: "00:00"
  session_end: "23:59"
  
# Not stock market hours like:
# session_start: "09:30"  # This limits crypto trading!
```

---

### **4. High Loss Rate / Poor Performance**

**Symptoms:**
```
Win Rate: 25%
Profit Factor: 0.6
Total Return: -15%
```

**Analysis Steps:**

#### **Step 1: Run Performance Analysis**
```bash
python scripts/analyze_poor_performance.py
```

#### **Step 2: Check Individual Components**
```python
# Test each indicator separately
python scripts/test_ema_signals.py
python scripts/test_rsi_signals.py
python scripts/test_volume_confirmation.py
```

#### **Step 3: Market Condition Analysis**
```bash
# Check if strategy is suitable for current market
python scripts/market_regime_analysis.py
```

**Common Solutions:**

#### **Wrong Market Conditions:**
```yaml
# Bull market - let trends run
day_trading_strategy:
  rsi_overbought: 75    # Higher threshold
  take_profit_pct: 3.5  # Bigger targets

# Bear market - more conservative
day_trading_strategy:
  rsi_overbought: 65    # Lower threshold
  stop_loss_pct: 2.0    # Wider stops
```

#### **Over-Optimization:**
Reset to default parameters and test:
```yaml
# Reset to proven defaults
day_trading_strategy:
  fast_ema: 8
  medium_ema: 21
  slow_ema: 50
  rsi_overbought: 70
  rsi_oversold: 30
```

#### **Insufficient Data:**
```python
# Test on longer periods
start_date = end_date - timedelta(days=90)  # Use 3 months minimum
```

---

### **5. Leverage Trading Issues**

**Symptoms:**
```
[ERROR] TypeError: unsupported operand type(s) for *: 'decimal.Decimal' and 'float'
[ERROR] Insufficient margin for leveraged position
```

**Solutions:**

#### **Type Conversion Error:**
```python
# Fixed in latest version, but if you see this:
def _calculate_leveraged_position_size(self, current_price: float, confidence: float) -> float:
    base_position_size = float(self.position_size)  # Convert Decimal to float
    # ... rest of method
```

#### **Insufficient Margin:**
```yaml
# Reduce leverage or position size
day_trading_strategy:
  leverage: 2.0              # Reduce from higher leverage
  leverage_risk_factor: 0.8  # More conservative
  position_size: 0.01        # Smaller positions
```

#### **Margin Monitoring:**
```python
# Add margin checking before trades
def check_margin_before_trade(self, position_size: float, leverage: float):
    required_margin = position_size / leverage
    available_margin = self.get_available_margin()
    
    if required_margin > available_margin * 0.8:  # Keep 20% buffer
        raise InsufficientMarginError(f"Need {required_margin}, have {available_margin}")
```

---

### **6. Database/Redis Issues**

**Symptoms:**
```
[ERROR] Connection to Redis failed
[ERROR] Database connection timeout
```

**Solutions:**

#### **Redis Connection:**
```bash
# Check if Redis is running
redis-cli ping

# If not installed:
# macOS: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo systemctl start redis
```

#### **Database Issues:**
```python
# Create database if it doesn't exist
python scripts/setup_database.py
```

#### **Connection String Issues:**
```bash
# Check your environment variables
echo "DATABASE_URL: $DATABASE_URL"
echo "REDIS_URL: $REDIS_URL"

# Set them if missing:
export DATABASE_URL="sqlite:///trading.db"
export REDIS_URL="redis://localhost:6379"
```

---

### **7. Memory/Performance Issues**

**Symptoms:**
```
Process consuming high memory
Slow backtest execution
Application freezing
```

**Solutions:**

#### **Memory Optimization:**
```python
# Limit historical data in backtests
historical_data = historical_data.tail(1000)  # Only keep recent 1000 records

# Clear unused data
import gc
gc.collect()
```

#### **Parallel Processing:**
```python
# Use asyncio properly
async def run_multiple_backtests():
    tasks = [
        run_backtest(strategy1, data1),
        run_backtest(strategy2, data2),
        run_backtest(strategy3, data3)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

#### **Data Chunking:**
```python
# Process data in chunks for large backtests
def chunk_backtest(data, chunk_size=100):
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        yield run_backtest_chunk(chunk)
```

---

## 🔍 Advanced Debugging

### **Enable Debug Logging**

```python
import structlog
import logging

# Enable debug level logging
logging.basicConfig(level=logging.DEBUG)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
```

### **Strategy-Specific Debugging**

#### **Debug Signal Generation:**
```python
# Add to your strategy
def debug_signal_generation(self, symbol: str, data: pd.DataFrame):
    indicators = self._calculate_indicators(data)
    
    print(f"=== DEBUG {symbol} ===")
    print(f"Current Price: {data['close'].iloc[-1]}")
    print(f"EMA Fast: {indicators['ema_fast'].iloc[-1]:.2f}")
    print(f"EMA Medium: {indicators['ema_medium'].iloc[-1]:.2f}")
    print(f"EMA Slow: {indicators['ema_slow'].iloc[-1]:.2f}")
    print(f"RSI: {indicators['rsi'].iloc[-1]:.2f}")
    print(f"Volume Ratio: {indicators['volume_ratio'].iloc[-1]:.2f}")
    print("=" * 20)
```

#### **Debug Risk Calculations:**
```python
def debug_risk_calculation(self, signal: Signal):
    print(f"=== RISK DEBUG ===")
    print(f"Entry Price: {signal.price}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Take Profit: {signal.take_profit}")
    print(f"Risk Amount: {(signal.price - signal.stop_loss) * signal.quantity}")
    print(f"Reward Amount: {(signal.take_profit - signal.price) * signal.quantity}")
    print(f"Risk/Reward: {(signal.take_profit - signal.price) / (signal.price - signal.stop_loss):.2f}")
```

### **Performance Profiling**

```python
import cProfile
import pstats

def profile_backtest():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your backtest code here
    run_backtest()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)
```

---

## 📊 Diagnostic Scripts

Create these helper scripts for troubleshooting:

### **System Check Script**
```bash
# scripts/system_check.py
#!/usr/bin/env python3
import sys
import subprocess
import importlib

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version}")
    return True

def check_dependencies():
    required = ['pandas', 'numpy', 'asyncio', 'structlog']
    missing = []
    
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return len(missing) == 0

def check_environment():
    import os
    vars_to_check = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
    
    for var in vars_to_check:
        if os.getenv(var):
            print(f"✅ {var} set")
        else:
            print(f"❌ {var} missing")
            return False
    return True

if __name__ == "__main__":
    print("🔍 MagicBot System Check")
    print("=" * 30)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed. Fix issues above.")
```

### **Signal Debug Script**
```bash
# scripts/debug_signals.py
#!/usr/bin/env python3
import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager

async def debug_signals():
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get recent data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',
            start_date=start_date,
            end_date=end_date
        )
        
        # Create strategy
        strategy_config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 8,
            'medium_ema': 21,
            'slow_ema': 50,
            # ... other config
        }
        
        strategy = create_strategy('day_trading_strategy', strategy_config)
        
        # Debug signal generation
        data = historical_data['BTCUSDT'].tail(100)
        strategy.debug_signal_generation('BTCUSDT', data)
        
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_signals())
```

---

## 🆘 Getting Help

### **Before Asking for Help:**

1. **Check the logs** - Look for specific error messages
2. **Try the basic fixes** - API keys, internet connection, etc.
3. **Test with defaults** - Reset to default configuration
4. **Run diagnostics** - Use the system check script
5. **Search this guide** - Use Ctrl+F to find your error message

### **When Reporting Issues:**

Include this information:

```bash
# System info
python --version
pip list | grep -E "(pandas|numpy|asyncio)"

# Error details
tail -50 logs/trading.log

# Configuration (remove API keys!)
cat config/development.yaml | grep -v "key"

# Recent commands
history | tail -10
```

### **Debug Information Template:**

```
**Environment:**
- OS: [Windows/macOS/Linux]
- Python Version: [3.x.x]
- MagicBot Version: [x.x.x]

**Issue Description:**
[Clear description of what's not working]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Error Messages:**
[Copy exact error messages]

**Configuration:**
[Relevant config sections - remove API keys!]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [etc.]
```

---

## 🎯 Prevention Tips

### **Best Practices to Avoid Issues:**

1. **Always test on testnet first**
2. **Start with default parameters**
3. **Keep backups of working configurations**
4. **Update dependencies regularly**
5. **Monitor system resources**
6. **Use version control for configs**

### **Regular Maintenance:**

```bash
# Weekly maintenance script
#!/bin/bash

# Update dependencies
pip install --upgrade -r requirements.txt

# Clean up logs
find logs/ -name "*.log" -mtime +7 -delete

# Database maintenance
python scripts/optimize_database.py

# Backup configurations
cp config/development.yaml backups/config_$(date +%Y%m%d).yaml
```

---

**Remember**: Most issues have simple solutions. Work through this guide systematically, and don't hesitate to start with the basics like API keys and internet connectivity! 🔧✅