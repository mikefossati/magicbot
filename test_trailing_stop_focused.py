#!/usr/bin/env python3
"""
Focused test for trailing stop loss - ensures trailing stop triggers before take profit
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
from src.backtesting.engine import BacktestEngine

async def test_trailing_stop_only():
    """Test trailing stop loss with take profit disabled"""
    
    # Create test configuration with no take profit (set very high)
    config = {
        'symbols': ['TESTUSDT'],
        'position_size': 0.1,
        'fast_period': 5,
        'slow_period': 10,
        'timeframes': ['1h'],
        'stop_loss_pct': 10.0,  # Set very high so it doesn't trigger
        'take_profit_pct': 20.0,  # Set very high so it doesn't trigger
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 3.0,  # 3% trailing distance
        'trailing_stop_type': 'percentage'
    }
    
    print("Testing trailing stop loss (focused test)...")
    print(f"Configuration: trailing_stop_distance={config['trailing_stop_distance']}%")
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(config)
    
    # Create price data: uptrend to establish position, then sharp drop to trigger trailing stop
    timestamps = []
    base_time = datetime.now() - timedelta(hours=40)
    
    prices = []
    for i in range(40):
        if i < 15:  # Sideways movement
            price = 100 + (i % 3) * 0.3
        elif i < 20:  # Create bullish crossover
            price = 100 + (i - 15) * 1.0  # Uptrend
        elif i < 30:  # Strong uptrend to build trailing stop
            price = 105 + (i - 20) * 0.8  # Goes up to 113
        else:  # Sharp drop to trigger trailing stop at 3% below peak
            # Peak should be around 113, so 3% trailing stop = 109.61
            # Drop to 108 to trigger trailing stop
            price = 113 - (i - 30) * 1.5  # Drop to around 98
        
        prices.append(price)
        timestamps.append(base_time + timedelta(hours=i))
    
    print(f"Price pattern: starts at {prices[0]:.1f}, peaks at {max(prices):.1f}, ends at {prices[-1]:.1f}")
    
    # Create DataFrame
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': price,
            'high': price + 0.2,
            'low': price - 0.2,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(data)
    historical_data = {'TESTUSDT': df}
    
    # Create backtest engine
    engine = BacktestEngine(initial_balance=10000.0, fast_mode=False, signal_interval=1)
    
    # Run backtest
    start_date = timestamps[0]
    end_date = timestamps[-1]
    
    results = await engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n=== RESULTS ===")
    print(f"Total Trades: {results['trades']['total']}")
    
    # Look for trailing stop trades specifically
    trailing_stops = [trade for trade in results['trades_detail'] if 'trailing_stop' in str(trade.strategy)]
    
    print(f"\n=== TRAILING STOP ANALYSIS ===")
    print(f"Trades closed by trailing stop: {len(trailing_stops)}")
    
    if trailing_stops:
        for trade in trailing_stops:
            print(f"✅ TRAILING STOP TRIGGERED!")
            print(f"   Entry: ${trade.entry_price:.2f}")
            print(f"   Exit: ${trade.exit_price:.2f}")
            print(f"   P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            print(f"   Peak before drop: ~${max(prices):.2f}")
            print(f"   Expected trailing stop level: ~${max(prices) * 0.97:.2f}")
    else:
        print("❌ No trailing stop executions found")
        print("Checking all trades:")
        for trade in results['trades_detail']:
            print(f"   Trade: {trade.strategy} - Entry ${trade.entry_price:.2f} -> Exit ${trade.exit_price:.2f}")

if __name__ == "__main__":
    asyncio.run(test_trailing_stop_only())