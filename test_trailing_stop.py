#!/usr/bin/env python3
"""
Test script for trailing stop loss functionality
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
from src.backtesting.engine import BacktestEngine

async def test_trailing_stop_loss():
    """Test trailing stop loss with simulated data"""
    
    # Create test configuration with trailing stop enabled
    config = {
        'symbols': ['TESTUSDT'],
        'position_size': 0.1,
        'fast_period': 5,
        'slow_period': 10,
        'timeframes': ['1h'],
        'stop_loss_pct': 3.0,
        'take_profit_pct': 6.0,
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 2.0,  # 2% trailing distance
        'trailing_stop_type': 'percentage'
    }
    
    print("Testing trailing stop loss functionality...")
    print(f"Configuration: {config}")
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(config)
    print(f"Strategy created successfully: {strategy.strategy_name}")
    
    # Create simulated price data that goes up then down to test trailing stop
    # Need enough data for MA calculation (slow_period = 10, so need at least 15 points)
    timestamps = []
    base_time = datetime.now() - timedelta(hours=50)
    
    # Generate data with crossover and subsequent movement
    prices = []
    for i in range(50):
        if i < 15:  # Initial period - sideways
            price = 100 + (i % 3) * 0.5  # Small oscillations around 100
        elif i < 20:  # Create uptrend for bullish crossover
            price = 100 + (i - 15) * 0.8  # Price goes up
        elif i < 35:  # Strong uptrend after signal
            price = 104 + (i - 20) * 0.7  # Continue going up to test trailing stop
        else:  # Downtrend to trigger trailing stop
            price = 114.5 - (i - 35) * 1.2  # Sharp drop to trigger trailing stop
        
        prices.append(price)
        timestamps.append(base_time + timedelta(hours=i))
    
    # Create DataFrame
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': price,
            'high': price + 0.5,
            'low': price - 0.5,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(data)
    historical_data = {'TESTUSDT': df}
    
    print(f"Created test data with {len(data)} price points")
    print(f"Price range: {min(prices):.2f} to {max(prices):.2f}")
    
    # Print first few and last few prices to see the pattern
    print("First 10 prices:", [f"{p:.2f}" for p in prices[:10]])
    print("Last 10 prices:", [f"{p:.2f}" for p in prices[-10:]])
    
    # Create backtest engine with more frequent signal checking
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
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${results['capital']['initial']:,.2f}")
    print(f"Final Capital: ${results['capital']['final']:,.2f}")
    print(f"Total Return: {results['capital']['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['trades']['total']}")
    print(f"Winning Trades: {results['trades']['winning']}")
    print(f"Losing Trades: {results['trades']['losing']}")
    
    # Check for trailing stop executions
    trailing_stops = [trade for trade in results['trades_detail'] if 'trailing_stop' in str(trade.strategy)]
    
    print(f"\n=== TRAILING STOP ANALYSIS ===")
    print(f"Trades closed by trailing stop: {len(trailing_stops)}")
    
    if trailing_stops:
        for trade in trailing_stops:
            print(f"Trailing stop trade: Entry ${trade.entry_price:.2f} -> Exit ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f}")
    
    # Print all trades for analysis
    print(f"\n=== ALL TRADES ===")
    for i, trade in enumerate(results['trades_detail']):
        print(f"Trade {i+1}: {trade.side} {trade.symbol}")
        print(f"  Entry: ${trade.entry_price:.2f} at {trade.entry_time}")
        print(f"  Exit: ${trade.exit_price:.2f} at {trade.exit_time}")
        print(f"  P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        print(f"  Strategy: {trade.strategy}")
        print()
    
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_trailing_stop_loss())