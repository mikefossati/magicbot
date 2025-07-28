#!/usr/bin/env python3
"""
Test the improved MA crossover strategy on a bullish trend
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
from src.backtesting.engine import BacktestEngine

async def test_improved_strategy():
    """Test improved strategy on simulated bullish data"""
    
    # Use the new optimized configuration
    config = {
        'symbols': ['ETHUSDT'],
        'position_size': 0.01,
        'fast_period': 5,        # Updated parameters
        'slow_period': 15,       # Updated parameters
        'timeframes': ['1m'],
        'stop_loss_pct': 1.0,    # Tighter stops
        'take_profit_pct': 2.0,  # Smaller targets
        'atr_period': 10,
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 0.8,
        'trailing_stop_type': 'percentage'
    }
    
    print("Testing improved MA crossover strategy...")
    print(f"Parameters: fast_period={config['fast_period']}, slow_period={config['slow_period']}")
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(config)
    
    # Create realistic bullish price data (similar to ETH's recent movement)
    # Start around 3755, end around 3846 with realistic intraday volatility
    timestamps = []
    base_time = datetime.now() - timedelta(hours=24)
    
    prices = []
    base_price = 3755.0
    trend_slope = (3846 - 3755) / 1440  # 24 hours * 60 minutes = 1440 minutes
    
    for i in range(1440):  # 24 hours of 1-minute data
        # Add trend + realistic volatility
        trend_price = base_price + i * trend_slope
        
        # Add realistic intraday noise (0.1% to 0.3% volatility)
        noise = np.random.normal(0, trend_price * 0.002)  # 0.2% standard deviation
        
        # Add some occasional larger moves (simulate real market)
        if i % 120 == 0:  # Every 2 hours, add a potential momentum move
            momentum = np.random.normal(0, trend_price * 0.005)  # 0.5% momentum move
            noise += momentum
        
        final_price = trend_price + noise
        prices.append(final_price)
        timestamps.append(base_time + timedelta(minutes=i))
    
    print(f"Created {len(prices)} price points")
    print(f"Price movement: ${prices[0]:.2f} -> ${prices[-1]:.2f} ({((prices[-1]/prices[0])-1)*100:.2f}%)")
    
    # Create DataFrame  
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Add realistic OHLC based on price
        high = price + abs(np.random.normal(0, price * 0.0005))
        low = price - abs(np.random.normal(0, price * 0.0005))
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': open_price,
            'high': max(open_price, high, price),
            'low': min(open_price, low, price),
            'close': price,
            'volume': np.random.uniform(800, 1200)
        })
    
    df = pd.DataFrame(data)
    historical_data = {'ETHUSDT': df}
    
    # Create backtest engine
    engine = BacktestEngine(initial_balance=10000.0, fast_mode=False, signal_interval=5)
    
    # Run backtest
    start_date = timestamps[0]
    end_date = timestamps[-1]
    
    results = await engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\n=== IMPROVED STRATEGY RESULTS ===")
    print(f"Market Return: {((prices[-1]/prices[0])-1)*100:.2f}%")
    print(f"Strategy Return: {results['capital']['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['trades']['total']}")
    print(f"Win Rate: {results['trades']['win_rate_pct']:.1f}%")
    print(f"Winning Trades: {results['trades']['winning']}")
    print(f"Losing Trades: {results['trades']['losing']}")
    
    if results['trades']['total'] > 0:
        print(f"Avg Win: ${results['trades']['avg_win']:.2f}")
        print(f"Avg Loss: ${results['trades']['avg_loss']:.2f}")
        print(f"Profit Factor: {results['trades']['profit_factor']:.2f}")
    
    # Analyze signal quality
    total_signals = len(results['signals_log'])
    buy_signals = len([s for s in results['signals_log'] if s['action'] == 'BUY'])
    sell_signals = len([s for s in results['signals_log'] if s['action'] == 'SELL'])
    
    print(f"\n=== SIGNAL ANALYSIS ===")
    print(f"Total Signals Generated: {total_signals}")
    print(f"Buy Signals: {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    print(f"Signal to Trade Ratio: {results['trades']['total']}/{total_signals} = {results['trades']['total']/max(1,total_signals):.2f}")
    
    # Check for any trailing stop executions
    trailing_stops = [trade for trade in results['trades_detail'] if 'trailing_stop' in str(trade.strategy)]
    if trailing_stops:
        print(f"\nTrailing stop executions: {len(trailing_stops)}")
        for trade in trailing_stops:
            print(f"  ${trade.entry_price:.2f} -> ${trade.exit_price:.2f} ({trade.pnl_pct:.2f}%)")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    asyncio.run(test_improved_strategy())