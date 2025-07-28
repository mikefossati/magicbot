#!/usr/bin/env python3
"""
Test the balanced MA crossover strategy with adjusted thresholds
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategies.ma_crossover_simple import SimpleMovingAverageCrossover
from src.backtesting.engine import BacktestEngine

async def test_balanced_strategy():
    """Test balanced strategy with realistic thresholds"""
    
    # Use the balanced configuration
    config = {
        'symbols': ['ETHUSDT'],
        'position_size': 0.02,
        'fast_period': 8,
        'slow_period': 21,
        'timeframes': ['5m'],
        'stop_loss_pct': 2.5,
        'take_profit_pct': 5.0,
        'atr_period': 14,
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 1.8,
        'trailing_stop_type': 'percentage'
    }
    
    print("Testing balanced MA crossover strategy...")
    print(f"Parameters: {config['fast_period']}/{config['slow_period']} MA, 5m timeframe")
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(config)
    
    # Create realistic 5-minute price data similar to recent ETH movement
    # Simulate the actual ETH movement: 3759 -> 3844 over 24 hours
    timestamps = []
    base_time = datetime.now() - timedelta(hours=24)
    
    prices = []
    base_price = 3759.0
    target_price = 3844.0
    total_periods = 288  # 24 hours * 12 (5-minute periods per hour)
    
    # Create realistic price action with trends and pullbacks
    for i in range(total_periods):
        # Overall uptrend with realistic volatility and pullbacks
        progress = i / total_periods
        
        # Base trend line
        trend_price = base_price + (target_price - base_price) * progress
        
        # Add realistic price patterns
        if i < 50:  # Initial sideways/slight up
            pattern_price = trend_price + np.sin(i * 0.2) * 8
        elif i < 150:  # Strong uptrend phase
            momentum = (i - 50) / 100
            pattern_price = trend_price + momentum * 15 + np.random.normal(0, 3)
        elif i < 200:  # Pullback/consolidation
            pullback = np.sin((i - 150) * 0.15) * 20
            pattern_price = trend_price + pullback + np.random.normal(0, 5)
        else:  # Final push higher
            momentum = (i - 200) / 88
            pattern_price = trend_price + momentum * 10 + np.random.normal(0, 4)
        
        # Add some realistic noise
        noise = np.random.normal(0, 2)
        final_price = pattern_price + noise
        
        prices.append(final_price)
        timestamps.append(base_time + timedelta(minutes=i*5))
    
    print(f"Created {len(prices)} price points (5-minute intervals)")
    print(f"Price movement: ${prices[0]:.2f} -> ${prices[-1]:.2f} ({((prices[-1]/prices[0])-1)*100:.2f}%)")
    print(f"Max price: ${max(prices):.2f}, Min price: ${min(prices):.2f}")
    
    # Create DataFrame with realistic OHLC
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = price
        else:
            open_price = prices[i-1]
        
        # Realistic high/low based on 5-minute volatility
        volatility = abs(np.random.normal(0, price * 0.002))
        high = price + volatility
        low = price - volatility
        
        # Ensure OHLC makes sense
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.uniform(1000, 2000)  # Consistent volume
        })
    
    df = pd.DataFrame(data)
    historical_data = {'ETHUSDT': df}
    
    # Create backtest engine
    engine = BacktestEngine(initial_balance=10000.0, fast_mode=False, signal_interval=3)
    
    # Run backtest
    start_date = timestamps[0]
    end_date = timestamps[-1]
    
    results = await engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"\n=== BALANCED STRATEGY RESULTS ===")
    print(f"Market Return: {((prices[-1]/prices[0])-1)*100:.2f}%")
    print(f"Strategy Return: {results['capital']['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['trades']['total']}")
    
    if results['trades']['total'] > 0:
        print(f"Win Rate: {results['trades']['win_rate_pct']:.1f}%")
        print(f"Winning Trades: {results['trades']['winning']}")
        print(f"Losing Trades: {results['trades']['losing']}")
        print(f"Avg Win: ${results['trades']['avg_win']:.3f}")
        print(f"Avg Loss: ${results['trades']['avg_loss']:.3f}")
        
        # Show recent trades
        print(f"\n=== TRADE DETAILS ===")
        for i, trade in enumerate(results['trades_detail'][-5:]):  # Last 5 trades
            entry_time = trade.entry_time.strftime("%H:%M")
            exit_time = trade.exit_time.strftime("%H:%M") 
            print(f"Trade {i+1}: {trade.side} @ ${trade.entry_price:.2f} ({entry_time}) -> ${trade.exit_price:.2f} ({exit_time})")
            print(f"  P&L: ${trade.pnl:.3f} ({trade.pnl_pct:.2f}%) - {trade.strategy}")
        
        # Check trailing stop usage
        trailing_stops = [t for t in results['trades_detail'] if 'trailing_stop' in str(t.strategy)]
        if trailing_stops:
            print(f"\nTrailing stop executions: {len(trailing_stops)}")
    else:
        print("⚠️  No trades generated - strategy may still be too restrictive")
    
    # Signal analysis
    total_signals = len(results['signals_log'])
    if total_signals > 0:
        buy_signals = len([s for s in results['signals_log'] if s['action'] == 'BUY'])
        sell_signals = len([s for s in results['signals_log'] if s['action'] == 'SELL'])
        print(f"\n=== SIGNAL ANALYSIS ===")
        print(f"Total Signals: {total_signals} (Buy: {buy_signals}, Sell: {sell_signals})")
        print(f"Signal Efficiency: {results['trades']['total']}/{total_signals} = {results['trades']['total']/max(1,total_signals):.2f}")
    else:
        print(f"\n⚠️  No signals generated during {((prices[-1]/prices[0])-1)*100:.2f}% market move")
    
    return results

if __name__ == "__main__":
    np.random.seed(123)  # For reproducible results
    asyncio.run(test_balanced_strategy())