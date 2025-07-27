#!/usr/bin/env python3
"""
Quick Momentum Strategy Optimization - Relaxed Parameters
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

async def quick_test():
    """Quick test with relaxed parameters"""
    
    # Generate strong trending data
    periods = 500
    data_points = []
    base_price = 50000
    current_price = base_price
    
    for i in range(periods):
        timestamp = datetime.now() - timedelta(hours=periods-i)
        
        # Strong consistent uptrend
        trend = 0.005  # 0.5% per hour
        noise = np.random.normal(0, 0.002)  # Small noise
        
        price_change = trend + noise
        current_price *= (1 + price_change)
        
        # OHLCV data with high volume
        open_price = current_price * (1 + np.random.normal(0, 0.001))
        high_price = current_price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = current_price * (1 - abs(np.random.normal(0, 0.005)))
        close_price = current_price
        volume = 5000 + np.random.uniform(0, 10000)  # High volume
        
        data_points.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': open_price,
            'high': max(high_price, open_price, close_price),
            'low': min(low_price, open_price, close_price),
            'close': close_price,
            'volume': volume
        })
    
    historical_data = {'BTCUSDT': pd.DataFrame(data_points)}
    
    # Test configurations
    configs = [
        # Ultra Relaxed
        {
            'symbols': ['BTCUSDT'],
            'trend_ema_fast': 8,
            'trend_ema_slow': 21,
            'trend_strength_threshold': 0.005,  # Very low threshold
            'rsi_period': 10,
            'volume_surge_multiplier': 1.1,  # Very low volume requirement
            'volume_confirmation_required': False,  # Disable volume filter
            'momentum_alignment_required': False,  # Disable MACD filter
            'base_position_size': 0.05,  # Larger positions
            'stop_loss_atr_multiplier': 3.0,  # Wider stops
            'take_profit_risk_reward': 2.0,  # Lower R:R for quicker exits
            'trend_strength_scaling': True
        },
        # Volume Only
        {
            'symbols': ['BTCUSDT'],
            'trend_ema_fast': 12,
            'trend_ema_slow': 26,
            'trend_strength_threshold': 0.01,
            'rsi_period': 14,
            'volume_surge_multiplier': 1.2,
            'volume_confirmation_required': True,
            'momentum_alignment_required': False,  # Only volume filter
            'base_position_size': 0.03,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_risk_reward': 3.0,
            'trend_strength_scaling': True
        },
        # MACD Only
        {
            'symbols': ['BTCUSDT'],
            'trend_ema_fast': 12,
            'trend_ema_slow': 26,
            'trend_strength_threshold': 0.01,
            'rsi_period': 14,
            'volume_surge_multiplier': 1.5,
            'volume_confirmation_required': False,  # Only MACD filter
            'momentum_alignment_required': True,
            'base_position_size': 0.03,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_risk_reward': 3.0,
            'trend_strength_scaling': True
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Testing Configuration {i+1}/3")
        print(f"   Volume filter: {config['volume_confirmation_required']}")
        print(f"   MACD filter: {config['momentum_alignment_required']}")
        print(f"   Trend threshold: {config['trend_strength_threshold']}")
        
        try:
            # Create strategy
            strategy = MomentumTradingStrategy(config)
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=config['base_position_size']
            )
            
            # Run backtest
            engine = BacktestEngine(backtest_config)
            
            # Get date range
            timestamps = pd.to_datetime(historical_data['BTCUSDT']['timestamp'], unit='ms')
            start_date = timestamps.min()
            end_date = timestamps.max()
            
            result = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"   ✅ Result: {result['capital']['total_return_pct']:.2f}% return")
            print(f"      Trades: {result['trades']['total']}")
            print(f"      Win Rate: {result['trades']['win_rate_pct']:.1f}%")
            
            result['config'] = config
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Find best result
    if results:
        best = max(results, key=lambda x: x['capital']['total_return_pct'])
        
        print("\n" + "="*60)
        print("🎯 BEST MOMENTUM CONFIGURATION")
        print("="*60)
        print(f"💰 Total Return: {best['capital']['total_return_pct']:.2f}%")
        print(f"📊 Total Trades: {best['trades']['total']}")
        print(f"🎯 Win Rate: {best['trades']['win_rate_pct']:.1f}%")
        print(f"📈 Sharpe Ratio: {best['risk_metrics']['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {best['risk_metrics']['max_drawdown_pct']:.2f}%")
        
        print(f"\n🔧 OPTIMAL PARAMETERS:")
        for key, value in best['config'].items():
            if key != 'symbols':
                print(f"   {key}: {value}")
        
        print("\n" + "="*60)
        
        return best['config']
    else:
        print("❌ No successful configurations found")
        return None

if __name__ == "__main__":
    asyncio.run(quick_test())