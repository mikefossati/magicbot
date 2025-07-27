#!/usr/bin/env python3
"""
Minimal Momentum Optimizer - Quick Parameter Testing
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Disable ALL logging
import logging
logging.disable(logging.CRITICAL)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

async def quick_test():
    """Ultra-minimal optimization test"""
    
    # Generate 100 data points (4 days hourly)
    periods = 100
    data_points = []
    price = 50000
    
    for i in range(periods):
        timestamp = datetime.now() - timedelta(hours=periods-i)
        
        # Simple uptrend
        price *= 1.002  # 0.2% per hour
        noise = np.random.normal(0, 0.002)
        price *= (1 + noise)
        
        data_points.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': price * 0.999,
            'high': price * 1.001,
            'low': price * 0.999,
            'close': price,
            'volume': 3000
        })
    
    data = {'BTCUSDT': pd.DataFrame(data_points)}
    market_return = ((price / 50000) - 1) * 100
    
    # Test 3 key configurations
    configs = [
        # Current (ultra-fast)
        {'name': 'current', 'fast': 5, 'slow': 10, 'threshold': 0.001, 'rsi': 7, 'size': 0.05, 'rr': 1.5},
        # Balanced
        {'name': 'balanced', 'fast': 8, 'slow': 21, 'threshold': 0.002, 'rsi': 14, 'size': 0.03, 'rr': 2.5},
        # Conservative
        {'name': 'conservative', 'fast': 12, 'slow': 26, 'threshold': 0.005, 'rsi': 14, 'size': 0.02, 'rr': 3.0}
    ]
    
    print(f"🚀 Quick Momentum Optimization (Market: +{market_return:.1f}%)")
    
    results = []
    for config in configs:
        try:
            # Build full config
            full_config = {
                'symbols': ['BTCUSDT'],
                'trend_ema_fast': config['fast'],
                'trend_ema_slow': config['slow'],
                'trend_strength_threshold': config['threshold'],
                'rsi_period': config['rsi'],
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': config['size'],
                'max_position_size': config['size'] * 2,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': config['rr'],
                'max_risk_per_trade': 0.02
            }
            
            # Test strategy
            strategy = MomentumTradingStrategy(full_config)
            engine = BacktestEngine(BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                position_sizing='percentage',
                position_size=config['size']
            ))
            
            timestamps = pd.to_datetime(data['BTCUSDT']['timestamp'], unit='ms')
            result = await engine.run_backtest(strategy, data, timestamps.min(), timestamps.max())
            
            return_pct = result['capital']['total_return_pct']
            trades = result['trades']['total']
            win_rate = result['trades']['win_rate_pct']
            
            print(f"   {config['name']:12s}: {return_pct:6.1f}% ({trades:2d} trades, {win_rate:4.1f}% win)")
            
            results.append({
                'name': config['name'],
                'config': config,
                'return': return_pct,
                'trades': trades,
                'win_rate': win_rate
            })
            
        except Exception as e:
            print(f"   {config['name']:12s}: ERROR - {str(e)[:50]}")
    
    if results:
        best = max(results, key=lambda x: x['return'])
        print(f"\n🏆 Best: {best['name']} with {best['return']:.1f}% return")
        print(f"   EMA: {best['config']['fast']}/{best['config']['slow']}")
        print(f"   Threshold: {best['config']['threshold']}")
        print(f"   RSI Period: {best['config']['rsi']}")
        print(f"   Position Size: {best['config']['size']}")
        print(f"   Risk/Reward: {best['config']['rr']}")
        return best['config']
    
    print("❌ No successful results")
    return None

if __name__ == "__main__":
    best = asyncio.run(quick_test())