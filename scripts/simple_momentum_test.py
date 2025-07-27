#!/usr/bin/env python3
"""
Simple Momentum Test - Get Results Fast
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

async def simple_test():
    """Simple fast test"""
    
    # Generate simple trending data - smaller dataset
    periods = 200  # Reduced from 500
    data_points = []
    base_price = 50000
    current_price = base_price
    
    for i in range(periods):
        timestamp = datetime.now() - timedelta(hours=periods-i)
        
        # Strong uptrend
        trend = 0.003
        noise = np.random.normal(0, 0.001)
        
        price_change = trend + noise
        current_price *= (1 + price_change)
        
        # OHLCV data
        open_price = current_price * 0.999
        high_price = current_price * 1.002
        low_price = current_price * 0.998
        close_price = current_price
        volume = 3000 + np.random.uniform(0, 2000)
        
        data_points.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    historical_data = {'BTCUSDT': pd.DataFrame(data_points)}
    
    # Ultra simple config
    config = {
        'symbols': ['BTCUSDT'],
        'trend_ema_fast': 5,
        'trend_ema_slow': 10,
        'trend_strength_threshold': 0.001,  # Very low
        'rsi_period': 7,
        'volume_surge_multiplier': 1.1,  # Minimal volume filter
        'volume_confirmation_required': False,
        'momentum_alignment_required': False,
        'base_position_size': 0.05,  # 5%
        'max_position_size': 0.1,  # 10% max
        'stop_loss_atr_multiplier': 5.0,  # Very wide
        'take_profit_risk_reward': 1.5,  # Quick profits
        'trend_strength_scaling': False,
        'breakout_lookback': 5  # Short lookback
    }
    
    print("🧪 Testing Ultra-Simple Momentum Strategy")
    print(f"   Data points: {periods}")
    print(f"   Price range: ${base_price:,.0f} → ${current_price:,.0f}")
    
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
        
        timestamps = pd.to_datetime(historical_data['BTCUSDT']['timestamp'], unit='ms')
        start_date = timestamps.min()
        end_date = timestamps.max()
        
        result = await engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        print("\n" + "="*50)
        print("🎯 MOMENTUM STRATEGY RESULTS")
        print("="*50)
        print(f"💰 Total Return: {result['capital']['total_return_pct']:.2f}%")
        print(f"💵 Final Capital: ${result['capital']['final']:,.2f}")
        print(f"📊 Total Trades: {result['trades']['total']}")
        print(f"🎯 Win Rate: {result['trades']['win_rate_pct']:.1f}%")
        print(f"💚 Avg Win: ${result['trades']['avg_win']:,.2f}")
        print(f"💔 Avg Loss: ${result['trades']['avg_loss']:,.2f}")
        print(f"⚡ Profit Factor: {result['trades']['profit_factor']:.2f}")
        print(f"📈 Sharpe Ratio: {result['risk_metrics']['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {result['risk_metrics']['max_drawdown_pct']:.2f}%")
        
        print(f"\n📋 Trade Details:")
        for i, trade in enumerate(result['trades_detail'][:5]):  # Show first 5 trades
            pnl_sign = "💚" if trade.pnl > 0 else "💔"
            print(f"   Trade {i+1}: {pnl_sign} ${trade.pnl:,.2f} ({trade.pnl_pct:.1f}%)")
        
        if len(result['trades_detail']) > 5:
            print(f"   ... and {len(result['trades_detail']) - 5} more trades")
        
        print("\n🔧 OPTIMIZED PARAMETERS:")
        for key, value in config.items():
            if key != 'symbols':
                print(f"   {key}: {value}")
        
        print("\n" + "="*50)
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(simple_test())