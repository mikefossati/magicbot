#!/usr/bin/env python3
"""
Debug the specific 'builtin_function_or_method' object is not subscriptable error
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager

async def debug_subscriptable_error():
    """Debug the subscriptable error step by step"""
    
    print("🔍 Debugging 'builtin_function_or_method' object is not subscriptable Error")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get larger dataset that triggers the error
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print("📊 Fetching larger dataset...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',
            start_date=start_date,
            end_date=end_date
        )
        
        data = historical_data['BTCUSDT']
        print(f"✅ Got {len(data)} records")
        
        # Create strategy
        strategy_config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 8,
            'medium_ema': 21,
            'slow_ema': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_neutral_high': 60,
            'rsi_neutral_low': 40,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_period': 20,
            'volume_multiplier': 1.5,
            'pivot_period': 10,
            'support_resistance_threshold': 0.2,
            'stop_loss_pct': 1.5,
            'take_profit_pct': 2.5,
            'position_size': 0.02,
            'session_start': "00:00",
            'session_end': "23:59",
            'max_daily_trades': 3
        }
        
        strategy = create_strategy('day_trading_strategy', strategy_config)
        print("✅ Strategy created")
        
        # Test with incrementally larger datasets to find the breaking point
        for size in [60, 70, 80, 90, 100, 150]:
            if size > len(data):
                break
                
            print(f"\n🧪 Testing with {size} records...")
            test_data = data.tail(size)
            
            try:
                # Try the analyze_market_data method that's failing in backtest
                signal = await strategy.analyze_market_data('BTCUSDT', test_data)
                print(f"✅ Size {size}: Success - Signal: {signal.action if signal else 'None'}")
                
            except Exception as e:
                print(f"❌ Size {size}: ERROR - {str(e)}")
                print("Full traceback:")
                traceback.print_exc()
                
                # Try to isolate the exact line causing the error
                print("\n🔧 Isolating the error...")
                
                try:
                    # Check if it's in indicator calculation
                    indicators = strategy._calculate_indicators(test_data)
                    print("✅ Indicators calculated successfully")
                    
                    # Check if it's in signal generation
                    signal_result = strategy._generate_signal('BTCUSDT', test_data, indicators)
                    print(f"✅ Signal generated: {signal_result}")
                    
                except Exception as inner_e:
                    print(f"❌ Inner error: {str(inner_e)}")
                    traceback.print_exc()
                
                break  # Stop on first error
        
        print("\n✅ Debug completed!")
        
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_subscriptable_error())