#!/usr/bin/env python3
"""
Debug Day Trading Strategy Issues
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager

async def debug_day_trading():
    """Debug the day trading strategy step by step"""
    
    print("🔍 Debugging Day Trading Strategy")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        print("📊 Fetching data...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',
            start_date=start_date,
            end_date=end_date
        )
        
        data = historical_data['BTCUSDT']
        print(f"✅ Got {len(data)} records")
        
        # Create strategy with minimal config
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
        
        # Test with enough data
        test_data = data.tail(100)  # Use last 100 records
        print(f"🧪 Testing with {len(test_data)} records")
        
        # Debug indicator calculation step by step
        print("\n🔧 Testing indicator calculations...")
        
        try:
            indicators = strategy._calculate_indicators(test_data)
            print("✅ Indicators calculated successfully")
            
            # Check each indicator
            for key, value in indicators.items():
                if hasattr(value, 'iloc'):
                    print(f"  {key}: Series with {len(value)} values, last = {value.iloc[-1]}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
                    
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test signal generation
        print("\n🎯 Testing signal generation...")
        try:
            signal = strategy._generate_signal('BTCUSDT', test_data, indicators)
            if signal:
                print(f"✅ Signal generated: {signal.action} at {signal.price}")
            else:
                print("⚪ No signal generated (normal)")
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Debug completed successfully!")
        
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_day_trading())