#!/usr/bin/env python3
"""
Debug Aggressive Strategy
Identify where the errors are occurring in the strategy execution
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.day_trading_strategy import DayTradingStrategy
from src.data.historical_manager import HistoricalDataManager
import structlog
import traceback

# Configure detailed logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def debug_strategy():
    """Debug the aggressive strategy to find the source of errors"""
    
    print("=" * 60)
    print("AGGRESSIVE STRATEGY DEBUG SESSION")
    print("=" * 60)
    
    # Simple aggressive config
    config = {
        'symbols': ['BTCUSDT'],
        'fast_ema': 5,
        'medium_ema': 13,
        'slow_ema': 34,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_neutral_high': 60,
        'rsi_neutral_low': 40,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'volume_period': 20,
        'volume_multiplier': 1.0,
        'min_volume_ratio': 0.8,
        'pivot_period': 10,
        'support_resistance_threshold': 1.0,
        'min_signal_score': 0.5,
        'strong_signal_score': 0.8,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0,
        'trailing_stop_pct': 1.0,
        'max_daily_trades': 5,
        'session_start': '00:00',
        'session_end': '23:59',
        'position_size': 0.02,
        'leverage': 1.0,
        'use_leverage': False
    }
    
    print("🔧 Configuration loaded")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        print("📡 Exchange connected")
        
        # Get minimal recent data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)  # Just 3 days
        
        print(f"📅 Fetching data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            historical_data = await data_manager.get_multiple_symbols_data(
                symbols=['BTCUSDT'],
                interval='1h',
                start_date=start_date,
                end_date=end_date
            )
            
            if 'BTCUSDT' not in historical_data:
                print("❌ No BTCUSDT data returned")
                return
            
            data = historical_data['BTCUSDT']
            print(f"📊 Data received: {len(data)} records")
            
            if len(data) < 50:
                print(f"⚠️ Only {len(data)} records, might be insufficient")
            
            # Show data sample
            print(f"📋 Data sample:")
            print(f"   First: {data[0]}")
            print(f"   Last: {data[-1]}")
            
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            traceback.print_exc()
            return
        
        try:
            print("🔄 Creating strategy...")
            strategy = DayTradingStrategy(config)
            print("✅ Strategy created successfully")
            
        except Exception as e:
            print(f"❌ Strategy creation error: {e}")
            traceback.print_exc()
            return
        
        try:
            print("🎯 Generating signals...")
            market_data = {'BTCUSDT': data}
            
            # Add debugging to see what happens
            print("   Calling strategy.generate_signals...")
            signals = await strategy.generate_signals(market_data)
            
            print(f"✅ Signal generation completed: {len(signals)} signals")
            
            if signals:
                for i, signal in enumerate(signals):
                    print(f"   Signal {i+1}:")
                    print(f"      Action: {signal.action}")
                    print(f"      Price: ${signal.price}")
                    print(f"      Confidence: {signal.confidence:.3f}")
                    print(f"      Quantity: {signal.quantity}")
                    
                    # Check metadata
                    if signal.metadata:
                        print(f"      Stop Loss: {signal.metadata.get('stop_loss', 'N/A')}")
                        print(f"      Take Profit: {signal.metadata.get('take_profit', 'N/A')}")
                        print(f"      Volume Ratio: {signal.metadata.get('volume_ratio', 'N/A')}")
            else:
                print("   No signals generated")
                
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            traceback.print_exc()
            return
        
        # Test some calculations manually
        try:
            print("\n🧮 Manual calculation test...")
            
            # Check if we can calculate indicators manually
            print("   Testing indicator calculations...")
            
            # This might reveal where the division by zero occurs
            import pandas as pd
            df = pd.DataFrame(data)
            
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Test basic calculations
            closes = df['close']
            print(f"   Close prices range: ${closes.min():.2f} - ${closes.max():.2f}")
            
            volumes = df['volume']
            print(f"   Volume range: {volumes.min():.0f} - {volumes.max():.0f}")
            
            # Test EMA calculation
            ema_5 = closes.ewm(span=5).mean()
            print(f"   EMA-5 last value: ${ema_5.iloc[-1]:.2f}")
            
            ema_13 = closes.ewm(span=13).mean()
            print(f"   EMA-13 last value: ${ema_13.iloc[-1]:.2f}")
            
            # Test volume ratio
            vol_avg = volumes.rolling(window=20).mean()
            vol_ratio = volumes / vol_avg
            print(f"   Volume ratio last value: {vol_ratio.iloc[-1]:.2f}")
            
            # Check for any NaN or zero values that might cause division errors
            nan_count = vol_ratio.isna().sum()
            zero_count = (vol_avg == 0).sum()
            print(f"   Volume avg zeros: {zero_count}")
            print(f"   Volume ratio NaNs: {nan_count}")
            
            if zero_count > 0:
                print("   ⚠️ Found zero volume averages - this could cause division by zero")
            
        except Exception as e:
            print(f"❌ Manual calculation error: {e}")
            traceback.print_exc()
        
        print("\n✅ Debug session completed successfully")
        
    except Exception as e:
        print(f"❌ Debug session error: {e}")
        traceback.print_exc()
    
    finally:
        await exchange.disconnect()
        print("📡 Exchange disconnected")

if __name__ == "__main__":
    asyncio.run(debug_strategy())