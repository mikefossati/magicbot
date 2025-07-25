#!/usr/bin/env python3
"""
Test strategy signal generation
"""

import asyncio
import sys
import os
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.ma_crossover import MovingAverageCrossover

async def test_strategy_signals():
    """Test if the strategy generates signals with real data"""
    print("🧠 Testing Strategy Signal Generation")
    print("=" * 50)
    
    exchange = None
    
    try:
        # Connect to exchange
        exchange = BinanceExchange()
        await exchange.connect()
        print("✅ Connected to Binance")
        
        # Get recent data
        print("📈 Getting recent market data...")
        klines = await exchange.get_klines("BTCUSDT", "1h", 100)
        print(f"✅ Retrieved {len(klines)} candles")
        
        if len(klines) < 50:
            print("❌ Not enough data for strategy testing")
            return False
        
        # Show data range
        first_time = datetime.fromtimestamp(klines[0]['timestamp'] / 1000)
        last_time = datetime.fromtimestamp(klines[-1]['timestamp'] / 1000)
        print(f"📅 Data range: {first_time.strftime('%Y-%m-%d %H:%M')} to {last_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Test different strategy configurations
        test_configs = [
            {"fast": 5, "slow": 10, "name": "Very Fast (5/10)"},
            {"fast": 10, "slow": 20, "name": "Fast (10/20)"},
            {"fast": 10, "slow": 30, "name": "Default (10/30)"},
            {"fast": 20, "slow": 50, "name": "Slow (20/50)"},
        ]
        
        for config in test_configs:
            print(f"\n🔧 Testing {config['name']} configuration...")
            
            try:
                # Create strategy
                strategy_config = {
                    'symbols': ['BTCUSDT'],
                    'fast_period': config['fast'],
                    'slow_period': config['slow'],
                    'position_size': 0.1
                }
                
                strategy = MovingAverageCrossover(strategy_config)
                print(f"   ✅ Strategy created: {config['fast']}/{config['slow']} periods")
                
                # Test signal generation
                market_data = {"BTCUSDT": klines}
                signals = await strategy.generate_signals(market_data)
                
                print(f"   🎯 Generated {len(signals)} signals")
                
                if signals:
                    for i, signal in enumerate(signals):
                        print(f"      Signal {i+1}: {signal.action} at ${signal.price:.2f} (confidence: {signal.confidence:.2f})")
                        if signal.metadata:
                            print(f"                  Fast MA: {signal.metadata.get('fast_ma', 'N/A'):.2f}")
                            print(f"                  Slow MA: {signal.metadata.get('slow_ma', 'N/A'):.2f}")
                else:
                    print("      📊 No signals generated (normal - depends on market conditions)")
                
                # Show moving averages for debugging
                import pandas as pd
                df = pd.DataFrame(klines)
                df['close'] = df['close'].astype(float)
                
                fast_ma = df['close'].rolling(window=config['fast']).mean()
                slow_ma = df['close'].rolling(window=config['slow']).mean()
                
                if len(fast_ma) >= config['slow']:
                    current_fast = fast_ma.iloc[-1]
                    current_slow = slow_ma.iloc[-1]
                    prev_fast = fast_ma.iloc[-2]
                    prev_slow = slow_ma.iloc[-2]
                    
                    print(f"      📊 Current MA values:")
                    print(f"         Fast MA: {current_fast:.2f} (prev: {prev_fast:.2f})")
                    print(f"         Slow MA: {current_slow:.2f} (prev: {prev_slow:.2f})")
                    
                    if current_fast > current_slow:
                        print(f"         📈 Fast MA is above Slow MA (bullish)")
                    else:
                        print(f"         📉 Fast MA is below Slow MA (bearish)")
                
            except Exception as e:
                print(f"   ❌ Error testing strategy: {e}")
        
        # Test with ETH data too
        print(f"\n🔄 Testing with ETH data...")
        try:
            eth_klines = await exchange.get_klines("ETHUSDT", "1h", 50)
            print(f"✅ Retrieved {len(eth_klines)} ETH candles")
            
            strategy_config = {
                'symbols': ['ETHUSDT'],
                'fast_period': 10,
                'slow_period': 30,
                'position_size': 0.1
            }
            
            strategy = MovingAverageCrossover(strategy_config)
            market_data = {"ETHUSDT": eth_klines}
            signals = await strategy.generate_signals(market_data)
            
            print(f"🎯 ETH signals generated: {len(signals)}")
            for signal in signals:
                print(f"   📈 {signal.action} ETH at ${signal.price:.2f}")
        
        except Exception as e:
            print(f"❌ ETH test failed: {e}")
        
        print(f"\n💡 Strategy Testing Summary:")
        print("   • If no signals are generated, it means the market conditions")
        print("     don't meet the crossover criteria")
        print("   • Try different fast/slow period combinations")
        print("   • Signals are only generated when MAs cross over")
        print("   • This is normal behavior for trend-following strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        if exchange:
            await exchange.disconnect()

def main():
    print("🧪 Strategy Signal Generation Test\n")
    
    success = asyncio.run(test_strategy_signals())
    
    if success:
        print("\n✅ Strategy testing completed!")
    else:
        print("\n❌ Strategy testing failed")
    
    return success

if __name__ == "__main__":
    main()