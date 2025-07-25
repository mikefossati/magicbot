import asyncio
import sys
import os
import ccxt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.ma_crossover import MovingAverageCrossover

async def test_setup():
    """Test basic functionality"""
    print("🚀 Testing Magicbot setup...")
    
    # Test exchange connection
    print("📡 Testing exchange connection...")
    exchange = BinanceExchange()
    
    try:
        await exchange.connect()
        print("✅ Exchange connection successful!")
        
        # Test getting balance
        balances = await exchange.get_account_balance()
        print(f"💰 Account has {len(balances)} assets with non-zero balance")
        
        # Test getting market data
        market_data = await exchange.get_market_data("BTCUSDT")
        print(f"📊 BTC price: ${market_data.price}")
        
        # Test strategy initialization
        print("🧠 Testing strategy...")
        strategy_config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 10,
            'slow_period': 30,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(strategy_config)
        print("✅ Strategy initialized successfully!")
        
        # Test getting historical data for strategy
        klines = await exchange.get_klines("BTCUSDT", "1h", 50)
        print(f"📈 Retrieved {len(klines)} historical data points")
        
        # Test signal generation
        signals = await strategy.generate_signals({"BTCUSDT": klines})
        print(f"🎯 Generated {len(signals)} signals")
        
        await exchange.disconnect()
        print("🎉 All tests passed! Magicbot is ready to trade!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_setup())
    if not success:
        exit(1)
