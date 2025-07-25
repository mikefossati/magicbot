#!/usr/bin/env python3
"""
Super simple test using standard python-binance
No custom sessions, no complex SSL handling
"""

import asyncio
import os
import sys
import warnings
from dotenv import load_dotenv
from binance import AsyncClient

# Suppress warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

async def super_simple_test():
    """Very basic test of Binance API"""
    print("🧪 Super Simple Binance Test")
    print("=" * 40)
    
    # Load environment
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API keys not found")
        print("Make sure your .env file has:")
        print("BINANCE_API_KEY=your_key")
        print("BINANCE_SECRET_KEY=your_secret")
        return False
    
    print(f"🔑 Using API key: {api_key[:8]}...{api_key[-8:]}")
    
    # Disable SSL verification for development
    original_verify = os.environ.get('PYTHONHTTPSVERIFY')
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    client = None
    
    try:
        # Create client with basic settings
        print("🔄 Creating Binance client...")
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=secret_key,
            testnet=True
        )
        print("✅ Client created")
        
        # Test ping
        print("🏓 Testing connection...")
        await client.ping()
        print("✅ Connection successful")
        
        # Test account info
        print("💰 Getting account info...")
        account = await client.get_account()
        print("✅ Account info retrieved")
        
        # Show account details
        can_trade = account.get('canTrade', False)
        account_type = account.get('accountType', 'SPOT')
        print(f"   Account type: {account_type}")
        print(f"   Trading enabled: {can_trade}")
        
        # Show balances
        balances = account.get('balances', [])
        non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
        
        print(f"   Assets with balance: {len(non_zero_balances)}")
        
        if non_zero_balances:
            print("   Your testnet funds:")
            for balance in non_zero_balances[:5]:  # Show first 5
                total = float(balance['free']) + float(balance['locked'])
                print(f"     {balance['asset']}: {total}")
        else:
            print("   ⚠️  No funds found in testnet account")
            print("   Go to https://testnet.binance.vision/ and click 'Get Test Funds'")
        
        # Test market data
        print("📊 Getting BTC price...")
        ticker = await client.get_symbol_ticker(symbol="BTCUSDT")
        btc_price = float(ticker['price'])
        print(f"✅ BTC price: ${btc_price:,.2f}")
        
        # Test historical data
        print("📈 Getting recent price data...")
        klines = await client.get_klines(symbol="BTCUSDT", interval="1h", limit=5)
        print(f"✅ Got {len(klines)} price candles")
        
        if klines:
            latest = klines[-1]
            open_price = float(latest[1])
            close_price = float(latest[4])
            print(f"   Latest 1h candle: ${open_price:.2f} → ${close_price:.2f}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Binance testnet connection is working perfectly!")
        print("\n🚀 Next step: Run a backtest!")
        print("   python scripts/run_backtest.py --symbols BTCUSDT --start-date 2024-01-01 --end-date 2024-01-31")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Error-specific help
        error_str = str(e).lower()
        if "api-key format invalid" in error_str:
            print("\n🔑 API Key Problem:")
            print("   Your API keys are not valid")
            print("   1. Go to https://testnet.binance.vision/")
            print("   2. Create new API keys")
            print("   3. Enable 'Spot & Margin Trading'")
            print("   4. Update your .env file")
        elif "signature" in error_str:
            print("\n🔐 Secret Key Problem:")
            print("   Your secret key might be wrong")
        elif "timestamp" in error_str:
            print("\n⏰ Time sync issue - try again in a moment")
        elif "ssl" in error_str or "certificate" in error_str:
            print("\n🔒 SSL issue persists")
            print("   Try: pip install --upgrade certifi urllib3")
        
        return False
    
    finally:
        # Cleanup
        if client:
            try:
                await client.close_connection()
            except:
                pass
        
        # Restore SSL setting
        if original_verify is not None:
            os.environ['PYTHONHTTPSVERIFY'] = original_verify
        elif 'PYTHONHTTPSVERIFY' in os.environ:
            del os.environ['PYTHONHTTPSVERIFY']

def main():
    print("🔬 Running super simple API test...")
    print("This test uses basic python-binance functionality\n")
    
    success = asyncio.run(super_simple_test())
    
    if success:
        print("\n✅ SUCCESS! Your setup works!")
    else:
        print("\n❌ Test failed - check the error messages above")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test cancelled")
        sys.exit(1)