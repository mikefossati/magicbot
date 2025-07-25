#!/usr/bin/env python3
"""
Debug Environment Variables
Shows exactly what environment variables are being loaded
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_env_file():
    """Check .env file contents"""
    print("🔍 Environment File Debug")
    print("=" * 40)
    
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    print(f"📁 .env file exists: {env_path.absolute()}")
    
    # Read raw file contents
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    print(f"📄 File has {len(lines)} lines")
    
    # Parse and show relevant lines
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith('BINANCE_'):
            if '=' in line:
                key, value = line.split('=', 1)
                print(f"Line {i}: {key}={value[:10]}...{value[-10:] if len(value) > 20 else value}")
            else:
                print(f"Line {i}: {line} (malformed)")
        elif line and not line.startswith('#'):
            print(f"Line {i}: {line}")
    
    return True

def check_environment_loading():
    """Check how environment variables are loaded"""
    print("\n🔄 Environment Loading Test")
    print("=" * 40)
    
    # Method 1: Direct os.getenv (without dotenv)
    print("Method 1: Direct os.getenv")
    api_key_direct = os.getenv('BINANCE_API_KEY')
    secret_direct = os.getenv('BINANCE_SECRET_KEY')
    testnet_direct = os.getenv('BINANCE_TESTNET')
    
    print(f"   API_KEY: {'Found' if api_key_direct else 'Not found'}")
    print(f"   SECRET_KEY: {'Found' if secret_direct else 'Not found'}")
    print(f"   TESTNET: {testnet_direct}")
    
    # Method 2: Using python-dotenv
    try:
        from dotenv import load_dotenv
        print("\nMethod 2: Using python-dotenv")
        
        # Load .env file
        loaded = load_dotenv()
        print(f"   load_dotenv() result: {loaded}")
        
        api_key_dotenv = os.getenv('BINANCE_API_KEY')
        secret_dotenv = os.getenv('BINANCE_SECRET_KEY')
        testnet_dotenv = os.getenv('BINANCE_TESTNET')
        
        print(f"   API_KEY: {'Found' if api_key_dotenv else 'Not found'}")
        print(f"   SECRET_KEY: {'Found' if secret_dotenv else 'Not found'}")
        print(f"   TESTNET: {testnet_dotenv}")
        
        # Show lengths and previews
        if api_key_dotenv:
            print(f"   API_KEY length: {len(api_key_dotenv)}")
            print(f"   API_KEY preview: {api_key_dotenv[:8]}...{api_key_dotenv[-8:]}")
        
        if secret_dotenv:
            print(f"   SECRET_KEY length: {len(secret_dotenv)}")
            print(f"   SECRET_KEY preview: {secret_dotenv[:8]}...{secret_dotenv[-8:]}")
        
    except ImportError:
        print("\nMethod 2: python-dotenv not available")
        print("   Install with: pip install python-dotenv")

def check_config_loading():
    """Check how our config system loads the variables"""
    print("\n⚙️  Config System Test")
    print("=" * 40)
    
    try:
        from src.core.config import settings, config
        
        print("Settings object:")
        print(f"   binance_api_key: {'Set' if settings.binance_api_key else 'Not set'}")
        print(f"   binance_secret_key: {'Set' if settings.binance_secret_key else 'Not set'}")
        print(f"   binance_testnet: {settings.binance_testnet}")
        
        print("Config dict:")
        exchange_config = config.get('exchange', {}).get('binance', {})
        print(f"   api_key: {'Set' if exchange_config.get('api_key') else 'Not set'}")
        print(f"   secret_key: {'Set' if exchange_config.get('secret_key') else 'Not set'}")
        print(f"   testnet: {exchange_config.get('testnet')}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")

def test_binance_directly():
    """Test Binance client creation directly"""
    print("\n🔗 Direct Binance Test")
    print("=" * 40)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("❌ API keys not found in environment")
            return
        
        print(f"✅ API keys loaded from environment")
        print(f"   API key length: {len(api_key)}")
        print(f"   Secret key length: {len(secret_key)}")
        
        # Test with python-binance directly
        from binance import AsyncClient
        import asyncio
        
        async def test_client():
            try:
                client = await AsyncClient.create(
                    api_key=api_key,
                    api_secret=secret_key,
                    testnet=True
                )
                
                print("✅ AsyncClient created successfully")
                
                # Test ping
                await client.ping()
                print("✅ Ping successful")
                
                # Test account info (this is where it usually fails)
                account = await client.get_account()
                print("✅ Account info retrieved successfully!")
                print(f"   Account type: {account.get('accountType', 'unknown')}")
                
                await client.close_connection()
                
            except Exception as e:
                print(f"❌ Binance client error: {e}")
                print(f"   Error type: {type(e).__name__}")
        
        asyncio.run(test_client())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Run all debug checks"""
    print("🐛 Magicbot Environment Debug Tool\n")
    
    if not check_env_file():
        return
    
    check_environment_loading()
    check_config_loading()
    test_binance_directly()
    
    print("\n💡 Next Steps:")
    print("1. If API keys are not found, check your .env file format")
    print("2. If keys are found but Binance test fails, get new testnet keys")
    print("3. Make sure you're using https://testnet.binance.vision/ keys")

if __name__ == "__main__":
    main()