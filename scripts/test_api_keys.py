#!/usr/bin/env python3
"""
Test script to verify API key loading and provide options for development
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
import structlog

logger = structlog.get_logger()

def main():
    print("🔑 Testing API Key Configuration...")
    
    # Test configuration loading
    print(f"📊 Configuration loaded:")
    print(f"   - Testnet: {settings.binance_testnet}")
    print(f"   - API Key: {settings.binance_api_key[:10]}...{settings.binance_api_key[-10:] if len(settings.binance_api_key) > 20 else 'SHORT/INVALID'}")
    print(f"   - Secret Key: {settings.binance_secret_key[:10]}...{settings.binance_secret_key[-10:] if len(settings.binance_secret_key) > 20 else 'SHORT/INVALID'}")
    
    # Validate key format
    if len(settings.binance_api_key) < 20 or len(settings.binance_secret_key) < 20:
        print("❌ API keys appear to be too short or missing")
        print("\n💡 Solutions:")
        print("1. Get new Binance testnet API keys from: https://testnet.binance.vision/")
        print("2. Update your .env file with valid keys")
        print("3. Or use the system without API keys for backtesting only")
        return False
    
    print("✅ API keys format looks correct")
    print("\n📝 Note: If you're still getting 'API-key format invalid' errors,")
    print("   the keys may be expired. Binance testnet keys expire periodically.")
    print("   Get fresh keys from: https://testnet.binance.vision/")
    
    return True

if __name__ == "__main__":
    main()
