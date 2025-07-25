#!/usr/bin/env python3
"""
API Key Validation Script for Magicbot Trading System
Checks if your Binance API keys are properly configured

Usage: python scripts/validate_api_keys.py
"""

import os
import sys
import re
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    env_vars = {}
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                else:
                    print(f"⚠️  Warning: Line {line_num} in .env file is malformed: {line}")
    
    return env_vars

def validate_api_keys():
    """Validate API key configuration"""
    print("🔐 Magicbot API Key Validator")
    print("=" * 50)
    
    # Check if .env file exists
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        print("📝 Please copy config/.env.template to .env and fill in your API keys")
        print("\nTo fix this:")
        print("1. cp config/.env.template .env")
        print("2. Edit .env with your Binance testnet API keys")
        return False
    
    print(f"📁 .env file: ✅ Found ({env_path.absolute()})")
    
    # Load environment variables
    env_vars = load_env_file()
    
    # Check required variables
    required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY', 'BINANCE_TESTNET']
    missing_vars = []
    
    for var in required_vars:
        if var in env_vars and env_vars[var]:
            print(f"🔑 {var}: ✅ Set")
        else:
            print(f"🔑 {var}: ❌ Missing or empty")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing required variables: {', '.join(missing_vars)}")
        print("\nPlease add these to your .env file:")
        print("BINANCE_API_KEY=your_testnet_api_key_here")
        print("BINANCE_SECRET_KEY=your_testnet_secret_key_here")
        print("BINANCE_TESTNET=true")
        return False
    
    # Get the values
    api_key = env_vars.get('BINANCE_API_KEY', '')
    secret_key = env_vars.get('BINANCE_SECRET_KEY', '')
    testnet = env_vars.get('BINANCE_TESTNET', '')
    
    print(f"\n🔍 Detailed Analysis:")
    print("-" * 30)
    
    # Validate API key format
    print(f"API Key Analysis:")
    print(f"   Length: {len(api_key)} characters")
    print(f"   Preview: {api_key[:8]}...{api_key[-8:] if len(api_key) >= 16 else api_key}")
    
    print(f"\nSecret Key Analysis:")
    print(f"   Length: {len(secret_key)} characters")
    print(f"   Preview: {secret_key[:8]}...{secret_key[-8:] if len(secret_key) >= 16 else secret_key}")
    
    print(f"\nTestnet Setting: {testnet}")
    
    # Check for common format issues
    issues = []
    warnings = []
    
    # API Key validation
    if len(api_key) != 64:
        issues.append(f"❌ API key should be exactly 64 characters, got {len(api_key)}")
    
    if not re.match(r'^[A-Za-z0-9]+$', api_key):
        issues.append("❌ API key should only contain letters and numbers")
    
    if api_key.startswith(' ') or api_key.endswith(' '):
        issues.append("❌ API key has leading/trailing spaces")
    
    # Secret Key validation
    if len(secret_key) != 64:
        issues.append(f"❌ Secret key should be exactly 64 characters, got {len(secret_key)}")
    
    if not re.match(r'^[A-Za-z0-9]+$', secret_key):
        issues.append("❌ Secret key should only contain letters and numbers")
    
    if secret_key.startswith(' ') or secret_key.endswith(' '):
        issues.append("❌ Secret key has leading/trailing spaces")
    
    # Testnet validation
    if testnet.lower() != 'true':
        issues.append(f"❌ BINANCE_TESTNET should be 'true' for development, got '{testnet}'")
    
    # Check for quotes in the .env file
    with open('.env', 'r') as f:
        env_content = f.read()
        
    if '"' in env_content or "'" in env_content:
        warnings.append("⚠️  .env file contains quotes - make sure API keys don't have quotes around them")
    
    # Check if keys look like they might be from live Binance
    if api_key and not any(char in api_key.lower() for char in 'abcdef'):
        warnings.append("⚠️  API key doesn't contain hex characters - are you sure this is from Binance?")
    
    # Display results
    print(f"\n📋 Validation Results:")
    print("-" * 30)
    
    if issues:
        print("🚨 Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print()
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    if not issues and not warnings:
        print("✅ All validations passed!")
    
    return len(issues) == 0

def check_testnet_instructions():
    """Provide instructions for getting testnet API keys"""
    print("\n🧪 Binance Testnet Setup Instructions:")
    print("=" * 50)
    print("1. 🌐 Go to: https://testnet.binance.vision/")
    print("2. 🔐 Login with your GitHub account")
    print("3. 👤 Click your profile icon (top right)")
    print("4. ⚙️  Select 'API Management'")
    print("5. ➕ Click 'Create API Key'")
    print("6. ✅ Enable 'Spot & Margin Trading'")
    print("7. 📋 Copy both API Key and Secret Key")
    print("8. 🔄 Paste them into your .env file")
    
    print("\n⚠️  Important Notes:")
    print("   • Use TESTNET keys (testnet.binance.vision), NOT live Binance keys")
    print("   • Don't add quotes around the keys in .env")
    print("   • Make sure there are no extra spaces")
    print("   • Keep BINANCE_TESTNET=true")

def check_file_permissions():
    """Check if .env file has proper permissions"""
    env_path = Path('.env')
    if env_path.exists():
        stat = env_path.stat()
        mode = oct(stat.st_mode)[-3:]
        
        print(f"\n🔒 File Permissions:")
        print(f"   .env file permissions: {mode}")
        
        if mode != '600':
            print("   ⚠️  Consider setting .env permissions to 600 for security:")
            print("   chmod 600 .env")

def test_environment_loading():
    """Test if environment variables can be loaded properly"""
    print(f"\n🔄 Environment Loading Test:")
    print("-" * 30)
    
    try:
        # Try to load with python-dotenv if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            testnet = os.getenv('BINANCE_TESTNET')
            
            print("✅ python-dotenv loaded successfully")
            print(f"   API_KEY loaded: {'✅' if api_key else '❌'}")
            print(f"   SECRET_KEY loaded: {'✅' if secret_key else '❌'}")
            print(f"   TESTNET loaded: {'✅' if testnet else '❌'}")
            
        except ImportError:
            print("⚠️  python-dotenv not available, using manual parsing")
            
    except Exception as e:
        print(f"❌ Error loading environment: {e}")

def main():
    """Main validation function"""
    print("🚀 Starting Magicbot API Key Validation...\n")
    
    success = validate_api_keys()
    
    if success:
        print("\n🎉 Configuration looks good!")
        check_file_permissions()
        test_environment_loading()
        print("\n✅ You can now try running:")
        print("   python scripts/test_connection.py")
    else:
        print("\n🔧 Please fix the issues above.")
        check_testnet_instructions()
        print("\n🔄 After fixing, run this script again:")
        print("   python scripts/validate_api_keys.py")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)