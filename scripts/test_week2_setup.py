#!/usr/bin/env python3
"""Test Week 2 setup"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def test_database():
    """Test database connection and operations"""
    print("🔧 Testing database connection...")
    
    try:
        from src.database.connection import db
        
        # Initialize database
        await db.initialize()
        print("✅ Database connection initialized")
        
        # Test basic query
        result = await db.fetch_one("SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'")
        print(f"✅ Found {result['table_count']} tables in database")
        
        # Test table creation
        await db.execute("""
            INSERT INTO market_data 
            (timestamp, symbol, open_price, high_price, low_price, close_price, volume, interval_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT DO NOTHING
        """, 1640995200000, 'TEST2USDT', 100.0, 105.0, 95.0, 102.0, 1000.0, '1h')
        
        count = await db.fetch_one("SELECT COUNT(*) as count FROM market_data WHERE symbol = 'TEST2USDT'")
        print(f"✅ Database operations working: {count['count']} test records")
        
        # Cleanup
        await db.execute("DELETE FROM market_data WHERE symbol = 'TEST2USDT'")
        await db.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def main():
    print("🧪 Testing Week 2 Setup")
    print("=" * 40)
    
    # Test database
    db_success = await test_database()
    
    if db_success:
        print("\n🎉 Week 2 database setup is working!")
        print("\n📋 Next steps:")
        print("1. Create remaining source files (I'll help with this)")
        print("2. Start the enhanced API server")
        print("3. Access the web dashboard")
    else:
        print("\n❌ Week 2 setup has issues")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)