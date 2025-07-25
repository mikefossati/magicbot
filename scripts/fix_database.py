#!/usr/bin/env python3
"""
Quick script to fix the database schema
"""

import asyncio
import asyncpg
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fix_database():
    """Fix the database schema for TimescaleDB compatibility"""
    
    # Load environment variables
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://magicbot:password@localhost:5432/magicbot')
    
    # Convert asyncpg URL format
    if database_url.startswith('postgresql+asyncpg://'):
        database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    try:
        logger.info("🔧 Connecting to database...")
        conn = await asyncpg.connect(database_url)
        
        logger.info("🗑️  Dropping existing market_data table if it exists...")
        await conn.execute("DROP TABLE IF EXISTS market_data CASCADE;")
        
        logger.info("📝 Reading new schema...")
        schema_path = Path("database/schema.sql")
        
        if not schema_path.exists():
            logger.error("❌ Schema file not found. Please save the fixed schema to database/schema.sql")
            return False
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        logger.info("🔧 Executing new schema...")
        
        # Split into individual statements and execute
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            if statement:
                try:
                    logger.info(f"   Executing statement {i+1}/{len(statements)}")
                    await conn.execute(statement)
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"   Skipping (already exists): {str(e)[:100]}...")
                    else:
                        logger.warning(f"   Statement {i+1} failed: {str(e)[:200]}...")
        
        logger.info("✅ Schema updated successfully!")
        
        # Test the new schema
        logger.info("🧪 Testing new schema...")
        
        # Check if hypertable was created
        result = await conn.fetchval("""
            SELECT COUNT(*) FROM timescaledb_information.hypertables 
            WHERE table_name = 'market_data'
        """)
        
        if result > 0:
            logger.info("✅ TimescaleDB hypertable created successfully!")
        else:
            logger.warning("⚠️  Hypertable not created (might be using regular PostgreSQL)")
        
        # Test basic operations
        await conn.execute("""
            INSERT INTO market_data 
            (timestamp, symbol, open_price, high_price, low_price, close_price, volume, interval_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT DO NOTHING
        """, 1640995200000, 'TESTUSDT', 100.0, 105.0, 95.0, 102.0, 1000.0, '1h')
        
        count = await conn.fetchval("SELECT COUNT(*) FROM market_data WHERE symbol = 'TESTUSDT'")
        
        if count > 0:
            logger.info("✅ Database operations working correctly")
            await conn.execute("DELETE FROM market_data WHERE symbol = 'TESTUSDT'")
        else:
            logger.error("❌ Database test failed")
            return False
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database fix failed: {e}")
        return False

async def main():
    logger.info("🔧 Fixing Magicbot Database Schema")
    logger.info("=" * 50)
    
    success = await fix_database()
    
    if success:
        logger.info("\n🎉 Database schema fixed successfully!")
        logger.info("✅ You can now run the main setup or start the API server")
    else:
        logger.error("\n❌ Database fix failed")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        exit(1)