#!/usr/bin/env python3
"""
Week 2 Setup Script - Simplified Version
Sets up database, runs migrations, and initializes new components
"""

import asyncio
import asyncpg
import sys
import os
from pathlib import Path
import logging

# Setup basic logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_database_connection():
    """Check if database is accessible"""
    database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://magicbot:password@localhost:5432/magicbot')
    
    # Convert asyncpg URL format
    if database_url.startswith('postgresql+asyncpg://'):
        database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    try:
        conn = await asyncpg.connect(database_url)
        await conn.fetchval('SELECT 1')
        await conn.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.info("💡 Make sure TimescaleDB is running:")
        logger.info("   Docker: docker run -d --name magicbot-db -p 5432:5432 -e POSTGRES_DB=magicbot -e POSTGRES_USER=magicbot -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg15")
        return False

async def setup_database():
    """Setup database and run migrations"""
    logger.info("🔧 Setting up database...")
    
    database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://magicbot:password@localhost:5432/magicbot')
    
    # Convert asyncpg URL format
    if database_url.startswith('postgresql+asyncpg://'):
        database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    try:
        conn = await asyncpg.connect(database_url)
        
        # Read and execute schema
        schema_path = Path("database/schema.sql")
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Split and execute each statement
            statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement:
                    try:
                        await conn.execute(statement)
                    except Exception as e:
                        # Some statements might fail if tables already exist
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Statement failed: {e}")
            
            logger.info("✅ Database schema created successfully")
        else:
            logger.error("❌ Schema file not found. Please create database/schema.sql")
            return False
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

async def test_database_operations():
    """Test basic database operations"""
    logger.info("🧪 Testing database operations...")
    
    database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://magicbot:password@localhost:5432/magicbot')
    
    # Convert asyncpg URL format
    if database_url.startswith('postgresql+asyncpg://'):
        database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    try:
        conn = await asyncpg.connect(database_url)
        
        # Test market data insertion
        test_data = {
            'symbol': 'TESTUSDT',
            'timestamp': 1640995200000,  # 2022-01-01
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0,
            'interval': '1h'
        }
        
        query = """
        INSERT INTO market_data 
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, interval_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (symbol, timestamp, interval_type) DO NOTHING
        """
        
        await conn.execute(
            query,
            test_data['symbol'],
            test_data['timestamp'],
            test_data['open'],
            test_data['high'],
            test_data['low'],
            test_data['close'],
            test_data['volume'],
            test_data['interval']
        )
        
        # Verify insertion
        result = await conn.fetchval(
            "SELECT COUNT(*) FROM market_data WHERE symbol = $1",
            test_data['symbol']
        )
        
        if result > 0:
            logger.info("✅ Database operations working correctly")
            
            # Clean up test data
            await conn.execute("DELETE FROM market_data WHERE symbol = $1", test_data['symbol'])
            await conn.close()
            return True
        else:
            logger.error("❌ Database test failed")
            await conn.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Database test error: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "logs",
        "database",
        "src/database",
        "src/risk",
        "src/web/templates", 
        "src/web/static",
        "src/logging",
        "backtest_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def create_schema_file():
    """Create the database schema file if it doesn't exist"""
    schema_path = Path("database/schema.sql")
    
    if schema_path.exists():
        logger.info("✅ Schema file already exists")
        return
    
    logger.info("📝 Creating database schema file...")
    
    schema_content = """-- Historical market data
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    interval_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval_type)
);

-- Index for fast queries
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
ON market_data (symbol, timestamp DESC);

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_time TIMESTAMP NOT NULL,
    action VARCHAR(10) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy_time 
ON signals (strategy_name, signal_time DESC);

-- Executed trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    pnl DECIMAL(20,8),
    pnl_percentage DECIMAL(10,4),
    commission DECIMAL(20,8) DEFAULT 0,
    slippage DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'OPEN',
    exchange_order_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_strategy_time 
ON trades (strategy_name, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_status 
ON trades (symbol, status);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    avg_entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, strategy_name)
);

-- Risk events and violations
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    strategy_name VARCHAR(100),
    description TEXT NOT NULL,
    current_value DECIMAL(20,8),
    threshold_value DECIMAL(20,8),
    action_taken VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy performance metrics
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(10,4) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4),
    win_rate DECIMAL(5,2),
    avg_trade_duration INTERVAL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_name, date)
);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    logger VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function VARCHAR(100),
    line_number INTEGER,
    exception TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_level_time 
ON system_logs (level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_logger_time 
ON system_logs (logger, created_at DESC);
"""
    
    with open(schema_path, 'w') as f:
        f.write(schema_content)
    
    logger.info("✅ Schema file created")

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'asyncpg',
        'fastapi',
        'jinja2',
        'structlog'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ Missing required packages:")
        for package in missing_packages:
            logger.error(f"   - {package}")
        logger.info("💡 Install with: pip install asyncpg jinja2 python-multipart")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def check_environment():
    """Check environment configuration"""
    logger.info("🔧 Checking environment configuration...")
    
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning("⚠️  .env file not found")
        logger.info("💡 Make sure to create .env with DATABASE_URL")
        return False
    
    # Load .env file manually
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    required_vars = ['DATABASE_URL', 'BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if var not in env_vars and var not in os.environ:
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning("⚠️  Missing environment variables:")
        for var in missing_vars:
            logger.warning(f"   - {var}")
        return False
    
    logger.info("✅ Environment configuration looks good")
    return True

async def main():
    """Main setup function"""
    logger.info("🚀 Magicbot Week 2 Setup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        return False
    
    # Create directories
    setup_directories()
    
    # Create schema file
    create_schema_file()
    
    # Check database connection
    if not await check_database_connection():
        logger.error("❌ Database connection failed")
        logger.info("\n💡 To start TimescaleDB with Docker:")
        logger.info("docker run -d --name magicbot-db \\")
        logger.info("  -p 5432:5432 \\")
        logger.info("  -e POSTGRES_DB=magicbot \\")
        logger.info("  -e POSTGRES_USER=magicbot \\")
        logger.info("  -e POSTGRES_PASSWORD=password \\")
        logger.info("  timescale/timescaledb:latest-pg15")
        return False
    
    # Setup database
    db_success = await setup_database()
    if not db_success:
        logger.error("❌ Database setup failed")
        return False
    
    # Test database
    test_success = await test_database_operations()
    if not test_success:
        logger.error("❌ Database test failed")
        return False
    
    logger.info("\n🎉 Week 2 setup completed successfully!")
    logger.info("\n📋 Next steps:")
    logger.info("1. Make sure all the source files are in place:")
    logger.info("   - src/database/connection.py")
    logger.info("   - src/web/dashboard.py")
    logger.info("   - src/logging/logger_config.py")
    logger.info("2. Start the enhanced API server: python -m src.api.main")
    logger.info("3. Visit http://localhost:8000 to see the dashboard")
    logger.info("4. Run backtests with database storage enabled")
    
    return True

if __name__ == "__main__":
    # Load environment variables from .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    success = asyncio.run(main())
    if not success:
        sys.exit(1)