import asyncio
import asyncpg
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import structlog
import os
import time
from collections import deque

logger = structlog.get_logger()

class DatabaseManager:
    """Manages database connections and operations with dynamic pool sizing"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://magicbot:password@localhost:5432/magicbot')
        
        # Convert asyncpg URL format
        if self.database_url.startswith('postgresql+asyncpg://'):
            self.database_url = self.database_url.replace('postgresql+asyncpg://', 'postgresql://')
        
        # Dynamic pool sizing parameters
        self.min_size = 5
        self.max_size = 50
        self.target_pool_size = self.min_size
        self.usage_history = deque(maxlen=100)  # Track recent connection usage
        self.last_resize_time = 0
        self.resize_cooldown = 30  # seconds
        self.high_usage_threshold = 0.8  # 80% of pool in use
        self.low_usage_threshold = 0.3   # 30% of pool in use
        
    async def initialize(self) -> None:
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60,
                server_settings={
                    'application_name': 'magicbot_optimization',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3'
                },
                max_queries=50000,
                max_inactive_connection_lifetime=300.0
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            logger.info("Database connection pool initialized", 
                       min_size=self.min_size, 
                       max_size=self.max_size)
            
            # Start background task for pool monitoring
            asyncio.create_task(self._monitor_pool_usage())
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        start_time = time.time()
        try:
            async with self.pool.acquire() as conn:
                # Record usage metrics
                acquire_time = time.time() - start_time
                current_usage = len(self.pool._queue._getters) / self.pool.get_max_size()
                self.usage_history.append({
                    'timestamp': time.time(),
                    'acquire_time': acquire_time,
                    'usage_ratio': current_usage
                })
                yield conn
        except asyncio.TimeoutError:
            # Connection acquisition timed out - potential pool starvation
            logger.warning("Connection acquisition timeout", 
                          pool_size=self.pool.get_size(),
                          max_size=self.pool.get_max_size())
            raise
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch single row"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetch_all(self, query: str, *args) -> List[Dict]:
        """Fetch multiple rows"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def _monitor_pool_usage(self):
        """Background task to monitor and adjust pool size based on usage patterns"""
        while self.pool:
            try:
                await asyncio.sleep(self.resize_cooldown)
                await self._analyze_and_resize_pool()
            except Exception as e:
                logger.error("Error in pool monitoring", error=str(e))
                await asyncio.sleep(self.resize_cooldown)
    
    async def _analyze_and_resize_pool(self):
        """Analyze usage patterns and resize pool if needed"""
        if not self.usage_history or time.time() - self.last_resize_time < self.resize_cooldown:
            return
        
        # Calculate average usage over recent history
        recent_usage = [entry['usage_ratio'] for entry in self.usage_history 
                       if time.time() - entry['timestamp'] < 300]  # Last 5 minutes
        
        if not recent_usage:
            return
        
        avg_usage = sum(recent_usage) / len(recent_usage)
        max_usage = max(recent_usage)
        current_size = self.pool.get_size()
        
        # Determine if we need to scale up or down
        should_scale_up = (avg_usage > self.high_usage_threshold or 
                          max_usage > 0.9) and current_size < self.max_size
        
        should_scale_down = (avg_usage < self.low_usage_threshold and 
                            current_size > self.min_size)
        
        if should_scale_up:
            new_size = min(current_size + 5, self.max_size)
            await self._resize_pool(new_size)
            logger.info("Scaled pool up due to high usage", 
                       old_size=current_size, 
                       new_size=new_size,
                       avg_usage=avg_usage)
        
        elif should_scale_down:
            new_size = max(current_size - 2, self.min_size)
            await self._resize_pool(new_size)
            logger.info("Scaled pool down due to low usage", 
                       old_size=current_size, 
                       new_size=new_size,
                       avg_usage=avg_usage)
    
    async def _resize_pool(self, new_size: int):
        """Resize the connection pool"""
        try:
            # Create new pool with updated size
            old_pool = self.pool
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=min(new_size, self.min_size),
                max_size=max(new_size, self.min_size),
                command_timeout=60
            )
            
            # Close old pool
            if old_pool:
                await old_pool.close()
            
            self.last_resize_time = time.time()
            
        except Exception as e:
            logger.error("Failed to resize pool", error=str(e))
            # Keep the old pool if resize fails
            if 'old_pool' in locals():
                self.pool = old_pool

# Global database instance
db = DatabaseManager()

async def get_historical_data(
    start_date,
    end_date,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h"
):
    """
    Get historical market data for backtesting.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        symbol: Trading symbol
        timeframe: Data timeframe
        
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    import pandas as pd
    import numpy as np
    
    logger.info("Fetching historical data",
               symbol=symbol,
               start_date=start_date,
               end_date=end_date,
               timeframe=timeframe)
    
    # Generate sample data for development
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Simulate price data with some randomness
    np.random.seed(42)  # For reproducible results
    
    base_price = 50000  # Starting price for BTC
    price_changes = np.random.normal(0, 0.02, len(date_range))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    # Create OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(date_range, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        
        # Ensure OHLC relationships are valid
        high = max(high, price)
        low = min(low, price)
        
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info("Historical data fetched",
               symbol=symbol,
               records=len(df),
               date_range=f"{df.index[0]} to {df.index[-1]}")
    
    return {symbol: df}