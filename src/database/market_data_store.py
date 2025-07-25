import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import structlog
from .connection import db

logger = structlog.get_logger()

class MarketDataStore:
    """Handles storage and retrieval of market data"""
    
    async def store_market_data(
        self, 
        symbol: str, 
        data: List[Dict], 
        interval: str
    ) -> int:
        """Store market data in database"""
        if not data:
            return 0
        
        query = """
        INSERT INTO market_data 
        (symbol, timestamp, open_price, high_price, low_price, close_price, volume, interval_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (symbol, timestamp, interval_type) 
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume
        """
        
        records_inserted = 0
        
        async with db.get_connection() as conn:
            async with conn.transaction():
                for candle in data:
                    try:
                        await conn.execute(
                            query,
                            symbol,
                            candle['timestamp'],
                            float(candle['open']),
                            float(candle['high']),
                            float(candle['low']),
                            float(candle['close']),
                            float(candle['volume']),
                            interval
                        )
                        records_inserted += 1
                    except Exception as e:
                        logger.warning("Failed to insert candle", 
                                     symbol=symbol, 
                                     timestamp=candle['timestamp'],
                                     error=str(e))
        
        logger.info("Market data stored", 
                   symbol=symbol, 
                   interval=interval,
                   records=records_inserted)
        
        return records_inserted
    
    async def get_market_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """Retrieve market data from database"""
        
        query = """
        SELECT timestamp, open_price, high_price, low_price, close_price, volume
        FROM market_data
        WHERE symbol = $1 
        AND interval_type = $2
        AND timestamp >= $3
        AND timestamp <= $4
        ORDER BY timestamp ASC
        LIMIT $5
        """
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        rows = await db.fetch_all(
            query, 
            symbol, 
            interval, 
            start_timestamp, 
            end_timestamp, 
            limit
        )
        
        # Convert to expected format
        market_data = []
        for row in rows:
            market_data.append({
                'timestamp': row['timestamp'],
                'open': float(row['open_price']),
                'high': float(row['high_price']),
                'low': float(row['low_price']),
                'close': float(row['close_price']),
                'volume': float(row['volume'])
            })
        
        logger.debug("Market data retrieved", 
                    symbol=symbol,
                    interval=interval,
                    records=len(market_data))
        
        return market_data
    
    async def get_latest_data_time(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get timestamp of latest data for symbol"""
        query = """
        SELECT MAX(timestamp) as latest_time
        FROM market_data
        WHERE symbol = $1 AND interval_type = $2
        """
        
        row = await db.fetch_one(query, symbol, interval)
        
        if row and row['latest_time']:
            return datetime.fromtimestamp(row['latest_time'] / 1000)
        
        return None
