#!/usr/bin/env python3
"""
Real-time Day Trading Strategy Test Script

This script runs the day trading strategy in real-time using Binance testnet
and displays results on the web dashboard.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import structlog

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.config import config
from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.day_trading_strategy import DayTradingStrategy
from src.database.connection import db
from src.web.dashboard import websocket_manager

logger = structlog.get_logger()

class RealTimeDayTradingTester:
    """Real-time day trading strategy tester"""
    
    def __init__(self):
        self.exchange = None
        self.strategy = None
        self.running = False
        self.symbols = ["BTCUSDT", "ETHUSDT"]
        self.test_duration_minutes = 60  # Run for 1 hour by default
        
    async def initialize(self):
        """Initialize exchange and strategy"""
        try:
            # Initialize exchange
            self.exchange = BinanceExchange()
            await self.exchange.connect()
            logger.info("Exchange connected for real-time testing")
            
            # Initialize database
            await db.initialize()
            logger.info("Database initialized")
            
            # Initialize day trading strategy with testnet config
            strategy_config = config['strategies']['day_trading_strategy'].copy()
            strategy_config['symbols'] = self.symbols
            
            self.strategy = DayTradingStrategy(strategy_config)
            
            if not self.strategy.validate_parameters():
                raise ValueError("Strategy parameters validation failed")
            
            logger.info("Day trading strategy initialized for real-time testing",
                       symbols=self.symbols,
                       leverage_enabled=self.strategy.use_leverage,
                       leverage_ratio=self.strategy.leverage if self.strategy.use_leverage else 1.0)
            
            return True
            
        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False
    
    async def fetch_market_data(self) -> dict:
        """Fetch real-time market data for all symbols"""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get recent klines (5-minute intervals for day trading)
                klines = await self.exchange.get_klines(symbol, '5m', limit=100)
                
                if klines:
                    market_data[symbol] = klines
                    logger.debug("Fetched market data", 
                               symbol=symbol, 
                               data_points=len(klines),
                               latest_price=klines[-1]['close'])
                
            except Exception as e:
                logger.error("Failed to fetch market data", symbol=symbol, error=str(e))
        
        return market_data
    
    async def save_signal_to_db(self, signal):
        """Save trading signal to database"""
        try:
            query = """
            INSERT INTO signals (strategy_name, symbol, action, price, confidence, signal_time, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            await db.execute(query, 
                           'day_trading_strategy',
                           signal.symbol,
                           signal.action,
                           float(signal.price),
                           signal.confidence,
                           datetime.now(),
                           signal.metadata)
            
            logger.info("Signal saved to database", 
                       symbol=signal.symbol, 
                       action=signal.action,
                       price=signal.price,
                       confidence=signal.confidence)
        except Exception as e:
            logger.error("Failed to save signal to database", error=str(e))
    
    async def broadcast_update(self, data):
        """Broadcast real-time update to web dashboard"""
        try:
            update_message = {
                'type': 'real_time_update',
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            await websocket_manager.broadcast(update_message)
            logger.debug("Broadcasted update to dashboard clients")
        except Exception as e:
            logger.error("Failed to broadcast update", error=str(e))
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Fetch current market data
            market_data = await self.fetch_market_data()
            
            if not market_data:
                logger.warning("No market data available, skipping cycle")
                return
            
            # Generate signals using the day trading strategy
            signals = await self.strategy.generate_signals(market_data)
            
            cycle_data = {
                'timestamp': datetime.now(),
                'market_data_symbols': list(market_data.keys()),
                'signals_generated': len(signals),
                'signals': []
            }
            
            # Process any generated signals
            for signal in signals:
                logger.info("Trading signal generated",
                           symbol=signal.symbol,
                           action=signal.action,
                           price=signal.price,
                           confidence=signal.confidence,
                           stop_loss=signal.stop_loss,
                           take_profit=signal.take_profit,
                           leverage=signal.metadata.get('leverage', 1.0))
                
                # Save signal to database
                await self.save_signal_to_db(signal)
                
                # Add to cycle data
                cycle_data['signals'].append({
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'price': float(signal.price),
                    'confidence': signal.confidence,
                    'stop_loss': float(signal.stop_loss),
                    'take_profit': float(signal.take_profit),
                    'leverage': signal.metadata.get('leverage', 1.0),
                    'metadata': signal.metadata
                })
                
                # In a real implementation, you would place orders here
                # For testing, we just log the signals
                logger.info("📊 TESTNET SIGNAL (Not placing real order)",
                           symbol=signal.symbol,
                           action=signal.action,
                           price=signal.price,
                           quantity=signal.quantity,
                           leverage=signal.metadata.get('leverage', 1.0))
            
            # Broadcast update to dashboard
            await self.broadcast_update(cycle_data)
            
            if not signals:
                logger.info("No trading signals generated in this cycle")
            
        except Exception as e:
            logger.error("Error in trading cycle", error=str(e))
    
    async def run_test(self, duration_minutes: int = None):
        """Run real-time day trading test"""
        if duration_minutes:
            self.test_duration_minutes = duration_minutes
        
        logger.info("🚀 Starting real-time day trading strategy test",
                   duration_minutes=self.test_duration_minutes,
                   symbols=self.symbols,
                   testnet=True)
        
        if not await self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return False
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=self.test_duration_minutes)
        cycle_count = 0
        
        try:
            while self.running and datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"🔄 Running trading cycle #{cycle_count}")
                
                await self.run_trading_cycle()
                
                # Wait 30 seconds between cycles (adjust as needed)
                await asyncio.sleep(30)
            
            logger.info("✅ Real-time day trading test completed",
                       cycles_completed=cycle_count,
                       duration=datetime.now() - start_time)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            return True
        except Exception as e:
            logger.error("Test failed", error=str(e))
            return False
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.exchange:
            await self.exchange.disconnect()
            logger.info("Exchange disconnected")
        
        if db:
            await db.close()
            logger.info("Database connection closed")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Day Trading Strategy Tester')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in minutes (default: 60)')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols to test (default: BTCUSDT ETHUSDT)')
    
    args = parser.parse_args()
    
    # Setup logging
    from src.logging.logger_config import setup_logging
    setup_logging(log_level='INFO')
    
    tester = RealTimeDayTradingTester()
    tester.symbols = args.symbols
    
    print("=" * 60)
    print("🤖 MAGICBOT REAL-TIME DAY TRADING TEST")
    print("=" * 60)
    print(f"📊 Symbols: {', '.join(args.symbols)}")
    print(f"⏱️  Duration: {args.duration} minutes")
    print(f"🌐 Exchange: Binance Testnet")
    print(f"📈 Strategy: Day Trading (Multi-Indicator)")
    print(f"🎯 Dashboard: http://localhost:8000")
    print("=" * 60)
    print("Press Ctrl+C to stop the test early")
    print("=" * 60)
    
    success = await tester.run_test(args.duration)
    
    if success:
        print("\n✅ Test completed successfully!")
        print("💻 Check the web dashboard at http://localhost:8000 for results")
    else:
        print("\n❌ Test failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())