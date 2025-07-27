"""
Integration tests for trading strategies with real Binance testnet
Tests end-to-end workflow with real market data and API calls
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.day_trading_strategy import DayTradingStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.data.historical_manager import HistoricalDataManager


@pytest.mark.integration
class TestStrategyIntegration:
    """Integration tests using real Binance testnet"""

    @pytest.fixture(scope="class")
    async def exchange(self):
        """Setup exchange connection for integration tests"""
        exchange = BinanceExchange()
        await exchange.connect()
        yield exchange
        await exchange.disconnect()

    @pytest.fixture(scope="class")
    async def data_manager(self, exchange):
        """Setup data manager for historical data"""
        return HistoricalDataManager(exchange)

    @pytest.mark.asyncio
    async def test_ma_crossover_real_data(self, exchange):
        """Test MA crossover with real market data"""
        # Configure strategy
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,  # Shorter periods for faster testing
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get real market data
        klines = await exchange.get_klines("BTCUSDT", "1m", 20)
        assert len(klines) >= 10, "Insufficient market data from exchange"
        
        # Test signal generation
        market_data = {"BTCUSDT": klines}
        signals = await strategy.generate_signals(market_data)
        
        # Validate results
        assert isinstance(signals, list)
        for signal in signals:
            assert signal.symbol == 'BTCUSDT'
            assert signal.action in ['BUY', 'SELL']
            assert 0 <= signal.confidence <= 1
            assert signal.price > 0

    @pytest.mark.asyncio
    async def test_day_trading_real_data(self, exchange):
        """Test day trading strategy with real data"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 5,
            'medium_ema': 10,
            'slow_ema': 20,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'max_daily_trades': 3,
            'position_size': 0.001
        }
        
        strategy = DayTradingStrategy(config)
        
        # Get more data for day trading (needs multiple indicators)
        klines = await exchange.get_klines("BTCUSDT", "5m", 50)
        assert len(klines) >= 30, "Insufficient market data for day trading"
        
        market_data = {"BTCUSDT": klines}
        signals = await strategy.generate_signals(market_data)
        
        # Validate results
        assert isinstance(signals, list)
        for signal in signals:
            assert signal.symbol == 'BTCUSDT'
            assert signal.action in ['BUY', 'SELL']
            assert signal.stop_loss is not None
            assert signal.take_profit is not None

    @pytest.mark.asyncio
    async def test_multiple_strategies_concurrent(self, exchange):
        """Test multiple strategies running concurrently"""
        # Configure different strategies
        ma_config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        macd_config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 8,
            'slow_period': 17,
            'signal_period': 9,
            'position_size': 0.001
        }
        
        ma_strategy = MovingAverageCrossover(ma_config)
        macd_strategy = MACDStrategy(macd_config)
        
        # Get market data
        klines = await exchange.get_klines("BTCUSDT", "1m", 30)
        market_data = {"BTCUSDT": klines}
        
        # Run strategies concurrently
        ma_task = ma_strategy.generate_signals(market_data)
        macd_task = macd_strategy.generate_signals(market_data)
        
        ma_signals, macd_signals = await asyncio.gather(ma_task, macd_task)
        
        # Both should complete successfully
        assert isinstance(ma_signals, list)
        assert isinstance(macd_signals, list)

    @pytest.mark.asyncio
    async def test_historical_data_integration(self, exchange, data_manager):
        """Test integration with historical data manager"""
        strategy_config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 10,
            'slow_period': 20,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(strategy_config)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=6)
        
        historical_data = await data_manager.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(historical_data) > 0, "No historical data retrieved"
        
        # Convert to expected format
        market_data = {"BTCUSDT": historical_data.to_dict('records')}
        
        # Generate signals
        signals = await strategy.generate_signals(market_data)
        
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_low_latency_performance(self, exchange):
        """Test low latency performance requirements"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Pre-fetch data to exclude network latency
        klines = await exchange.get_klines("BTCUSDT", "1m", 20)
        market_data = {"BTCUSDT": klines}
        
        # Measure signal generation latency
        latencies = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            signals = await strategy.generate_signals(market_data)
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        # Performance requirements
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Assert low latency requirements
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms limit"
        assert max_latency < 200, f"Max latency {max_latency:.2f}ms exceeds 200ms limit"
        
        print(f"Performance metrics:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        print(f"  Min latency: {min(latencies):.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self, exchange):
        """Test concurrent processing of multiple symbols"""
        config = {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get data for multiple symbols concurrently
        btc_task = exchange.get_klines("BTCUSDT", "1m", 20)
        eth_task = exchange.get_klines("ETHUSDT", "1m", 20)
        
        btc_data, eth_data = await asyncio.gather(btc_task, eth_task)
        
        market_data = {
            "BTCUSDT": btc_data,
            "ETHUSDT": eth_data
        }
        
        # Measure concurrent processing time
        start_time = time.perf_counter()
        
        signals = await strategy.generate_signals(market_data)
        
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        
        # Should process multiple symbols efficiently
        assert processing_time < 500, f"Multi-symbol processing {processing_time:.2f}ms too slow"
        
        # Validate results
        symbols_processed = {signal.symbol for signal in signals}
        assert len(symbols_processed) <= 2  # Max 2 symbols

    @pytest.mark.asyncio
    async def test_market_data_validation(self, exchange):
        """Test handling of various market data conditions"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Test with minimal data
        min_klines = await exchange.get_klines("BTCUSDT", "1m", 5)
        market_data = {"BTCUSDT": min_klines}
        
        # Should handle gracefully
        signals = await strategy.generate_signals(market_data)
        assert isinstance(signals, list)
        
        # Test with sufficient data
        full_klines = await exchange.get_klines("BTCUSDT", "1m", 20)
        market_data = {"BTCUSDT": full_klines}
        
        signals = await strategy.generate_signals(market_data)
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_error_recovery(self, exchange):
        """Test error handling and recovery"""
        config = {
            'symbols': ['BTCUSDT', 'INVALID_SYMBOL'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get data for valid symbol only
        btc_data = await exchange.get_klines("BTCUSDT", "1m", 20)
        market_data = {
            "BTCUSDT": btc_data,
            "INVALID_SYMBOL": []  # Empty data
        }
        
        # Should handle invalid/missing data gracefully
        signals = await strategy.generate_signals(market_data)
        
        # Should still process valid symbols
        assert isinstance(signals, list)
        valid_symbols = {signal.symbol for signal in signals}
        assert 'INVALID_SYMBOL' not in valid_symbols

    @pytest.mark.asyncio
    async def test_real_time_data_freshness(self, exchange):
        """Test data freshness and real-time processing"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get current market data
        klines = await exchange.get_klines("BTCUSDT", "1m", 20)
        
        # Check data freshness (last candle should be recent)
        if klines:
            last_timestamp = klines[-1]['timestamp']
            current_time = int(datetime.now().timestamp() * 1000)
            
            # Data should be within last 5 minutes
            time_diff = current_time - last_timestamp
            assert time_diff < 5 * 60 * 1000, "Market data is not fresh enough"
        
        market_data = {"BTCUSDT": klines}
        signals = await strategy.generate_signals(market_data)
        
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_strategy_consistency(self, exchange):
        """Test strategy consistency across multiple runs"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get market data once
        klines = await exchange.get_klines("BTCUSDT", "1m", 20)
        market_data = {"BTCUSDT": klines}
        
        # Run strategy multiple times with same data
        results = []
        for _ in range(3):
            signals = await strategy.generate_signals(market_data)
            results.append(signals)
        
        # Results should be consistent (same input = same output)
        # Note: This assumes deterministic strategy behavior
        if results[0]:  # If signals were generated
            for i in range(1, len(results)):
                assert len(results[i]) == len(results[0])
                if results[i]:
                    assert results[i][0].action == results[0][0].action

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, exchange):
        """Test memory usage doesn't grow over time"""
        import psutil
        import os
        
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        # Run strategy many times
        for _ in range(50):
            klines = await exchange.get_klines("BTCUSDT", "1m", 20)
            market_data = {"BTCUSDT": klines}
            signals = await strategy.generate_signals(market_data)
        
        # Check memory after multiple runs
        final_memory = process.memory_info().rss
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be minimal (< 10MB)
        assert memory_growth < 10 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"

    @pytest.mark.asyncio 
    async def test_exchange_connection_stability(self, exchange):
        """Test exchange connection stability during extended use"""
        config = {
            'symbols': ['BTCUSDT'],
            'fast_period': 5,
            'slow_period': 10,
            'position_size': 0.001
        }
        
        strategy = MovingAverageCrossover(config)
        
        # Test multiple data requests over time
        for i in range(10):
            try:
                klines = await exchange.get_klines("BTCUSDT", "1m", 20)
                market_data = {"BTCUSDT": klines}
                signals = await strategy.generate_signals(market_data)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                pytest.fail(f"Exchange connection failed on iteration {i}: {e}")
        
        # Connection should remain stable throughout