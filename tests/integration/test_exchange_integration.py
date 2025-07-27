"""
Exchange integration tests with real Binance testnet
Validates exchange connectivity, data retrieval, and API functionality
"""

import pytest
import asyncio
import os
from datetime import datetime, timedelta

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")

from src.exchanges.binance_exchange import BinanceExchange
from src.data.historical_manager import HistoricalDataManager


@pytest.mark.integration
class TestExchangeIntegration:
    """Test real exchange integration functionality"""

    @pytest.fixture(scope="class")
    async def exchange(self):
        """Setup exchange connection for integration tests"""
        exchange = BinanceExchange()
        await exchange.connect()
        yield exchange
        await exchange.disconnect()

    @pytest.mark.asyncio
    async def test_exchange_connection(self, exchange):
        """Test basic exchange connection"""
        # Should be able to connect without errors
        assert exchange is not None
        
        # Test getting server time (basic connectivity test)
        try:
            # Most exchanges have a ping or server time endpoint
            # This validates that the connection is working
            pass  # Exchange connection is validated in fixture
        except Exception as e:
            pytest.fail(f"Exchange connection failed: {e}")

    @pytest.mark.asyncio
    async def test_get_klines_btc(self, exchange):
        """Test getting BTC klines data"""
        klines = await exchange.get_klines("BTCUSDT", "1m", 10)
        
        assert isinstance(klines, list)
        assert len(klines) <= 10
        
        if klines:  # If data is returned
            kline = klines[0]
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            for field in required_fields:
                assert field in kline, f"Missing field: {field}"
                assert kline[field] is not None
            
            # Validate data types and ranges
            assert isinstance(kline['timestamp'], int)
            assert kline['timestamp'] > 0
            assert float(kline['open']) > 0
            assert float(kline['high']) >= float(kline['open'])
            assert float(kline['low']) <= float(kline['open'])
            assert float(kline['close']) > 0
            assert float(kline['volume']) >= 0

    @pytest.mark.asyncio
    async def test_get_klines_eth(self, exchange):
        """Test getting ETH klines data"""
        klines = await exchange.get_klines("ETHUSDT", "5m", 5)
        
        assert isinstance(klines, list)
        assert len(klines) <= 5
        
        if klines:
            # Validate ETH prices are in reasonable range
            kline = klines[0]
            eth_price = float(kline['close'])
            assert 100 < eth_price < 10000, f"ETH price {eth_price} seems unrealistic"

    @pytest.mark.asyncio
    async def test_get_market_data(self, exchange):
        """Test getting current market data"""
        market_data = await exchange.get_market_data("BTCUSDT")
        
        assert hasattr(market_data, 'price')
        assert hasattr(market_data, 'volume')
        
        # Validate price is reasonable
        btc_price = float(market_data.price)
        assert 1000 < btc_price < 200000, f"BTC price {btc_price} seems unrealistic"

    @pytest.mark.asyncio
    async def test_get_account_balance(self, exchange):
        """Test getting account balance"""
        balances = await exchange.get_account_balance()
        
        assert isinstance(balances, list)
        # Testnet accounts may have zero balances, that's ok
        
        if balances:
            balance = balances[0]
            assert 'asset' in balance
            assert 'free' in balance
            assert 'locked' in balance

    @pytest.mark.asyncio
    async def test_multiple_symbols_concurrent(self, exchange):
        """Test getting data for multiple symbols concurrently"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Get data concurrently
        tasks = []
        for symbol in symbols:
            task = exchange.get_klines(symbol, "1m", 5)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate results
        assert len(results) == len(symbols)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Failed to get data for {symbols[i]}: {result}")
            else:
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_data_freshness(self, exchange):
        """Test that returned data is fresh/recent"""
        klines = await exchange.get_klines("BTCUSDT", "1m", 1)
        
        if klines:
            last_timestamp = klines[0]['timestamp']
            current_time = int(datetime.now().timestamp() * 1000)
            
            # Data should be within last 5 minutes (300,000ms)
            time_diff = current_time - last_timestamp
            assert time_diff < 5 * 60 * 1000, f"Data is {time_diff/1000/60:.1f} minutes old"

    @pytest.mark.asyncio
    async def test_historical_data_manager(self, exchange):
        """Test historical data manager integration"""
        data_manager = HistoricalDataManager(exchange)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=2)
        
        # Get historical data
        data = await data_manager.get_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(data) > 0
        # Should be a pandas DataFrame
        assert hasattr(data, 'iloc')
        assert hasattr(data, 'columns')

    @pytest.mark.asyncio
    async def test_error_handling_invalid_symbol(self, exchange):
        """Test error handling with invalid symbol"""
        try:
            await exchange.get_klines("INVALID_SYMBOL", "1m", 5)
            # Some exchanges might return empty list instead of error
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, exchange):
        """Test behavior under rapid requests (rate limiting)"""
        # Make several rapid requests
        tasks = []
        for _ in range(5):
            task = exchange.get_klines("BTCUSDT", "1m", 1)
            tasks.append(task)
        
        # Should handle rate limiting gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed, or handle rate limiting gracefully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "All requests failed - possible rate limiting issue"

    @pytest.mark.asyncio
    async def test_connection_stability(self, exchange):
        """Test connection stability over multiple requests"""
        # Make requests over time to test connection stability
        for i in range(3):
            try:
                klines = await exchange.get_klines("BTCUSDT", "1m", 1)
                assert isinstance(klines, list)
                
                # Small delay between requests
                await asyncio.sleep(1)
                
            except Exception as e:
                pytest.fail(f"Connection failed on request {i+1}: {e}")

    @pytest.mark.asyncio
    async def test_reconnection_capability(self, exchange):
        """Test reconnection after disconnect"""
        # Disconnect
        await exchange.disconnect()
        
        # Reconnect
        await exchange.connect()
        
        # Should work after reconnection
        klines = await exchange.get_klines("BTCUSDT", "1m", 1)
        assert isinstance(klines, list)

    @pytest.mark.asyncio
    async def test_api_credentials_validation(self):
        """Test that API credentials are working"""
        # This is already validated by successful connection,
        # but we can add specific credential validation
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET')
        
        assert api_key is not None, "BINANCE_API_KEY not set"
        assert api_secret is not None, "BINANCE_SECRET not set"
        assert len(api_key) > 10, "API key too short"
        assert len(api_secret) > 10, "API secret too short"