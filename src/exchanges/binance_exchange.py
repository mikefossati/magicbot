import asyncio
import ssl
import os
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Optional
from decimal import Decimal
import structlog

from .base import ExchangeInterface, Order, Balance, MarketData
from ..core.exceptions import ExchangeError
from ..core.config import config

logger = structlog.get_logger()

class BinanceExchange(ExchangeInterface):
    """Binance exchange implementation compatible with standard python-binance"""
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.testnet = config['exchange']['binance']['testnet']
        self.api_key = config['exchange']['binance']['api_key']
        self.secret_key = config['exchange']['binance']['secret_key']
        self._connected = False
        
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        if self._connected:
            return True
            
        try:
            # Set environment variable to disable SSL verification (development only)
            original_verify = os.environ.get('PYTHONHTTPSVERIFY')
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            
            try:
                # Create Binance client with basic parameters
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.secret_key,
                    testnet=self.testnet
                )
                
                # Test connection
                await self.client.ping()
                
                self._connected = True
                
                logger.info("Connected to Binance", 
                           testnet=self.testnet, 
                           ssl_disabled=True)
                return True
                
            finally:
                # Restore original SSL verification setting
                if original_verify is not None:
                    os.environ['PYTHONHTTPSVERIFY'] = original_verify
                elif 'PYTHONHTTPSVERIFY' in os.environ:
                    del os.environ['PYTHONHTTPSVERIFY']
            
        except Exception as e:
            logger.error("Failed to connect to Binance", error=str(e))
            self._connected = False
            if self.client:
                try:
                    await self.client.close_connection()
                except:
                    pass
                self.client = None
            raise ExchangeError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to Binance properly"""
        if not self._connected:
            return
            
        try:
            if self.client:
                await self.client.close_connection()
                self.client = None
            
            self._connected = False
            logger.info("Disconnected from Binance")
        except Exception as e:
            logger.warning("Error during disconnect", error=str(e))
    
    async def get_account_balance(self) -> List[Balance]:
        """Get account balance for all assets"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            account_info = await self.client.get_account()
            balances = []
            
            for balance in account_info['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    balances.append(Balance(
                        asset=balance['asset'],
                        free=Decimal(balance['free']),
                        locked=Decimal(balance['locked']),
                        total=Decimal(balance['free']) + Decimal(balance['locked'])
                    ))
            
            return balances
            
        except BinanceAPIException as e:
            logger.error("Failed to get account balance", error=str(e))
            raise ExchangeError(f"Balance retrieval failed: {e}")
    
    async def place_order(self, order: Order) -> str:
        """Place a trading order"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            order_params = {
                'symbol': order.symbol,
                'side': order.side,
                'type': order.order_type,
                'quantity': str(order.quantity),
            }
            
            if order.order_type == 'LIMIT' and order.price:
                order_params['price'] = str(order.price)
                order_params['timeInForce'] = 'GTC'
            
            result = await self.client.create_order(**order_params)
            
            logger.info("Order placed successfully", 
                       order_id=result['orderId'],
                       symbol=order.symbol,
                       side=order.side,
                       quantity=order.quantity)
            
            return str(result['orderId'])
            
        except BinanceAPIException as e:
            logger.error("Failed to place order", error=str(e))
            raise ExchangeError(f"Order placement failed: {e}")
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            await self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info("Order cancelled", order_id=order_id, symbol=symbol)
            return True
            
        except BinanceAPIException as e:
            logger.error("Failed to cancel order", error=str(e))
            return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            return await self.client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            logger.error("Failed to get order status", error=str(e))
            raise ExchangeError(f"Order status retrieval failed: {e}")
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return MarketData(
                symbol=symbol,
                price=Decimal(ticker['price']),
                volume=Decimal('0'),  # Not available in ticker
                timestamp=int(ticker.get('timestamp', 0))
            )
        except BinanceAPIException as e:
            logger.error("Failed to get market data", error=str(e))
            raise ExchangeError(f"Market data retrieval failed: {e}")
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get historical kline/candlestick data"""
        if not self.client or not self._connected:
            raise ExchangeError("Not connected to exchange")
            
        try:
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to more readable format
            formatted_klines = []
            for kline in klines:
                formatted_klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            return formatted_klines
            
        except BinanceAPIException as e:
            logger.error("Failed to get klines", error=str(e))
            raise ExchangeError(f"Klines retrieval failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        return False  # Don't suppress exceptions