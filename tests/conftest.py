"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")
    print("Install with: pip install python-dotenv")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.base import Signal


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_exchange():
    """Mock BinanceExchange for testing"""
    exchange = MagicMock(spec=BinanceExchange)
    exchange.connect = AsyncMock()
    exchange.disconnect = AsyncMock()
    exchange.get_klines = AsyncMock()
    exchange.get_market_data = AsyncMock()
    exchange.get_account_balance = AsyncMock()
    return exchange


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    return generate_ohlcv_data(length=100, trend='sideways')


@pytest.fixture
def bull_market_data():
    """Generate bullish market data"""
    return generate_ohlcv_data(length=100, trend='bull', volatility=0.015)


@pytest.fixture
def bear_market_data():
    """Generate bearish market data"""
    return generate_ohlcv_data(length=100, trend='bear', volatility=0.015)


@pytest.fixture
def volatile_market_data():
    """Generate highly volatile market data"""
    return generate_ohlcv_data(length=100, trend='sideways', volatility=0.05)


@pytest.fixture
def ma_crossover_config():
    """Standard MA crossover strategy configuration"""
    return {
        'symbols': ['BTCUSDT'],
        'fast_period': 10,
        'slow_period': 30,
        'position_size': 0.1
    }


@pytest.fixture
def day_trading_config():
    """Day trading strategy configuration"""
    return {
        'symbols': ['BTCUSDT'],
        'fast_ema': 8,
        'medium_ema': 21,
        'slow_ema': 50,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_neutral_high': 60,
        'rsi_neutral_low': 40,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'volume_period': 20,
        'volume_multiplier': 1.5,
        'pivot_period': 10,
        'support_resistance_threshold': 0.2,
        'stop_loss_pct': 1.5,
        'take_profit_pct': 2.5,
        'trailing_stop_pct': 1.0,
        'max_daily_trades': 3,
        'session_start': "09:30",
        'session_end': "15:30",
        'position_size': 0.03,  # OPTIMIZED: Updated from 0.02 to 0.03 for better default
        # OPTIMIZED signal scoring parameters
        'min_signal_score': 0.75,     # Increased from 0.5 for stricter signals
        'strong_signal_score': 0.85   # Increased from 0.8 for higher bar
    }


@pytest.fixture
def macd_config():
    """MACD strategy configuration"""
    return {
        'symbols': ['BTCUSDT'],
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'histogram_threshold': 0.0,
        'position_size': 0.1
    }


@pytest.fixture
def stochastic_config():
    """Stochastic strategy configuration"""
    return {
        'symbols': ['BTCUSDT'],
        'k_period': 14,
        'd_period': 3,
        'smooth_period': 3,
        'oversold': 20,
        'overbought': 80,
        'divergence_lookback': 5,
        'position_size': 0.1
    }


def generate_ohlcv_data(
    length: int = 100,
    trend: str = 'sideways',  # 'bull', 'bear', 'sideways'
    volatility: float = 0.02,
    start_price: float = 50000.0,
    start_time: datetime = None
) -> List[Dict]:
    """
    Generate realistic OHLCV data for testing
    
    Args:
        length: Number of candles to generate
        trend: Market trend direction
        volatility: Price volatility (standard deviation)
        start_price: Starting price
        start_time: Starting timestamp
    
    Returns:
        List of OHLCV dictionaries
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(hours=length)
    
    np.random.seed(42)  # For reproducible test data
    data = []
    current_price = start_price
    
    # Trend parameters
    trend_strength = {
        'bull': 0.001,
        'bear': -0.001,
        'sideways': 0.0
    }
    
    base_trend = trend_strength.get(trend, 0.0)
    
    for i in range(length):
        # Calculate timestamp
        timestamp = int((start_time + timedelta(hours=i)).timestamp() * 1000)
        
        # Generate price movement
        trend_component = base_trend * current_price
        random_component = np.random.normal(0, volatility) * current_price
        price_change = trend_component + random_component
        
        # Calculate OHLC
        open_price = current_price
        close_price = max(open_price + price_change, 0.01)  # Prevent negative prices
        
        # Generate realistic high/low within the range
        range_pct = abs(price_change) / open_price + volatility * 0.5
        high_price = max(open_price, close_price) * (1 + range_pct * np.random.uniform(0, 1))
        low_price = min(open_price, close_price) * (1 - range_pct * np.random.uniform(0, 1))
        
        # Generate volume (realistic range)
        base_volume = 1000
        volume = int(base_volume * (1 + np.random.normal(0, 0.5)))
        volume = max(volume, 100)  # Minimum volume
        
        data.append({
            'timestamp': timestamp,
            'open': float(open_price),
            'high': float(high_price),
            'low': float(low_price),
            'close': float(close_price),
            'volume': float(volume)
        })
        
        current_price = close_price
    
    return data


def generate_crossover_scenario(
    crossover_type: str = 'bullish',  # 'bullish', 'bearish'
    length: int = 60,
    fast_period: int = 10,
    slow_period: int = 30
) -> List[Dict]:
    """
    Generate market data that will trigger a specific crossover scenario
    """
    data = generate_ohlcv_data(length=length, trend='sideways')
    
    # Modify the data to create a crossover scenario
    df = pd.DataFrame(data)
    df['close'] = df['close'].astype(float)
    
    if crossover_type == 'bullish':
        # Create bullish crossover: fast MA crosses above slow MA
        for i in range(length - 10, length):
            df.loc[i, 'close'] = df.loc[i-1, 'close'] * 1.01  # Gradual price increase
    elif crossover_type == 'bearish':
        # Create bearish crossover: fast MA crosses below slow MA
        for i in range(length - 10, length):
            df.loc[i, 'close'] = df.loc[i-1, 'close'] * 0.99  # Gradual price decrease
    
    # Update OHLC to be consistent with close prices
    for i in range(1, length):
        prev_close = df.loc[i-1, 'close']
        current_close = df.loc[i, 'close']
        
        df.loc[i, 'open'] = prev_close
        df.loc[i, 'high'] = max(prev_close, current_close) * 1.001
        df.loc[i, 'low'] = min(prev_close, current_close) * 0.999
    
    return df.to_dict('records')


def generate_day_trading_scenarios() -> Dict[str, List[Dict]]:
    """
    Generate specific day trading scenarios for testing
    """
    scenarios = {}
    
    # Scenario 1: Morning breakout with volume confirmation
    scenarios['morning_breakout'] = generate_morning_breakout_data()
    
    # Scenario 2: Trend reversal at support/resistance
    scenarios['trend_reversal'] = generate_trend_reversal_data()
    
    # Scenario 3: Choppy sideways market (should avoid trades)
    scenarios['choppy_market'] = generate_choppy_market_data()
    
    # Scenario 4: Strong trending day
    scenarios['strong_trend'] = generate_strong_trend_data()
    
    # Scenario 5: High volatility with false breakouts
    scenarios['false_breakouts'] = generate_false_breakout_data()
    
    return scenarios


def generate_morning_breakout_data() -> List[Dict]:
    """Generate data showing a morning breakout pattern"""
    # Start with consolidation, then strong breakout with volume
    data = generate_ohlcv_data(length=100, trend='sideways', volatility=0.01)
    
    # Add breakout in the last 20 candles
    df = pd.DataFrame(data)
    for i in range(80, 100):
        df.loc[i, 'close'] = df.loc[i-1, 'close'] * 1.005  # Strong upward movement
        df.loc[i, 'volume'] = df.loc[i, 'volume'] * 2.0  # Increased volume
        
        # Update OHLC
        df.loc[i, 'open'] = df.loc[i-1, 'close']
        df.loc[i, 'high'] = df.loc[i, 'close'] * 1.002
        df.loc[i, 'low'] = df.loc[i, 'open'] * 0.998
    
    return df.to_dict('records')


def generate_trend_reversal_data() -> List[Dict]:
    """Generate data showing trend reversal at key levels"""
    # Start with downtrend, then reversal
    data = generate_ohlcv_data(length=100, trend='bear', volatility=0.02)
    
    # Add reversal in the last 15 candles
    df = pd.DataFrame(data)
    for i in range(85, 100):
        df.loc[i, 'close'] = df.loc[i-1, 'close'] * 1.01  # Reversal upward
        
        # Update OHLC
        df.loc[i, 'open'] = df.loc[i-1, 'close']
        df.loc[i, 'high'] = df.loc[i, 'close'] * 1.001
        df.loc[i, 'low'] = df.loc[i, 'open'] * 0.999
    
    return df.to_dict('records')


def generate_choppy_market_data() -> List[Dict]:
    """Generate choppy, sideways market data"""
    return generate_ohlcv_data(length=100, trend='sideways', volatility=0.03)


def generate_strong_trend_data() -> List[Dict]:
    """Generate strong trending market data"""
    return generate_ohlcv_data(length=100, trend='bull', volatility=0.015)


def generate_false_breakout_data() -> List[Dict]:
    """Generate data with false breakouts"""
    data = generate_ohlcv_data(length=100, trend='sideways', volatility=0.02)
    
    # Add false breakout pattern
    df = pd.DataFrame(data)
    
    # False breakout up, then down
    for i in range(50, 55):
        df.loc[i, 'close'] = df.loc[i-1, 'close'] * 1.01  # False breakout up
    for i in range(55, 65):
        df.loc[i, 'close'] = df.loc[i-1, 'close'] * 0.995  # Then down
    
    # Update OHLC for consistency
    for i in range(50, 65):
        if i > 50:
            df.loc[i, 'open'] = df.loc[i-1, 'close']
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'close']) * 1.001
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'close']) * 0.999
    
    return df.to_dict('records')


@pytest.fixture
def day_trading_scenarios():
    """Fixture providing day trading test scenarios"""
    return generate_day_trading_scenarios()