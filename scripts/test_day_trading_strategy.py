#!/usr/bin/env python3
"""
Test Day Trading Strategy
Basic validation and backtesting of the day trading strategy
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager
from src.backtesting.engine import BacktestEngine, BacktestConfig
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_day_trading_strategy():
    """Test the day trading strategy"""
    
    logger.info("Testing Day Trading Strategy")
    
    # Initialize exchange
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Strategy configuration
        strategy_config = {
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
            'position_size': 0.02
        }
        
        # Create strategy
        logger.info("Creating day trading strategy...")
        strategy = create_strategy('day_trading_strategy', strategy_config)
        
        # Test strategy info
        info = strategy.get_strategy_info()
        print("\n" + "=" * 60)
        print("DAY TRADING STRATEGY INFO")
        print("=" * 60)
        for key, value in info.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")
        
        # Get risk parameters
        risk_params = strategy.get_risk_parameters()
        print("\n" + "RISK PARAMETERS:")
        print("-" * 30)
        for key, value in risk_params.items():
            print(f"{key}: {value}")
        
        # Fetch test data (5 days of 15m data for day trading)
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        logger.info("Fetching historical data for backtesting...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',  # 15-minute data for day trading
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in historical_data or len(historical_data['BTCUSDT']) < 100:
            logger.error("Insufficient data for testing")
            return
        
        logger.info(f"Data loaded: {len(historical_data['BTCUSDT'])} records")
        
        # Test signal generation on recent data
        logger.info("Testing signal generation...")
        recent_data = historical_data['BTCUSDT'].tail(100)
        signal = await strategy.analyze_market_data('BTCUSDT', recent_data)
        
        if signal:
            print("\n" + "🎯 SIGNAL GENERATED:")
            print("-" * 40)
            print(f"Action: {signal.action}")
            print(f"Price: ${signal.price:.2f}")
            print(f"Confidence: {signal.confidence:.2%}")
            print(f"Stop Loss: ${signal.stop_loss:.2f}")
            print(f"Take Profit: ${signal.take_profit:.2f}")
            print(f"Metadata: {signal.metadata}")
        else:
            print("\n⚪ No signal generated with current data")
        
        # Run backtest
        logger.info("Running backtest...")
        backtest_config = BacktestConfig(
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            position_sizing='percentage',
            position_size=0.02
        )
        
        engine = BacktestEngine(backtest_config)
        backtest_results = await engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("DAY TRADING STRATEGY BACKTEST RESULTS")
        print("=" * 80)
        
        # Performance metrics
        capital = backtest_results['capital']
        trades = backtest_results['trades']
        risk_metrics = backtest_results['risk_metrics']
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"Initial Capital: ${capital['initial']:.2f}")
        print(f"Final Capital: ${capital['final']:.2f}")
        print(f"Total Return: {capital['total_return_pct']:.2f}%")
        print(f"Total Trades: {trades['total']}")
        
        if trades['total'] > 0:
            print(f"Winning Trades: {trades.get('winners', 0)}")
            print(f"Losing Trades: {trades.get('losers', 0)}")
            print(f"Win Rate: {trades.get('win_rate_pct', 0):.1f}%")
            print(f"Profit Factor: {trades.get('profit_factor', 0):.2f}")
            print(f"Avg Win: {trades.get('avg_win_pct', 0):.2f}%")
            print(f"Avg Loss: {trades.get('avg_loss_pct', 0):.2f}%")
        else:
            print("No trades executed during backtest period")
        
        print(f"\n⚖️ RISK METRICS:")
        print("-" * 30)
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
        print(f"Volatility: {risk_metrics['volatility_pct']:.2f}%")
        
        # Trading frequency analysis
        if trades['total'] > 0:
            trading_days = (end_date - start_date).days
            avg_trades_per_day = trades['total'] / trading_days
            print(f"\n📈 TRADING FREQUENCY:")
            print("-" * 35)
            print(f"Trading Period: {trading_days} days")
            print(f"Average Trades/Day: {avg_trades_per_day:.1f}")
            print(f"Max Daily Trades Limit: {strategy_config['max_daily_trades']}")
        
        # Performance evaluation
        print(f"\n🎯 STRATEGY EVALUATION:")
        print("-" * 40)
        
        if capital['total_return_pct'] > 0:
            print("✅ Profitable strategy")
        else:
            print("❌ Losing strategy")
            
        if trades['win_rate_pct'] > 50:
            print("✅ Good win rate (>50%)")
        else:
            print("⚠️ Low win rate (<50%)")
            
        if risk_metrics['sharpe_ratio'] > 1.0:
            print("✅ Good risk-adjusted returns (Sharpe > 1.0)")
        else:
            print("⚠️ Poor risk-adjusted returns (Sharpe < 1.0)")
            
        if risk_metrics['max_drawdown_pct'] < 15:
            print("✅ Acceptable drawdown (<15%)")
        else:
            print("⚠️ High drawdown (>15%)")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        logger.error("Error testing day trading strategy", error=str(e))
        raise
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(test_day_trading_strategy())