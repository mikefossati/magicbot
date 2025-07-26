#!/usr/bin/env python3
"""
Day Trading Strategy Backtest
Comprehensive backtesting of the day trading strategy with detailed analysis
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
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def backtest_day_trading_strategy():
    """Run comprehensive backtest of day trading strategy"""
    
    print("=" * 80)
    print("DAY TRADING STRATEGY BACKTEST")
    print("=" * 80)
    
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
            'session_start': "00:00",  # Allow all hours for crypto
            'session_end': "23:59",
            'position_size': 0.02
        }
        
        # Create strategy
        logger.info("Creating day trading strategy...")
        strategy = create_strategy('day_trading_strategy', strategy_config)
        
        # Fetch data for multiple timeframes
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        
        # Test different periods
        test_periods = [
            {"days": 7, "name": "1 Week", "interval": "15m"},
            {"days": 14, "name": "2 Weeks", "interval": "15m"},
            {"days": 30, "name": "1 Month", "interval": "1h"}
        ]
        
        results_summary = []
        
        for period in test_periods:
            print(f"\n{'='*60}")
            print(f"TESTING: {period['name']} ({period['days']} days, {period['interval']} interval)")
            print(f"{'='*60}")
            
            start_date = end_date - timedelta(days=period['days'])
            
            logger.info(f"Fetching {period['days']} days of {period['interval']} data...")
            historical_data = await data_manager.get_multiple_symbols_data(
                symbols=['BTCUSDT'],
                interval=period['interval'],
                start_date=start_date,
                end_date=end_date
            )
            
            if 'BTCUSDT' not in historical_data or len(historical_data['BTCUSDT']) < 100:
                print(f"❌ Insufficient data for {period['name']} test")
                continue
            
            data_points = len(historical_data['BTCUSDT'])
            print(f"📊 Data loaded: {data_points} records")
            print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,  # 0.1% commission
                slippage_rate=0.0005,   # 0.05% slippage
                position_sizing='percentage',
                position_size=0.02
            )
            
            # Run backtest
            engine = BacktestEngine(backtest_config)
            backtest_results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Analyze results
            capital = backtest_results['capital']
            trades = backtest_results['trades']
            risk_metrics = backtest_results['risk_metrics']
            
            # Calculate additional metrics
            total_return = capital['total_return_pct']
            trading_days = period['days']
            annualized_return = (total_return / 100 + 1) ** (365 / trading_days) - 1
            
            # Display results
            print(f"\n📈 PERFORMANCE RESULTS:")
            print(f"   Initial Capital: ${capital['initial']:,.2f}")
            print(f"   Final Capital: ${capital['final']:,.2f}")
            print(f"   Total Return: {total_return:.2f}%")
            print(f"   Annualized Return: {annualized_return*100:.2f}%")
            
            print(f"\n🔄 TRADING ACTIVITY:")
            print(f"   Total Trades: {trades['total']}")
            if trades['total'] > 0:
                print(f"   Win Rate: {trades.get('win_rate_pct', 0):.1f}%")
                print(f"   Profit Factor: {trades.get('profit_factor', 0):.2f}")
                print(f"   Avg Trades/Day: {trades['total']/trading_days:.1f}")
            else:
                print(f"   No trades executed")
            
            print(f"\n⚖️ RISK METRICS:")
            print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
            print(f"   Volatility: {risk_metrics['volatility_pct']:.2f}%")
            
            # Performance evaluation
            print(f"\n🎯 EVALUATION:")
            if total_return > 0:
                print(f"   ✅ Profitable (+{total_return:.2f}%)")
            else:
                print(f"   ❌ Losing ({total_return:.2f}%)")
                
            if trades['total'] > 0:
                win_rate = trades.get('win_rate_pct', 0)
                if win_rate >= 50:
                    print(f"   ✅ Good win rate ({win_rate:.1f}%)")
                else:
                    print(f"   ⚠️ Low win rate ({win_rate:.1f}%)")
            
            if risk_metrics['sharpe_ratio'] > 1.0:
                print(f"   ✅ Good risk-adjusted returns")
            elif risk_metrics['sharpe_ratio'] > 0:
                print(f"   ⚠️ Moderate risk-adjusted returns")
            else:
                print(f"   ❌ Poor risk-adjusted returns")
            
            if risk_metrics['max_drawdown_pct'] < 10:
                print(f"   ✅ Low drawdown risk")
            elif risk_metrics['max_drawdown_pct'] < 20:
                print(f"   ⚠️ Moderate drawdown risk")
            else:
                print(f"   ❌ High drawdown risk")
            
            # Store results for comparison
            results_summary.append({
                'period': period['name'],
                'days': period['days'],
                'interval': period['interval'],
                'return_pct': total_return,
                'annualized_return_pct': annualized_return * 100,
                'trades': trades['total'],
                'win_rate': trades.get('win_rate_pct', 0),
                'sharpe': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown_pct'],
                'trades_per_day': trades['total']/trading_days if trading_days > 0 else 0
            })
        
        # Summary comparison
        print(f"\n{'='*80}")
        print("STRATEGY PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        if results_summary:
            print(f"{'Period':<12} {'Return%':<8} {'Annual%':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7} {'MaxDD%':<7} {'T/Day':<6}")
            print("-" * 80)
            
            for result in results_summary:
                print(f"{result['period']:<12} "
                      f"{result['return_pct']:<8.2f} "
                      f"{result['annualized_return_pct']:<8.1f} "
                      f"{result['trades']:<7} "
                      f"{result['win_rate']:<6.1f} "
                      f"{result['sharpe']:<7.2f} "
                      f"{result['max_drawdown']:<7.2f} "
                      f"{result['trades_per_day']:<6.1f}")
        
        # Overall assessment
        print(f"\n🏆 OVERALL STRATEGY ASSESSMENT:")
        print("-" * 50)
        
        if results_summary:
            avg_return = sum(r['return_pct'] for r in results_summary) / len(results_summary)
            avg_sharpe = sum(r['sharpe'] for r in results_summary) / len(results_summary)
            avg_drawdown = sum(r['max_drawdown'] for r in results_summary) / len(results_summary)
            total_trades = sum(r['trades'] for r in results_summary)
            
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Sharpe: {avg_sharpe:.2f}")
            print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
            print(f"Total Trades Across Tests: {total_trades}")
            
            # Final verdict
            print(f"\n🎖️ STRATEGY VERDICT:")
            if avg_return > 2 and avg_sharpe > 0.5 and avg_drawdown < 15:
                print("✅ RECOMMENDED - Strategy shows consistent profitable performance")
            elif avg_return > 0 and avg_sharpe > 0:
                print("⚠️ CONDITIONAL - Strategy has potential but needs optimization")
            else:
                print("❌ NOT RECOMMENDED - Strategy shows poor risk-adjusted returns")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        logger.error("Error in day trading backtest", error=str(e))
        raise
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(backtest_day_trading_strategy())