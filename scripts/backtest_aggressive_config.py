#!/usr/bin/env python3
"""
Backtest Aggressive Day Trading Configuration
Run comprehensive backtest using optimized aggressive parameters
From 2025-01-01 to 2025-06-01 using testnet data
"""

import asyncio
import sys
import os
import json
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

async def backtest_aggressive_strategy():
    """Run comprehensive backtest with aggressive configuration"""
    
    print("=" * 80)
    print("AGGRESSIVE DAY TRADING STRATEGY BACKTEST")
    print("Period: 2025-01-01 to 2025-06-01 (5 months)")
    print("=" * 80)
    
    # Load aggressive configuration
    try:
        with open('day_trading_aggressive_trading_20250726_140500.json', 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"⚖️ Risk Level: {config['risk_level']}")
    except FileNotFoundError:
        print("❌ Aggressive configuration file not found, using parameters directly")
        config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 5,
            'medium_ema': 13,
            'slow_ema': 34,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_neutral_high': 60,
            'rsi_neutral_low': 40,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_period': 20,
            'volume_multiplier': 1.0,
            'min_volume_ratio': 0.8,
            'pivot_period': 10,
            'support_resistance_threshold': 1.0,
            'min_signal_score': 0.5,
            'strong_signal_score': 0.8,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'trailing_stop_pct': 1.0,
            'max_daily_trades': 5,
            'session_start': '00:00',
            'session_end': '23:59',
            'position_size': 0.02,
            'leverage': 1.0,
            'use_leverage': False
        }
    
    print(f"\n🔧 AGGRESSIVE CONFIGURATION PARAMETERS:")
    print(f"   EMA Periods: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
    print(f"   Volume Multiplier: {config['volume_multiplier']}")
    print(f"   Min Signal Score: {config['min_signal_score']}")
    print(f"   Stop Loss: {config['stop_loss_pct']}%")
    print(f"   Take Profit: {config['take_profit_pct']}%")
    print(f"   Max Daily Trades: {config['max_daily_trades']}")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Create strategy
        logger.info("Creating aggressive day trading strategy...")
        strategy = create_strategy('day_trading_strategy', config)
        
        # Set up date range
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 6, 1)
        
        print(f"\n📅 BACKTEST PERIOD:")
        print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
        print(f"   End: {end_date.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(end_date - start_date).days} days")
        
        # Fetch historical data for multiple timeframes
        data_manager = HistoricalDataManager(exchange)
        
        # Test different intervals for comprehensive analysis
        intervals = ['15m', '1h']
        results_summary = []
        
        for interval in intervals:
            print(f"\n{'='*60}")
            print(f"TESTING INTERVAL: {interval}")
            print(f"{'='*60}")
            
            logger.info(f"Fetching {interval} data from {start_date} to {end_date}...")
            
            try:
                historical_data = await data_manager.get_multiple_symbols_data(
                    symbols=['BTCUSDT'],
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'BTCUSDT' not in historical_data or len(historical_data['BTCUSDT']) < 100:
                    print(f"❌ Insufficient data for {interval} interval")
                    continue
                
                data_points = len(historical_data['BTCUSDT'])
                print(f"📊 Data loaded: {data_points} records")
                
                # Show data range
                first_record = historical_data['BTCUSDT'][0]
                last_record = historical_data['BTCUSDT'][-1]
                first_date = datetime.fromtimestamp(first_record['timestamp'] / 1000)
                last_date = datetime.fromtimestamp(last_record['timestamp'] / 1000)
                
                print(f"📅 Data range: {first_date.strftime('%Y-%m-%d %H:%M')} to {last_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"💰 Price range: ${first_record['close']:.2f} to ${last_record['close']:.2f}")
                
                # Configure backtest
                backtest_config = BacktestConfig(
                    initial_capital=10000.0,
                    commission_rate=0.001,  # 0.1% commission (aggressive trading)
                    slippage_rate=0.0005,   # 0.05% slippage
                    position_sizing='percentage',
                    position_size=0.02
                )
                
                print(f"\n💼 BACKTEST CONFIGURATION:")
                print(f"   Initial Capital: ${backtest_config.initial_capital:,.2f}")
                print(f"   Commission Rate: {backtest_config.commission_rate:.3f}")
                print(f"   Position Size: {backtest_config.position_size:.1%}")
                
                # Run backtest
                print(f"\n🔄 Running backtest for {interval} interval...")
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
                trading_days = (end_date - start_date).days
                annualized_return = (total_return / 100 + 1) ** (365 / trading_days) - 1
                
                # Display detailed results
                print(f"\n📈 PERFORMANCE RESULTS ({interval}):")
                print("-" * 50)
                print(f"   Initial Capital: ${capital['initial']:,.2f}")
                print(f"   Final Capital: ${capital['final']:,.2f}")
                print(f"   Total Return: {total_return:.2f}%")
                print(f"   Annualized Return: {annualized_return*100:.2f}%")
                
                if capital['peak'] > capital['initial']:
                    peak_return = ((capital['peak'] - capital['initial']) / capital['initial']) * 100
                    print(f"   Peak Return: {peak_return:.2f}%")
                
                print(f"\n🔄 TRADING ACTIVITY ({interval}):")
                print("-" * 50)
                print(f"   Total Trades: {trades['total']}")
                
                if trades['total'] > 0:
                    print(f"   Winning Trades: {trades.get('winning', 0)}")
                    print(f"   Losing Trades: {trades.get('losing', 0)}")
                    print(f"   Win Rate: {trades.get('win_rate_pct', 0):.1f}%")
                    print(f"   Profit Factor: {trades.get('profit_factor', 0):.2f}")
                    print(f"   Average Trade: {trades.get('avg_return_pct', 0):.2f}%")
                    print(f"   Best Trade: {trades.get('best_trade_pct', 0):.2f}%")
                    print(f"   Worst Trade: {trades.get('worst_trade_pct', 0):.2f}%")
                    print(f"   Avg Trades/Day: {trades['total']/trading_days:.1f}")
                    
                    # Calculate frequency metrics for aggressive strategy
                    trades_per_week = trades['total'] / (trading_days / 7)
                    print(f"   Trades per Week: {trades_per_week:.1f}")
                else:
                    print(f"   No trades executed")
                
                print(f"\n⚖️ RISK METRICS ({interval}):")
                print("-" * 50)
                print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
                print(f"   Volatility: {risk_metrics['volatility_pct']:.2f}%")
                print(f"   Calmar Ratio: {risk_metrics.get('calmar_ratio', 0):.2f}")
                
                # Performance evaluation specific to aggressive strategy
                print(f"\n🎯 AGGRESSIVE STRATEGY EVALUATION ({interval}):")
                print("-" * 50)
                
                # Profitability assessment
                if total_return > 5:
                    print(f"   ✅ Excellent profitability (+{total_return:.2f}%)")
                elif total_return > 1:
                    print(f"   ✅ Good profitability (+{total_return:.2f}%)")
                elif total_return > 0:
                    print(f"   ⚠️ Marginal profitability (+{total_return:.2f}%)")
                else:
                    print(f"   ❌ Unprofitable ({total_return:.2f}%)")
                
                # Trading frequency assessment (important for aggressive strategy)
                if trades['total'] > 0:
                    trades_per_day = trades['total'] / trading_days
                    if trades_per_day >= 1.0:
                        print(f"   ✅ High frequency achieved ({trades_per_day:.1f} trades/day)")
                    elif trades_per_day >= 0.5:
                        print(f"   ⚠️ Moderate frequency ({trades_per_day:.1f} trades/day)")
                    else:
                        print(f"   ❌ Low frequency ({trades_per_day:.1f} trades/day)")
                    
                    # Win rate assessment
                    win_rate = trades.get('win_rate_pct', 0)
                    if win_rate >= 55:
                        print(f"   ✅ Excellent win rate ({win_rate:.1f}%)")
                    elif win_rate >= 45:
                        print(f"   ✅ Good win rate ({win_rate:.1f}%)")
                    elif win_rate >= 35:
                        print(f"   ⚠️ Acceptable win rate ({win_rate:.1f}%)")
                    else:
                        print(f"   ❌ Poor win rate ({win_rate:.1f}%)")
                
                # Risk assessment
                max_dd = risk_metrics['max_drawdown_pct']
                if max_dd < 8:
                    print(f"   ✅ Low drawdown risk ({max_dd:.2f}%)")
                elif max_dd < 15:
                    print(f"   ⚠️ Moderate drawdown risk ({max_dd:.2f}%)")
                else:
                    print(f"   ❌ High drawdown risk ({max_dd:.2f}%)")
                
                # Sharpe ratio assessment
                sharpe = risk_metrics['sharpe_ratio']
                if sharpe > 1.5:
                    print(f"   ✅ Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
                elif sharpe > 1.0:
                    print(f"   ✅ Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
                elif sharpe > 0.5:
                    print(f"   ⚠️ Moderate risk-adjusted returns (Sharpe: {sharpe:.2f})")
                else:
                    print(f"   ❌ Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
                
                # Store results for comparison
                results_summary.append({
                    'interval': interval,
                    'data_points': data_points,
                    'return_pct': total_return,
                    'annualized_return_pct': annualized_return * 100,
                    'trades': trades['total'],
                    'win_rate': trades.get('win_rate_pct', 0),
                    'profit_factor': trades.get('profit_factor', 0),
                    'sharpe': risk_metrics['sharpe_ratio'],
                    'max_drawdown': risk_metrics['max_drawdown_pct'],
                    'trades_per_day': trades['total']/trading_days if trading_days > 0 else 0,
                    'volatility': risk_metrics['volatility_pct']
                })
                
            except Exception as e:
                print(f"❌ Error testing {interval} interval: {e}")
                logger.error(f"Error in {interval} backtest", error=str(e))
        
        # Comprehensive summary
        if results_summary:
            print(f"\n{'='*80}")
            print("AGGRESSIVE STRATEGY COMPREHENSIVE RESULTS")
            print(f"{'='*80}")
            
            print(f"{'Interval':<8} {'Return%':<8} {'Annual%':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7} {'MaxDD%':<7} {'T/Day':<6} {'Vol%':<6}")
            print("-" * 80)
            
            for result in results_summary:
                print(f"{result['interval']:<8} "
                      f"{result['return_pct']:<8.2f} "
                      f"{result['annualized_return_pct']:<8.1f} "
                      f"{result['trades']:<7} "
                      f"{result['win_rate']:<6.1f} "
                      f"{result['sharpe']:<7.2f} "
                      f"{result['max_drawdown']:<7.2f} "
                      f"{result['trades_per_day']:<6.2f} "
                      f"{result['volatility']:<6.1f}")
            
            # Best performing interval
            if len(results_summary) > 1:
                # Score each interval (return * sharpe / max_drawdown)
                for result in results_summary:
                    dd_factor = max(result['max_drawdown'], 1)  # Avoid division by zero
                    result['score'] = (result['return_pct'] * result['sharpe']) / dd_factor
                
                best_result = max(results_summary, key=lambda x: x['score'])
                
                print(f"\n🏆 BEST PERFORMING INTERVAL: {best_result['interval']}")
                print("-" * 50)
                print(f"   Total Return: {best_result['return_pct']:.2f}%")
                print(f"   Annualized Return: {best_result['annualized_return_pct']:.1f}%")
                print(f"   Total Trades: {best_result['trades']}")
                print(f"   Win Rate: {best_result['win_rate']:.1f}%")
                print(f"   Sharpe Ratio: {best_result['sharpe']:.2f}")
                print(f"   Max Drawdown: {best_result['max_drawdown']:.2f}%")
                print(f"   Trades per Day: {best_result['trades_per_day']:.2f}")
            
            # Overall assessment
            print(f"\n🎖️ AGGRESSIVE STRATEGY ASSESSMENT:")
            print("-" * 50)
            
            avg_return = sum(r['return_pct'] for r in results_summary) / len(results_summary)
            avg_sharpe = sum(r['sharpe'] for r in results_summary) / len(results_summary)
            avg_drawdown = sum(r['max_drawdown'] for r in results_summary) / len(results_summary)
            total_trades = sum(r['trades'] for r in results_summary)
            avg_trades_per_day = sum(r['trades_per_day'] for r in results_summary) / len(results_summary)
            
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Sharpe: {avg_sharpe:.2f}")
            print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
            print(f"Average Trades per Day: {avg_trades_per_day:.2f}")
            print(f"Total Trades Across Tests: {total_trades}")
            
            # Final verdict for aggressive strategy
            print(f"\n🎯 AGGRESSIVE STRATEGY VERDICT:")
            if avg_return > 3 and avg_sharpe > 1.0 and avg_drawdown < 15 and avg_trades_per_day >= 0.5:
                print("✅ EXCELLENT - Aggressive strategy shows strong performance")
                print("   High returns with good risk management and sufficient frequency")
            elif avg_return > 1 and avg_sharpe > 0.5 and avg_trades_per_day >= 0.3:
                print("⚠️ GOOD - Strategy has potential with room for improvement")
                print("   Consider optimization or market condition analysis")
            elif avg_return > 0:
                print("⚠️ MARGINAL - Strategy shows modest profitability")
                print("   May need parameter adjustments or different market conditions")
            else:
                print("❌ UNDERPERFORMING - Strategy needs significant improvements")
                print("   Consider different parameters or market analysis")
            
            # Specific recommendations for aggressive strategy
            print(f"\n📋 RECOMMENDATIONS FOR AGGRESSIVE TRADING:")
            if avg_trades_per_day < 0.5:
                print("   • Increase signal frequency by lowering min_signal_score")
                print("   • Reduce volume_multiplier to capture more opportunities")
            if avg_drawdown > 12:
                print("   • Implement tighter position sizing")
                print("   • Consider reducing max_daily_trades")
            if avg_sharpe < 1.0:
                print("   • Review stop loss and take profit ratios")
                print("   • Analyze signal quality and timing")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"aggressive_backtest_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'configuration': config,
                    'backtest_period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat(),
                        'days': trading_days
                    },
                    'results_by_interval': results_summary,
                    'summary': {
                        'avg_return_pct': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'avg_drawdown_pct': avg_drawdown,
                        'avg_trades_per_day': avg_trades_per_day,
                        'total_trades': total_trades
                    }
                }, f, indent=2, default=str)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        logger.error("Error in aggressive strategy backtest", error=str(e))
        print(f"❌ Backtest failed: {e}")
        raise
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(backtest_aggressive_strategy())