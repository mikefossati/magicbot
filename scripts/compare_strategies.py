#!/usr/bin/env python3
"""
Strategy Comparison Tool
Compare performance of multiple strategies side-by-side
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
import pandas as pd
from typing import Dict, List, Any

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy, get_available_strategies
from src.data.historical_manager import HistoricalDataManager
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.performance import PerformanceAnalyzer
from src.core.config import load_config
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level for cleaner output
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def run_strategy_comparison(
    strategy_names: List[str],
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 10000,
    commission_rate: float = 0.001,
    interval: str = '1h'
):
    """
    Compare multiple strategies on the same data
    
    Args:
        strategy_names: List of strategy names to compare
        symbols: List of trading pairs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital
        commission_rate: Commission rate
        interval: Timeframe
    """
    
    # Set defaults
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"\n🔍 COMPARING {len(strategy_names)} STRATEGIES")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Capital: ${initial_capital:,}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⏱️  Timeframe: {interval}")
    print("=" * 80)
    
    try:
        # 1. Initialize exchange and fetch data
        exchange = BinanceExchange()
        await exchange.connect()
        
        data_manager = HistoricalDataManager(exchange)
        print("📈 Fetching historical data...")
        
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=symbols,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt
        )
        
        total_records = sum(len(df) for df in historical_data.values())
        if total_records == 0:
            print("❌ No historical data retrieved")
            return None
        
        print(f"✅ Retrieved {total_records:,} data points")
        
        # 2. Load configuration
        config = load_config()
        
        # 3. Backtest each strategy
        results = {}
        
        for i, strategy_name in enumerate(strategy_names, 1):
            print(f"\n⚡ Testing Strategy {i}/{len(strategy_names)}: {strategy_name}")
            
            try:
                # Get strategy config
                if strategy_name in config['strategies']:
                    strategy_config = config['strategies'][strategy_name].copy()
                    strategy_config['symbols'] = symbols
                else:
                    # Default config for unlisted strategies
                    strategy_config = {
                        'symbols': symbols,
                        'position_size': 0.05  # 5% per trade
                    }
                
                # Create strategy
                strategy = create_strategy(strategy_name, strategy_config)
                
                # Configure backtest
                backtest_config = BacktestConfig(
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    slippage_rate=0.0005,
                    position_sizing='percentage',
                    position_size=0.05  # 5% per trade
                )
                
                # Run backtest
                engine = BacktestEngine(backtest_config)
                engine._strategy_name = strategy_name
                
                result = await engine.run_backtest(
                    strategy=strategy,
                    historical_data=historical_data,
                    start_date=start_dt,
                    end_date=end_dt
                )
                
                results[strategy_name] = result
                
                # Quick summary
                total_return = result['capital']['total_return_pct']
                total_trades = result['trades']['total']
                win_rate = result['trades']['win_rate_pct']
                
                print(f"   Return: {total_return:+.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        # 4. Generate comparison report
        if results:
            print(generate_comparison_report(results))
            
            # 5. Save results
            save_comparison_results(results, strategy_names, symbols, start_date, end_date)
        
        await exchange.disconnect()
        return results
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        raise

def generate_comparison_report(results: Dict[str, Any]) -> str:
    """Generate a formatted comparison report"""
    
    if not results:
        return "\n❌ No results to display"
    
    report = ["\n" + "=" * 100]
    report.append("📊 STRATEGY COMPARISON RESULTS")
    report.append("=" * 100)
    
    # Performance summary table
    report.append(f"\n{'Strategy':<20} {'Return %':<10} {'Trades':<8} {'Win %':<8} {'Sharpe':<8} {'Max DD %':<10} {'Final $':<12}")
    report.append("-" * 100)
    
    # Sort by total return
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['capital']['total_return_pct'], 
                          reverse=True)
    
    for strategy_name, result in sorted_results:
        capital = result['capital']
        trades = result['trades']
        risk = result['risk_metrics']
        
        report.append(
            f"{strategy_name:<20} "
            f"{capital['total_return_pct']:>+9.2f} "
            f"{trades['total']:>7} "
            f"{trades['win_rate_pct']:>7.1f} "
            f"{risk['sharpe_ratio']:>7.2f} "
            f"{risk['max_drawdown_pct']:>9.2f} "
            f"${capital['final']:>11,.0f}"
        )
    
    # Best performer analysis
    report.append(f"\n🏆 BEST PERFORMERS")
    report.append("-" * 40)
    
    best_return = max(results.items(), key=lambda x: x[1]['capital']['total_return_pct'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['risk_metrics']['sharpe_ratio'])
    best_winrate = max(results.items(), key=lambda x: x[1]['trades']['win_rate_pct'])
    
    report.append(f"💰 Highest Return:   {best_return[0]} ({best_return[1]['capital']['total_return_pct']:+.2f}%)")
    report.append(f"📈 Best Sharpe:      {best_sharpe[0]} ({best_sharpe[1]['risk_metrics']['sharpe_ratio']:.2f})")
    report.append(f"🎯 Best Win Rate:    {best_winrate[0]} ({best_winrate[1]['trades']['win_rate_pct']:.1f}%)")
    
    # Risk analysis
    report.append(f"\n⚠️  RISK ANALYSIS")
    report.append("-" * 40)
    
    lowest_dd = min(results.items(), key=lambda x: abs(x[1]['risk_metrics']['max_drawdown_pct']))
    highest_dd = max(results.items(), key=lambda x: abs(x[1]['risk_metrics']['max_drawdown_pct']))
    
    report.append(f"🛡️  Lowest Drawdown:  {lowest_dd[0]} ({lowest_dd[1]['risk_metrics']['max_drawdown_pct']:+.2f}%)")
    report.append(f"⚡ Highest Drawdown: {highest_dd[0]} ({highest_dd[1]['risk_metrics']['max_drawdown_pct']:+.2f}%)")
    
    # Strategy recommendations
    report.append(f"\n💡 RECOMMENDATIONS")
    report.append("-" * 40)
    
    if len(results) >= 2:
        # Find complementary strategies
        trend_strategies = [name for name, res in results.items() if 'ma_crossover' in name.lower() or 'breakout' in name.lower()]
        mean_reversion = [name for name, res in results.items() if 'rsi' in name.lower() or 'bollinger' in name.lower()]
        
        if trend_strategies and mean_reversion:
            report.append("🎯 Consider combining trend-following and mean-reversion strategies")
            report.append(f"   Trend: {', '.join(trend_strategies[:2])}")
            report.append(f"   Mean Reversion: {', '.join(mean_reversion[:2])}")
        
        # Portfolio suggestion
        top_2 = sorted_results[:2]
        if len(top_2) == 2:
            combined_return = (top_2[0][1]['capital']['total_return_pct'] + top_2[1][1]['capital']['total_return_pct']) / 2
            report.append(f"📊 Portfolio of top 2 strategies could yield ~{combined_return:.1f}% return")
    
    report.append("\n" + "=" * 100)
    
    return "\n".join(report)

def save_comparison_results(results: Dict[str, Any], strategy_names: List[str], 
                          symbols: List[str], start_date: str, end_date: str):
    """Save comparison results to files"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('backtest_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed comparison
    comparison_file = results_dir / f"strategy_comparison_{timestamp}.json"
    
    # Convert to JSON-serializable format
    json_results = {
        'comparison_metadata': {
            'strategies': strategy_names,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': timestamp
        },
        'results': {}
    }
    
    for strategy_name, result in results.items():
        json_results['results'][strategy_name] = {
            'strategy_name': result['strategy_name'],
            'backtest_period': {
                'start': result['backtest_period']['start'].isoformat() if result['backtest_period']['start'] else None,
                'end': result['backtest_period']['end'].isoformat() if result['backtest_period']['end'] else None,
                'duration_days': result['backtest_period']['duration_days']
            },
            'capital': result['capital'],
            'trades': result['trades'],
            'risk_metrics': result['risk_metrics']
        }
    
    with open(comparison_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save summary report
    report_file = results_dir / f"comparison_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(generate_comparison_report(results))
    
    print(f"\n💾 Results saved:")
    print(f"   📄 Detailed: {comparison_file}")
    print(f"   📋 Report: {report_file}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Compare Multiple Trading Strategies')
    
    available_strategies = list(get_available_strategies().keys())
    
    parser.add_argument('strategies', nargs='*', 
                       help=f'Strategies to compare (available: {", ".join(available_strategies)})')
    parser.add_argument('--all', action='store_true',
                       help='Compare all available strategies')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols (default: BTCUSDT ETHUSDT)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD, default: 90 days ago)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (default: 0.001)')
    parser.add_argument('--interval', type=str, default='1h',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Timeframe (default: 1h)')
    
    args = parser.parse_args()
    
    # Determine strategies to test
    if args.all:
        strategies_to_test = available_strategies
    elif args.strategies:
        strategies_to_test = args.strategies
        # Validate strategy names
        invalid_strategies = [s for s in strategies_to_test if s not in available_strategies]
        if invalid_strategies:
            print(f"❌ Invalid strategies: {', '.join(invalid_strategies)}")
            print(f"Available strategies: {', '.join(available_strategies)}")
            return
    else:
        print(f"Please specify strategies to compare or use --all")
        print(f"Available strategies: {', '.join(available_strategies)}")
        return
    
    print(f"🚀 Starting comparison of {len(strategies_to_test)} strategies...")
    
    asyncio.run(run_strategy_comparison(
        strategy_names=strategies_to_test,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        commission_rate=args.commission,
        interval=args.interval
    ))

if __name__ == "__main__":
    main()