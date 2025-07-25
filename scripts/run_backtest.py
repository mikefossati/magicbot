#!/usr/bin/env python3
"""
Magicbot Backtest Runner
Comprehensive backtesting script for trading strategies
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.ma_crossover import MovingAverageCrossover
from src.data.historical_manager import HistoricalDataManager
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.performance import PerformanceAnalyzer
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def run_ma_crossover_backtest(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    fast_period: int = 10,
    slow_period: int = 30,
    initial_capital: float = 10000,
    commission_rate: float = 0.001,
    interval: str = '1h'
):
    """
    Run backtest for Moving Average Crossover strategy
    
    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fast_period: Fast moving average period
        slow_period: Slow moving average period
        initial_capital: Starting capital in USD
        commission_rate: Commission rate (0.001 = 0.1%)
        interval: Timeframe ('1h', '4h', '1d')
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
    
    logger.info("Starting backtest",
               symbols=symbols,
               start_date=start_date,
               end_date=end_date,
               fast_period=fast_period,
               slow_period=slow_period,
               initial_capital=initial_capital)
    
    try:
        # 1. Initialize exchange connection
        exchange = BinanceExchange()
        await exchange.connect()
        
        # 2. Set up historical data manager
        data_manager = HistoricalDataManager(exchange)
        
        # 3. Fetch historical data
        logger.info("Fetching historical data...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=symbols,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # Check if we have data
        total_records = sum(len(df) for df in historical_data.values())
        if total_records == 0:
            logger.error("No historical data retrieved")
            return None
        
        logger.info("Historical data retrieved", total_records=total_records)
        
        # 4. Initialize strategy
        strategy_config = {
            'symbols': symbols,
            'fast_period': fast_period,
            'slow_period': slow_period,
            'position_size': 0.1  # 10% of capital per trade
        }
        
        strategy = MovingAverageCrossover(strategy_config)
        
        # 5. Configure backtest engine
        backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=0.0005,  # 0.05% slippage
            position_sizing='percentage',
            position_size=0.1  # 10% of capital
        )
        
        engine = BacktestEngine(backtest_config)
        engine._strategy_name = strategy.name  # Store strategy name for results
        
        # 6. Run backtest
        logger.info("Running backtest...")
        results = await engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=start_dt,
            end_date=end_dt
        )
        
        # 7. Analyze results
        analyzer = PerformanceAnalyzer(results)
        
        # Generate and display report
        report = analyzer.generate_report()
        print("\n" + report)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        # Save report
        report_path = results_dir / f"backtest_report_{timestamp}.txt"
        analyzer.generate_report(str(report_path))
        
        # Save raw results as JSON
        json_path = results_dir / f"backtest_data_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            'strategy_name': results['strategy_name'],
            'backtest_period': {
                'start': results['backtest_period']['start'].isoformat() if results['backtest_period']['start'] else None,
                'end': results['backtest_period']['end'].isoformat() if results['backtest_period']['end'] else None,
                'duration_days': results['backtest_period']['duration_days']
            },
            'capital': results['capital'],
            'trades': results['trades'],
            'risk_metrics': results['risk_metrics'],
            'config': {
                'symbols': symbols,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'initial_capital': initial_capital,
                'commission_rate': commission_rate,
                'interval': interval
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info("Backtest completed",
                   report_path=report_path,
                   json_path=json_path,
                   total_return=results['capital']['total_return_pct'])
        
        # Disconnect from exchange
        await exchange.disconnect()
        
        return results
        
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise

async def parameter_optimization():
    """
    Run parameter optimization for MA crossover strategy
    Tests different fast/slow period combinations
    """
    logger.info("Starting parameter optimization...")
    
    # Parameter ranges to test
    fast_periods = [5, 10, 15, 20]
    slow_periods = [20, 30, 50, 100]
    
    results = []
    
    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue
            
            logger.info("Testing parameters", fast=fast, slow=slow)
            
            try:
                result = await run_ma_crossover_backtest(
                    symbols=['BTCUSDT'],
                    fast_period=fast,
                    slow_period=slow,
                    initial_capital=10000
                )
                
                if result:
                    results.append({
                        'fast_period': fast,
                        'slow_period': slow,
                        'total_return': result['capital']['total_return_pct'],
                        'sharpe_ratio': result['risk_metrics']['sharpe_ratio'],
                        'max_drawdown': result['risk_metrics']['max_drawdown_pct'],
                        'total_trades': result['trades']['total'],
                        'win_rate': result['trades']['win_rate_pct']
                    })
                
            except Exception as e:
                logger.error("Parameter test failed", fast=fast, slow=slow, error=str(e))
                continue
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"{'Fast':<6} {'Slow':<6} {'Return%':<10} {'Sharpe':<8} {'Drawdown%':<12} {'Trades':<8} {'WinRate%':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['fast_period']:<6} {result['slow_period']:<6} "
              f"{result['total_return']:<10.2f} {result['sharpe_ratio']:<8.2f} "
              f"{result['max_drawdown']:<12.2f} {result['total_trades']:<8} "
              f"{result['win_rate']:<10.2f}")
    
    # Save optimization results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('backtest_results')
    results_dir.mkdir(exist_ok=True)
    
    optimization_path = results_dir / f"parameter_optimization_{timestamp}.json"
    with open(optimization_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Parameter optimization completed", results_file=optimization_path)
    
    return results

def main():
    """Command line interface for backtesting"""
    parser = argparse.ArgumentParser(description='Magicbot Backtesting System')
    
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols (default: BTCUSDT ETHUSDT)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD, default: 90 days ago)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--fast-period', type=int, default=10,
                       help='Fast MA period (default: 10)')
    parser.add_argument('--slow-period', type=int, default=30,
                       help='Slow MA period (default: 30)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (default: 0.001)')
    parser.add_argument('--interval', type=str, default='1h',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Timeframe (default: 1h)')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization')
    
    args = parser.parse_args()
    
    if args.optimize:
        asyncio.run(parameter_optimization())
    else:
        asyncio.run(run_ma_crossover_backtest(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            fast_period=args.fast_period,
            slow_period=args.slow_period,
            initial_capital=args.initial_capital,
            commission_rate=args.commission,
            interval=args.interval
        ))

if __name__ == "__main__":
    main()
