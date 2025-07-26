#!/usr/bin/env python3
"""
Magicbot Multi-Strategy Backtest Runner
Run backtests with multiple strategies simultaneously
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
from typing import Dict, List, Any
import pandas as pd

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
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class MultiStrategyBacktester:
    """Handles backtesting multiple strategies simultaneously"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategies = {}
        self.strategy_results = {}
        self.portfolio_results = {}
        
    def add_strategy(self, name: str, strategy_config: Dict[str, Any]):
        """Add a strategy to the backtest"""
        try:
            strategy = create_strategy(name, strategy_config)
            self.strategies[name] = strategy
            logger.info("Strategy added", name=name, config=strategy_config)
        except Exception as e:
            logger.error("Failed to add strategy", name=name, error=str(e))
            raise
    
    async def run_individual_backtests(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ):
        """Run backtests for each strategy individually"""
        
        for strategy_name, strategy in self.strategies.items():
            logger.info("Running backtest", strategy=strategy_name)
            
            try:
                # Create separate engine for each strategy
                engine = BacktestEngine(self.config)
                engine._strategy_name = strategy_name
                
                # Run backtest
                results = await engine.run_backtest(
                    strategy=strategy,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                self.strategy_results[strategy_name] = results
                
                logger.info("Strategy backtest completed",
                           strategy=strategy_name,
                           total_return=results['capital']['total_return_pct'],
                           total_trades=results['trades']['total'])
                
            except Exception as e:
                logger.error("Strategy backtest failed", strategy=strategy_name, error=str(e))
                continue
    
    async def run_portfolio_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        allocation_method: str = 'equal_weight'
    ):
        """Run a portfolio backtest with multiple strategies"""
        
        logger.info("Running portfolio backtest", allocation_method=allocation_method)
        
        # Create portfolio config with reduced position size per strategy
        portfolio_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
            position_sizing=self.config.position_sizing,
            position_size=self.config.position_size / len(self.strategies)  # Divide capital among strategies
        )
        
        # Portfolio tracking
        total_capital = self.config.initial_capital
        portfolio_trades = []
        portfolio_equity_curve = []
        strategy_engines = {}
        
        # Initialize engines for each strategy
        for strategy_name, strategy in self.strategies.items():
            engine = BacktestEngine(portfolio_config)
            engine._strategy_name = strategy_name
            strategy_engines[strategy_name] = engine
        
        # Get all timestamps for iteration
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)
        
        sorted_timestamps = sorted(all_timestamps)
        
        logger.info("Processing portfolio backtest", total_timestamps=len(sorted_timestamps))
        
        # Simulate day-by-day portfolio performance
        for i, timestamp in enumerate(sorted_timestamps):
            current_data = {}
            
            # Get data up to current timestamp for each symbol
            for symbol, df in historical_data.items():
                mask = df.index <= timestamp
                if mask.any():
                    current_data[symbol] = df[mask].to_dict('records')
            
            # Generate signals from all strategies
            all_signals = []
            for strategy_name, strategy in self.strategies.items():
                try:
                    signals = await strategy.generate_signals(current_data)
                    for signal in signals:
                        signal.metadata = signal.metadata or {}
                        signal.metadata['strategy'] = strategy_name
                        all_signals.append(signal)
                except Exception as e:
                    logger.debug("Signal generation failed", 
                               strategy=strategy_name, 
                               timestamp=timestamp, 
                               error=str(e))
                    continue
            
            # Execute signals through respective engines
            for signal in all_signals:
                strategy_name = signal.metadata.get('strategy')
                if strategy_name in strategy_engines:
                    engine = strategy_engines[strategy_name]
                    # Process signal through engine (simplified)
                    # In a real implementation, you'd need to adapt the engine
                    # to handle individual signal processing
            
            # Update portfolio equity curve every 24 hours or at significant intervals
            if i % 24 == 0 or i == len(sorted_timestamps) - 1:
                total_portfolio_value = sum(
                    engine.capital for engine in strategy_engines.values()
                )
                portfolio_equity_curve.append((timestamp, total_portfolio_value))
        
        # Aggregate results from all strategy engines
        total_trades = sum(len(engine.trades) for engine in strategy_engines.values())
        total_capital = sum(engine.capital for engine in strategy_engines.values())
        
        # Calculate portfolio performance metrics
        portfolio_return = (total_capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        self.portfolio_results = {
            'strategy_name': 'Multi-Strategy Portfolio',
            'total_return_pct': portfolio_return,
            'final_capital': total_capital,
            'total_trades': total_trades,
            'equity_curve': portfolio_equity_curve,
            'strategy_breakdown': {
                name: {
                    'capital': engine.capital,
                    'trades': len(engine.trades),
                    'return_pct': (engine.capital - portfolio_config.initial_capital) / portfolio_config.initial_capital * 100
                }
                for name, engine in strategy_engines.items()
            }
        }
        
        logger.info("Portfolio backtest completed",
                   total_return=portfolio_return,
                   final_capital=total_capital,
                   total_trades=total_trades)
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report"""
        
        report = []
        report.append("=" * 100)
        report.append("MULTI-STRATEGY BACKTEST COMPARISON REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Individual strategy performance
        report.append("📊 INDIVIDUAL STRATEGY PERFORMANCE")
        report.append("-" * 60)
        report.append(f"{'Strategy':<20} {'Return %':<12} {'Trades':<8} {'Win Rate %':<12} {'Sharpe':<8} {'Max DD %':<10}")
        report.append("-" * 60)
        
        for strategy_name, results in self.strategy_results.items():
            report.append(
                f"{strategy_name:<20} "
                f"{results['capital']['total_return_pct']:<12.2f} "
                f"{results['trades']['total']:<8} "
                f"{results['trades']['win_rate_pct']:<12.2f} "
                f"{results['risk_metrics']['sharpe_ratio']:<8.2f} "
                f"{results['risk_metrics']['max_drawdown_pct']:<10.2f}"
            )
        
        report.append("")
        
        # Portfolio performance (if available)
        if self.portfolio_results:
            report.append("🎯 PORTFOLIO PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Portfolio Return: {self.portfolio_results['total_return_pct']:.2f}%")
            report.append(f"Final Capital: ${self.portfolio_results['final_capital']:,.2f}")
            report.append(f"Total Trades: {self.portfolio_results['total_trades']}")
            report.append("")
            
            report.append("Strategy Breakdown:")
            for name, breakdown in self.portfolio_results['strategy_breakdown'].items():
                report.append(f"  {name}: {breakdown['return_pct']:.2f}% ({breakdown['trades']} trades)")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_results(self, results_dir: Path, timestamp: str):
        """Save all backtest results"""
        
        # Save individual strategy results
        for strategy_name, results in self.strategy_results.items():
            strategy_file = results_dir / f"backtest_{strategy_name}_{timestamp}.json"
            
            # Convert to JSON-serializable format
            json_results = {
                'strategy_name': results['strategy_name'],
                'backtest_period': {
                    'start': results['backtest_period']['start'].isoformat() if results['backtest_period']['start'] else None,
                    'end': results['backtest_period']['end'].isoformat() if results['backtest_period']['end'] else None,
                    'duration_days': results['backtest_period']['duration_days']
                },
                'capital': results['capital'],
                'trades': results['trades'],
                'risk_metrics': results['risk_metrics']
            }
            
            with open(strategy_file, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        # Save portfolio results
        if self.portfolio_results:
            portfolio_file = results_dir / f"portfolio_backtest_{timestamp}.json"
            with open(portfolio_file, 'w') as f:
                json.dump(self.portfolio_results, f, indent=2, default=str)
        
        # Save comparison report
        report_file = results_dir / f"multi_strategy_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_comparison_report())
        
        logger.info("Results saved", 
                   individual_strategies=len(self.strategy_results),
                   portfolio_saved=bool(self.portfolio_results),
                   report_file=report_file)

async def run_multi_strategy_backtest(
    strategy_names: List[str],
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 10000,
    commission_rate: float = 0.001,
    interval: str = '1h',
    run_portfolio: bool = True
):
    """
    Run multi-strategy backtest
    
    Args:
        strategy_names: List of strategy names to test
        symbols: List of trading pairs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital in USD
        commission_rate: Commission rate
        interval: Timeframe
        run_portfolio: Whether to run portfolio backtest
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
    
    logger.info("Starting multi-strategy backtest",
               strategies=strategy_names,
               symbols=symbols,
               start_date=start_date,
               end_date=end_date,
               initial_capital=initial_capital)
    
    try:
        # 1. Initialize exchange connection
        exchange = BinanceExchange()
        await exchange.connect()
        
        # 2. Fetch historical data
        data_manager = HistoricalDataManager(exchange)
        logger.info("Fetching historical data...")
        
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=symbols,
            interval=interval,
            start_date=start_dt,
            end_date=end_dt
        )
        
        total_records = sum(len(df) for df in historical_data.values())
        if total_records == 0:
            logger.error("No historical data retrieved")
            return None
        
        logger.info("Historical data retrieved", total_records=total_records)
        
        # 3. Load configuration
        config = load_config()
        
        # 4. Initialize multi-strategy backtester
        backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=0.0005,
            position_sizing='percentage',
            position_size=0.1
        )
        
        backtester = MultiStrategyBacktester(backtest_config)
        
        # 5. Add strategies
        for strategy_name in strategy_names:
            if strategy_name in config['strategies']:
                strategy_config = config['strategies'][strategy_name].copy()
                strategy_config['symbols'] = symbols
                backtester.add_strategy(strategy_name, strategy_config)
            else:
                logger.warning("Strategy not found in config", strategy=strategy_name)
                # Use default config
                default_config = {
                    'symbols': symbols,
                    'position_size': 0.1 / len(strategy_names)
                }
                backtester.add_strategy(strategy_name, default_config)
        
        # 6. Run individual strategy backtests
        await backtester.run_individual_backtests(historical_data, start_dt, end_dt)
        
        # 7. Run portfolio backtest (optional)
        if run_portfolio and len(backtester.strategies) > 1:
            await backtester.run_portfolio_backtest(historical_data, start_dt, end_dt)
        
        # 8. Generate and display report
        report = backtester.generate_comparison_report()
        print("\n" + report)
        
        # 9. Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        backtester.save_results(results_dir, timestamp)
        
        # Disconnect from exchange
        await exchange.disconnect()
        
        return {
            'individual_results': backtester.strategy_results,
            'portfolio_results': backtester.portfolio_results
        }
        
    except Exception as e:
        logger.error("Multi-strategy backtest failed", error=str(e))
        raise

def main():
    """Command line interface for multi-strategy backtesting"""
    parser = argparse.ArgumentParser(description='Magicbot Multi-Strategy Backtesting System')
    
    # Get available strategies
    available_strategies = list(get_available_strategies().keys())
    
    parser.add_argument('--strategies', nargs='+', 
                       choices=available_strategies,
                       default=['ma_crossover', 'rsi_strategy'],
                       help=f'Strategies to test (choices: {available_strategies})')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Trading symbols (default: BTCUSDT ETHUSDT)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD, default: 90 days ago)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (default: 0.001)')
    parser.add_argument('--interval', type=str, default='1h',
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Timeframe (default: 1h)')
    parser.add_argument('--no-portfolio', action='store_true',
                       help='Skip portfolio backtest (individual strategies only)')
    
    args = parser.parse_args()
    
    # Show available strategies if none specified correctly
    if not args.strategies:
        print(f"Available strategies: {', '.join(available_strategies)}")
        return
    
    asyncio.run(run_multi_strategy_backtest(
        strategy_names=args.strategies,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        interval=args.interval,
        run_portfolio=not args.no_portfolio
    ))

if __name__ == "__main__":
    main()