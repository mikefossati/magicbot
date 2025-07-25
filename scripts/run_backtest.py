#!/usr/bin/env python3
"""
Backtesting script for MagicBot strategies
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.backtesting.engine import BacktestEngine
from src.strategies.registry import StrategyRegistry
from src.core.config import load_config


def main():
    """Run backtesting for specified strategy"""
    parser = argparse.ArgumentParser(description="Run strategy backtesting")
    parser.add_argument("--strategy", required=True, help="Strategy name to backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Initial balance")
    
    args = parser.parse_args()
    
    config = load_config()
    registry = StrategyRegistry()
    strategy = registry.get_strategy(args.strategy)
    
    engine = BacktestEngine(config)
    results = engine.run_backtest(
        strategy=strategy,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    print(f"Backtest Results for {args.strategy}:")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")


if __name__ == "__main__":
    main()
