#!/usr/bin/env python3
"""
Trading bot startup script for MagicBot
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import load_config
from src.strategies.registry import StrategyRegistry
from src.exchanges.binance_exchange import BinanceExchange
from src.risk.risk_manager import RiskManager
from src.core.events import EventBus


def main():
    """Start the trading bot"""
    parser = argparse.ArgumentParser(description="Start MagicBot trading")
    parser.add_argument("--strategy", required=True, help="Strategy name to run")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Initialize components
    event_bus = EventBus()
    exchange = BinanceExchange(config.exchanges.binance)
    risk_manager = RiskManager(config.risk)
    
    # Get strategy
    registry = StrategyRegistry()
    strategy = registry.get_strategy(args.strategy)
    
    print(f"Starting MagicBot with strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Dry run: {args.dry_run}")
    
    # Start trading loop
    try:
        strategy.start(
            exchange=exchange,
            risk_manager=risk_manager,
            event_bus=event_bus,
            symbol=args.symbol,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\nShutting down MagicBot...")
        strategy.stop()


if __name__ == "__main__":
    main()
