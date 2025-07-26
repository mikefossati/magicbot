#!/usr/bin/env python3
"""
Quick Scalping Strategy Parameter Optimization
Focuses on most impactful parameters with limited combinations
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple

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
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARN level for less noise
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def quick_optimize_scalping():
    """Quick optimization focusing on key parameters"""
    
    logger.info("Starting quick scalping optimization")
    
    # Initialize exchange
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Fetch data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)  # Shorter period for speed
        
        logger.info("Fetching 1 day of 1m data for scalping...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='1m',
            start_date=end_date - timedelta(days=1),
            end_date=end_date
        )
        
        # Define focused parameter combinations
        parameter_sets = [
            # Conservative scalping
            {'fast_ema': 3, 'slow_ema': 8, 'signal_ema': 21, 'volume_multiplier': 1.5, 
             'stop_loss_pct': 0.3, 'take_profit_pct': 0.6, 'position_size': 0.02},
            
            # Balanced scalping  
            {'fast_ema': 5, 'slow_ema': 13, 'signal_ema': 21, 'volume_multiplier': 1.5,
             'stop_loss_pct': 0.5, 'take_profit_pct': 1.0, 'position_size': 0.02},
             
            # Aggressive scalping
            {'fast_ema': 3, 'slow_ema': 8, 'signal_ema': 13, 'volume_multiplier': 2.0,
             'stop_loss_pct': 0.7, 'take_profit_pct': 1.5, 'position_size': 0.03},
             
            # High volume threshold
            {'fast_ema': 5, 'slow_ema': 13, 'signal_ema': 21, 'volume_multiplier': 2.5,
             'stop_loss_pct': 0.5, 'take_profit_pct': 1.0, 'position_size': 0.02},
             
            # Wide risk/reward
            {'fast_ema': 8, 'slow_ema': 21, 'signal_ema': 34, 'volume_multiplier': 1.2,
             'stop_loss_pct': 0.5, 'take_profit_pct': 2.0, 'position_size': 0.01},
             
            # Very fast EMAs
            {'fast_ema': 3, 'slow_ema': 5, 'signal_ema': 8, 'volume_multiplier': 1.8,
             'stop_loss_pct': 0.3, 'take_profit_pct': 0.9, 'position_size': 0.025},
             
            # Moderate speed
            {'fast_ema': 8, 'slow_ema': 13, 'signal_ema': 21, 'volume_multiplier': 1.3,
             'stop_loss_pct': 0.4, 'take_profit_pct': 1.2, 'position_size': 0.02},
             
            # Low volume requirement
            {'fast_ema': 5, 'slow_ema': 13, 'signal_ema': 21, 'volume_multiplier': 1.1,
             'stop_loss_pct': 0.6, 'take_profit_pct': 1.2, 'position_size': 0.02},
        ]
        
        results = []
        
        for i, params in enumerate(parameter_sets):
            logger.info("Testing parameter set", set=i+1, total=len(parameter_sets))
            
            try:
                # Add fixed parameters
                strategy_config = {
                    'symbols': ['BTCUSDT'],
                    'rsi_overbought': 75,
                    'rsi_oversold': 25,
                    'rsi_period': 14,
                    'volume_period': 10,
                    'min_price_movement': 0.1,
                    'consolidation_period': 3,
                    **params
                }
                
                # Create strategy
                strategy = create_strategy('ema_scalping_strategy', strategy_config)
                
                # Run backtest
                backtest_config = BacktestConfig(
                    initial_capital=10000.0,
                    commission_rate=0.001,
                    slippage_rate=0.0005,
                    position_sizing='percentage',
                    position_size=params['position_size']
                )
                
                engine = BacktestEngine(backtest_config)
                backtest_results = await engine.run_backtest(
                    strategy=strategy,
                    historical_data=historical_data,
                    start_date=end_date - timedelta(days=1),
                    end_date=end_date
                )
                
                # Calculate metrics
                total_return = backtest_results['capital']['total_return_pct']
                sharpe_ratio = backtest_results['risk_metrics']['sharpe_ratio']
                max_drawdown = backtest_results['risk_metrics']['max_drawdown_pct']
                total_trades = backtest_results['trades']['total']
                win_rate = backtest_results['trades']['win_rate_pct']
                profit_factor = backtest_results['trades']['profit_factor']
                
                # Calculate composite score
                score = calculate_score(total_return, sharpe_ratio, max_drawdown, win_rate, total_trades)
                
                result = {
                    'parameters': params,
                    'total_return_pct': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': max_drawdown,
                    'total_trades': total_trades,
                    'win_rate_pct': win_rate,
                    'profit_factor': profit_factor,
                    'composite_score': score,
                    'risk_reward_ratio': params['take_profit_pct'] / params['stop_loss_pct']
                }
                
                results.append(result)
                
                logger.info("Test completed",
                           return_pct=f"{total_return:.2f}%",
                           sharpe=f"{sharpe_ratio:.2f}",
                           trades=total_trades,
                           win_rate=f"{win_rate:.1f}%",
                           score=f"{score:.3f}")
                
            except Exception as e:
                logger.error("Parameter test failed", params=params, error=str(e))
        
        # Sort results by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Display results
        print("\n" + "=" * 100)
        print("QUICK SCALPING PARAMETER OPTIMIZATION RESULTS")
        print("=" * 100)
        print(f"{'Rank':<5} {'Return%':<8} {'Sharpe':<7} {'MaxDD%':<7} {'Trades':<7} {'WinRate%':<9} {'R:R':<5} {'Score':<6}")
        print("-" * 100)
        
        for i, result in enumerate(results, 1):
            print(
                f"{i:<5} "
                f"{result['total_return_pct']:<8.2f} "
                f"{result['sharpe_ratio']:<7.2f} "
                f"{result['max_drawdown_pct']:<7.2f} "
                f"{result['total_trades']:<7} "
                f"{result['win_rate_pct']:<9.1f} "
                f"{result['risk_reward_ratio']:<5.1f} "
                f"{result['composite_score']:<6.3f}"
            )
        
        print("\n" + "🏆 OPTIMAL PARAMETERS (Best Overall)")
        print("-" * 50)
        if results:
            best = results[0]
            params = best['parameters']
            print(f"Fast EMA: {params['fast_ema']}")
            print(f"Slow EMA: {params['slow_ema']}")
            print(f"Signal EMA: {params['signal_ema']}")
            print(f"Volume Multiplier: {params['volume_multiplier']}")
            print(f"Stop Loss %: {params['stop_loss_pct']}")
            print(f"Take Profit %: {params['take_profit_pct']}")
            print(f"Position Size: {params['position_size']}")
            print(f"Risk/Reward Ratio: {params['take_profit_pct']/params['stop_loss_pct']:.2f}")
            print(f"\nPerformance:")
            print(f"Total Return: {best['total_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {best['max_drawdown_pct']:.2f}%")
            print(f"Total Trades: {best['total_trades']}")
            print(f"Win Rate: {best['win_rate_pct']:.1f}%")
            print(f"Composite Score: {best['composite_score']:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        filename = results_dir / f"quick_scalping_optimization_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'optimization_timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2, default=str)
        
        logger.info("Results saved", filename=filename)
        
        return results
        
    finally:
        await exchange.disconnect()

def calculate_score(total_return: float, sharpe_ratio: float, max_drawdown: float, 
                   win_rate: float, total_trades: int) -> float:
    """Calculate composite performance score"""
    # Normalize metrics
    return_score = max(0, min(1, (total_return + 20) / 40))  # -20% to +20% -> 0 to 1
    sharpe_score = max(0, min(1, (sharpe_ratio + 1) / 3))    # -1 to +2 -> 0 to 1
    drawdown_score = max(0, 1 - max_drawdown / 30)           # 0% to 30% -> 1 to 0
    winrate_score = win_rate / 100                           # 0% to 100% -> 0 to 1
    trades_score = min(1, total_trades / 20)                 # 0 to 20+ trades -> 0 to 1
    
    # Weighted composite
    composite = (
        return_score * 0.30 +      # 30% weight on returns
        sharpe_score * 0.25 +      # 25% weight on risk-adjusted returns
        drawdown_score * 0.20 +    # 20% weight on drawdown control
        winrate_score * 0.15 +     # 15% weight on win rate
        trades_score * 0.10        # 10% weight on trade frequency
    )
    
    return composite

if __name__ == "__main__":
    asyncio.run(quick_optimize_scalping())