#!/usr/bin/env python3
"""
Quick Day Trading Strategy Parameter Optimization
Test key parameter combinations to find optimal settings quickly
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

# Configure logging (minimal output)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(50),  # CRITICAL level only
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def quick_optimize():
    """Quick parameter optimization focusing on key parameters"""
    
    print("=" * 70)
    print("QUICK DAY TRADING PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Load historical data (14 days, 15min for quick testing)
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        print(f"📊 Loading 14 days of 15m data...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in historical_data or len(historical_data['BTCUSDT']) < 100:
            print("❌ Insufficient data")
            return
        
        print(f"✅ Loaded {len(historical_data['BTCUSDT'])} data points")
        
        # Configure backtest
        backtest_config = BacktestConfig(
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            position_sizing='percentage',
            position_size=0.02
        )
        
        # Define strategic parameter combinations (focused on most impactful parameters)
        test_configs = [
            # Current default
            {
                'name': 'Current Default',
                'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
                'volume_multiplier': 1.2, 'min_signal_score': 0.6,
                'support_resistance_threshold': 0.8,
                'stop_loss_pct': 1.5, 'take_profit_pct': 2.5,
                'max_daily_trades': 3
            },
            
            # More sensitive (faster signals)
            {
                'name': 'More Sensitive',
                'fast_ema': 5, 'medium_ema': 13, 'slow_ema': 34,
                'volume_multiplier': 1.0, 'min_signal_score': 0.5,
                'support_resistance_threshold': 1.0,
                'stop_loss_pct': 1.0, 'take_profit_pct': 2.0,
                'max_daily_trades': 5
            },
            
            # More conservative (higher quality signals)
            {
                'name': 'Conservative',
                'fast_ema': 12, 'medium_ema': 26, 'slow_ema': 55,
                'volume_multiplier': 1.5, 'min_signal_score': 0.7,
                'support_resistance_threshold': 0.5,
                'stop_loss_pct': 2.0, 'take_profit_pct': 3.0,
                'max_daily_trades': 2
            },
            
            # High volume confirmation
            {
                'name': 'Volume Focus',
                'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
                'volume_multiplier': 2.0, 'min_signal_score': 0.6,
                'support_resistance_threshold': 0.8,
                'stop_loss_pct': 1.5, 'take_profit_pct': 2.5,
                'max_daily_trades': 3
            },
            
            # Higher risk/reward ratio
            {
                'name': 'High R:R',
                'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
                'volume_multiplier': 1.2, 'min_signal_score': 0.6,
                'support_resistance_threshold': 0.8,
                'stop_loss_pct': 1.5, 'take_profit_pct': 4.0,
                'max_daily_trades': 3
            },
            
            # Tight stops, more trades
            {
                'name': 'Active Trading',
                'fast_ema': 6, 'medium_ema': 15, 'slow_ema': 40,
                'volume_multiplier': 1.1, 'min_signal_score': 0.55,
                'support_resistance_threshold': 1.0,
                'stop_loss_pct': 1.0, 'take_profit_pct': 1.8,
                'max_daily_trades': 8
            },
            
            # Quality over quantity
            {
                'name': 'Quality Focus',
                'fast_ema': 10, 'medium_ema': 25, 'slow_ema': 60,
                'volume_multiplier': 1.8, 'min_signal_score': 0.8,
                'support_resistance_threshold': 0.5,
                'stop_loss_pct': 1.8, 'take_profit_pct': 3.5,
                'max_daily_trades': 2
            }
        ]
        
        results = []
        
        print(f"\n🔄 Testing {len(test_configs)} strategic configurations...\n")
        
        for i, test_config in enumerate(test_configs):
            print(f"Testing {i+1}/{len(test_configs)}: {test_config['name']}...", end=" ")
            
            try:
                # Create full strategy config
                strategy_config = {
                    'symbols': ['BTCUSDT'],
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'rsi_neutral_high': 60,
                    'rsi_neutral_low': 40,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'volume_period': 20,
                    'pivot_period': 10,
                    'trailing_stop_pct': 1.0,
                    'session_start': "00:00",
                    'session_end': "23:59",
                    'position_size': 0.02
                }
                
                # Add test parameters
                for key in ['fast_ema', 'medium_ema', 'slow_ema', 'volume_multiplier', 
                           'min_signal_score', 'support_resistance_threshold',
                           'stop_loss_pct', 'take_profit_pct', 'max_daily_trades']:
                    strategy_config[key] = test_config[key]
                
                # Create strategy and run backtest
                strategy = create_strategy('day_trading_strategy', strategy_config)
                engine = BacktestEngine(backtest_config)
                
                result = await engine.run_backtest(
                    strategy=strategy,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Extract metrics
                capital = result['capital']
                trades = result['trades']
                risk_metrics = result['risk_metrics']
                
                total_return = capital['total_return_pct']
                sharpe_ratio = risk_metrics['sharpe_ratio']
                max_drawdown = risk_metrics['max_drawdown_pct']
                win_rate = trades.get('win_rate_pct', 0)
                total_trades = trades['total']
                profit_factor = trades.get('profit_factor', 0)
                
                # Calculate optimization score
                score = (
                    total_return * 0.3 +
                    sharpe_ratio * 20 * 0.4 +
                    win_rate * 0.2 +
                    -max_drawdown * 0.1
                )
                
                results.append({
                    'name': test_config['name'],
                    'config': strategy_config,
                    'total_return_pct': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': max_drawdown,
                    'win_rate_pct': win_rate,
                    'total_trades': total_trades,
                    'profit_factor': profit_factor,
                    'score': score,
                    'successful': True
                })
                
                print(f"✅ Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, Score: {score:.2f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}...")
                results.append({
                    'name': test_config['name'],
                    'error': str(e),
                    'successful': False,
                    'score': -1000
                })
        
        # Analyze results
        successful_results = [r for r in results if r['successful']]
        
        if not successful_results:
            print("\n❌ No successful configurations!")
            return
        
        # Sort by score
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        print(f"✅ Successful: {len(successful_results)}/{len(results)} configurations")
        
        # Results table
        print(f"\n{'Configuration':<15} {'Return%':<8} {'Sharpe':<7} {'Win%':<6} {'Trades':<7} {'MaxDD%':<7} {'Score':<8}")
        print("-" * 70)
        
        for result in successful_results:
            print(f"{result['name']:<15} "
                  f"{result['total_return_pct']:<8.2f} "
                  f"{result['sharpe_ratio']:<7.2f} "
                  f"{result['win_rate_pct']:<6.1f} "
                  f"{result['total_trades']:<7} "
                  f"{result['max_drawdown_pct']:<7.2f} "
                  f"{result['score']:<8.2f}")
        
        # Best configuration details
        best = successful_results[0]
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print("-" * 40)
        
        config = best['config']
        print(f"EMA Periods: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
        print(f"Volume Multiplier: {config['volume_multiplier']}")
        print(f"Min Signal Score: {config['min_signal_score']}")
        print(f"S/R Threshold: {config['support_resistance_threshold']}%")
        print(f"Risk Management: {config['stop_loss_pct']}% SL / {config['take_profit_pct']}% TP")
        print(f"Max Daily Trades: {config['max_daily_trades']}")
        
        print(f"\n📊 Performance:")
        print(f"   Return: {best['total_return_pct']:.2f}%")
        print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
        print(f"   Total Trades: {best['total_trades']}")
        print(f"   Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        
        # Save best config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = f"optimized_day_trading_config_{timestamp}.json"
        
        with open(config_file, 'w') as f:
            json.dump(best['config'], f, indent=2)
        
        print(f"\n💾 Best configuration saved to: {config_file}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if best['total_return_pct'] > 3:
            print("✅ Strong performance - recommended for live trading")
        elif best['total_return_pct'] > 1:
            print("⚠️ Moderate performance - consider further optimization")
        else:
            print("❌ Poor performance - strategy needs significant improvements")
            
        if best['sharpe_ratio'] > 1.0:
            print("✅ Good risk-adjusted returns")
        elif best['sharpe_ratio'] > 0.5:
            print("⚠️ Acceptable risk-adjusted returns")
        else:
            print("❌ Poor risk-adjusted returns")
            
        print(f"\n{'='*70}")
        
        return best['config']
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(quick_optimize())