#!/usr/bin/env python3
"""
Day Trading Strategy Parameter Optimization
Systematically test different parameter combinations to find optimal settings
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
import pandas as pd

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager
from src.backtesting.engine import BacktestEngine, BacktestConfig
import structlog

# Configure logging for optimization (less verbose)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(40),  # ERROR level only
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ParameterOptimizer:
    """Parameter optimization engine for day trading strategy"""
    
    def __init__(self):
        self.exchange = None
        self.data_manager = None
        self.historical_data = None
        self.backtest_config = None
        
    async def initialize(self):
        """Initialize exchange and data manager"""
        self.exchange = BinanceExchange()
        await self.exchange.connect()
        self.data_manager = HistoricalDataManager(self.exchange)
        
        # Configure backtest
        self.backtest_config = BacktestConfig(
            initial_capital=10000.0,
            commission_rate=0.001,  # 0.1% commission
            slippage_rate=0.0005,   # 0.05% slippage
            position_sizing='percentage',
            position_size=0.02
        )
        
    async def load_data(self, days=30, interval="15m"):
        """Load historical data for optimization"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Loading {days} days of {interval} data...")
        self.historical_data = await self.data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in self.historical_data or len(self.historical_data['BTCUSDT']) < 100:
            raise ValueError("Insufficient historical data for optimization")
            
        print(f"✅ Loaded {len(self.historical_data['BTCUSDT'])} data points")
        
    def get_parameter_combinations(self):
        """Define parameter ranges for optimization"""
        
        # Define parameter ranges to test
        parameter_ranges = {
            # EMA periods - test different trend detection sensitivities
            'ema_configs': [
                {'fast_ema': 5, 'medium_ema': 13, 'slow_ema': 34},   # More sensitive
                {'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50},   # Default
                {'fast_ema': 12, 'medium_ema': 26, 'slow_ema': 55},  # Less sensitive
            ],
            
            # Volume requirements - test different volume confirmation levels
            'volume_multiplier': [1.0, 1.2, 1.5, 2.0],
            
            # Signal scoring thresholds - test different entry criteria
            'min_signal_score': [0.5, 0.6, 0.7, 0.8],
            
            # Support/resistance sensitivity
            'support_resistance_threshold': [0.5, 0.8, 1.0, 1.5],
            
            # Risk management parameters
            'risk_configs': [
                {'stop_loss_pct': 1.0, 'take_profit_pct': 2.0},
                {'stop_loss_pct': 1.5, 'take_profit_pct': 2.5},   # Default
                {'stop_loss_pct': 2.0, 'take_profit_pct': 3.0},
                {'stop_loss_pct': 1.5, 'take_profit_pct': 3.5},   # Higher R:R
            ],
            
            # Trading frequency
            'max_daily_trades': [2, 3, 5, 8],
        }
        
        # Generate all combinations
        combinations = []
        for ema_config in parameter_ranges['ema_configs']:
            for volume_mult in parameter_ranges['volume_multiplier']:
                for min_score in parameter_ranges['min_signal_score']:
                    for sr_threshold in parameter_ranges['support_resistance_threshold']:
                        for risk_config in parameter_ranges['risk_configs']:
                            for max_trades in parameter_ranges['max_daily_trades']:
                                
                                # Base configuration
                                config = {
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
                                
                                # Add variable parameters
                                config.update(ema_config)
                                config['volume_multiplier'] = volume_mult
                                config['min_signal_score'] = min_score
                                config['support_resistance_threshold'] = sr_threshold
                                config.update(risk_config)
                                config['max_daily_trades'] = max_trades
                                
                                combinations.append(config)
        
        return combinations
    
    async def test_parameter_combination(self, config, combination_id):
        """Test a single parameter combination"""
        try:
            # Create strategy with current parameters
            strategy = create_strategy('day_trading_strategy', config)
            
            # Run backtest
            engine = BacktestEngine(self.backtest_config)
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=self.historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Extract key metrics
            capital = results['capital']
            trades = results['trades']
            risk_metrics = results['risk_metrics']
            
            # Calculate optimization score (combination of return, sharpe, and drawdown)
            total_return = capital['total_return_pct']
            sharpe_ratio = risk_metrics['sharpe_ratio']
            max_drawdown = risk_metrics['max_drawdown_pct']
            win_rate = trades.get('win_rate_pct', 0)
            
            # Composite score: prioritize consistent profitable returns with low risk
            # Penalize high drawdown and reward good sharpe ratio
            score = (
                total_return * 0.3 +  # 30% weight on returns
                sharpe_ratio * 20 * 0.4 +  # 40% weight on risk-adjusted returns
                win_rate * 0.2 +  # 20% weight on win rate
                -max_drawdown * 0.1  # 10% penalty for drawdown
            )
            
            return {
                'combination_id': combination_id,
                'config': config,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': win_rate,
                'total_trades': trades['total'],
                'profit_factor': trades.get('profit_factor', 0),
                'score': score,
                'successful': True
            }
            
        except Exception as e:
            logger.error(f"Error testing combination {combination_id}", error=str(e))
            return {
                'combination_id': combination_id,
                'config': config,
                'error': str(e),
                'successful': False,
                'score': -1000  # Very low score for failed combinations
            }
    
    async def optimize_parameters(self, max_combinations=50):
        """Run parameter optimization"""
        print("=" * 80)
        print("DAY TRADING STRATEGY PARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Get all parameter combinations
        all_combinations = self.get_parameter_combinations()
        total_combinations = len(all_combinations)
        
        print(f"📊 Total parameter combinations: {total_combinations}")
        
        # Limit combinations if too many
        if total_combinations > max_combinations:
            print(f"⚠️ Limiting to {max_combinations} combinations for performance")
            # Sample combinations strategically (every nth combination)
            step = total_combinations // max_combinations
            combinations = all_combinations[::step][:max_combinations]
        else:
            combinations = all_combinations
        
        print(f"🔄 Testing {len(combinations)} parameter combinations...")
        print()
        
        results = []
        
        for i, config in enumerate(combinations):
            print(f"Testing combination {i+1}/{len(combinations)}...", end=" ")
            
            result = await self.test_parameter_combination(config, i+1)
            results.append(result)
            
            if result['successful']:
                print(f"✅ Score: {result['score']:.2f}, Return: {result['total_return_pct']:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def analyze_results(self, results):
        """Analyze optimization results and find best parameters"""
        successful_results = [r for r in results if r['successful']]
        
        if not successful_results:
            print("❌ No successful parameter combinations found!")
            return None
        
        # Sort by score (best first)
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        print(f"📊 Successful combinations: {len(successful_results)}/{len(results)}")
        print(f"📈 Best optimization score: {successful_results[0]['score']:.2f}")
        
        # Show top 10 results
        print(f"\n🏆 TOP 10 PARAMETER COMBINATIONS:")
        print("-" * 120)
        print(f"{'Rank':<4} {'Score':<8} {'Return%':<8} {'Sharpe':<7} {'WinRate%':<9} {'MaxDD%':<7} {'Trades':<7} {'Key Parameters'}")
        print("-" * 120)
        
        for i, result in enumerate(successful_results[:10]):
            config = result['config']
            key_params = (f"EMA:{config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}, "
                         f"Vol:{config['volume_multiplier']}, Score:{config['min_signal_score']}, "
                         f"Risk:{config['stop_loss_pct']}/{config['take_profit_pct']}")
            
            print(f"{i+1:<4} "
                  f"{result['score']:<8.2f} "
                  f"{result['total_return_pct']:<8.2f} "
                  f"{result['sharpe_ratio']:<7.2f} "
                  f"{result['win_rate_pct']:<9.1f} "
                  f"{result['max_drawdown_pct']:<7.2f} "
                  f"{result['total_trades']:<7} "
                  f"{key_params}")
        
        # Analyze best configuration
        best_config = successful_results[0]['config']
        best_result = successful_results[0]
        
        print(f"\n🥇 BEST CONFIGURATION:")
        print("-" * 50)
        print("EMA Settings:")
        print(f"  Fast EMA: {best_config['fast_ema']}")
        print(f"  Medium EMA: {best_config['medium_ema']}")
        print(f"  Slow EMA: {best_config['slow_ema']}")
        
        print("\nSignal Settings:")
        print(f"  Volume Multiplier: {best_config['volume_multiplier']}")
        print(f"  Min Signal Score: {best_config['min_signal_score']}")
        print(f"  Support/Resistance Threshold: {best_config['support_resistance_threshold']}%")
        
        print("\nRisk Management:")
        print(f"  Stop Loss: {best_config['stop_loss_pct']}%")
        print(f"  Take Profit: {best_config['take_profit_pct']}%")
        print(f"  Max Daily Trades: {best_config['max_daily_trades']}")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  Total Return: {best_result['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {best_result['win_rate_pct']:.1f}%")
        print(f"  Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {best_result['total_trades']}")
        print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        
        return best_config, successful_results
    
    def save_results(self, results, best_config):
        """Save optimization results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best configuration
        config_file = f"best_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   Detailed results: {results_file}")
        print(f"   Best configuration: {config_file}")
        
        return results_file, config_file
    
    async def cleanup(self):
        """Clean up resources"""
        if self.exchange:
            await self.exchange.disconnect()

async def main():
    """Main optimization function"""
    optimizer = ParameterOptimizer()
    
    try:
        # Initialize
        await optimizer.initialize()
        
        # Load data (30 days of 15-minute data for comprehensive testing)
        await optimizer.load_data(days=30, interval="15m")
        
        # Run optimization
        results = await optimizer.optimize_parameters(max_combinations=50)
        
        # Analyze results
        best_config, all_results = optimizer.analyze_results(results)
        
        if best_config:
            # Save results
            optimizer.save_results(results, best_config)
            
            print(f"\n🎯 OPTIMIZATION COMPLETE!")
            print(f"   Best configuration identified and saved")
            print(f"   Ready for live trading or further backtesting")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        logger.error("Optimization error", error=str(e))
        
    finally:
        await optimizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())