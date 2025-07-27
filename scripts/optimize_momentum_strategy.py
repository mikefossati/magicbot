#!/usr/bin/env python3
"""
Momentum Trading Strategy Parameter Optimization Script

This script optimizes the momentum trading strategy parameters to maximize profit
by testing different parameter combinations using the backtest engine directly.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import structlog
import itertools

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig
logger = structlog.get_logger()

class MomentumOptimizer:
    """Momentum strategy parameter optimizer"""
    
    def __init__(self):
        self.best_results = []
        
    async def load_test_data(self, symbol: str = 'BTCUSDT', days: int = 90) -> Dict[str, pd.DataFrame]:
        """Load historical data for testing"""
        logger.info("Generating synthetic trending data for optimization", symbol=symbol, days=days)
        return self._generate_synthetic_data(symbol, days)
    
    def _generate_synthetic_data(self, symbol: str, days: int) -> Dict[str, pd.DataFrame]:
        """Generate synthetic trending market data"""
        periods = days * 24  # Hourly data
        
        # Create multiple trend phases
        data_points = []
        base_price = 50000
        current_price = base_price
        
        for i in range(periods):
            timestamp = datetime.now() - timedelta(hours=periods-i)
            
            # Create trending phases
            if i < periods * 0.3:  # First 30% - uptrend
                trend = 0.002
                noise = np.random.normal(0, 0.01)
            elif i < periods * 0.6:  # Next 30% - consolidation
                trend = 0.0
                noise = np.random.normal(0, 0.005)
            else:  # Last 40% - strong uptrend
                trend = 0.003
                noise = np.random.normal(0, 0.015)
            
            # Price movement
            price_change = trend + noise
            current_price *= (1 + price_change)
            
            # OHLCV data
            open_price = current_price * (1 + np.random.normal(0, 0.001))
            high_price = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = current_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = current_price
            volume = np.random.uniform(1000, 5000) * (1 + abs(price_change) * 10)  # Higher volume on big moves
            
            data_points.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': open_price,
                'high': max(high_price, open_price, close_price),
                'low': min(low_price, open_price, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data_points)
        logger.info("Generated synthetic data", periods=periods, 
                   start_price=base_price, end_price=current_price)
        
        return {symbol: df}
    
    def get_parameter_combinations(self) -> List[Dict]:
        """Define parameter combinations to test"""
        param_grid = {
            'trend_ema_fast': [8, 12, 16],
            'trend_ema_slow': [21, 26, 32],
            'trend_strength_threshold': [0.01, 0.02, 0.03],
            'rsi_period': [10, 14, 18],
            'volume_surge_multiplier': [1.2, 1.5, 2.0],
            'volume_confirmation_required': [True, False],
            'momentum_alignment_required': [True, False],
            'base_position_size': [0.01, 0.02, 0.03],
            'stop_loss_atr_multiplier': [1.5, 2.0, 2.5],
            'take_profit_risk_reward': [2.0, 3.0, 4.0],
            'trend_strength_scaling': [True, False]
        }
        
        # Create more focused combinations to avoid combinatorial explosion
        focused_combinations = [
            # Conservative setups
            {
                'trend_ema_fast': 12, 'trend_ema_slow': 26, 'trend_strength_threshold': 0.02,
                'rsi_period': 14, 'volume_surge_multiplier': 1.5, 'volume_confirmation_required': True,
                'momentum_alignment_required': True, 'base_position_size': 0.02,
                'stop_loss_atr_multiplier': 2.0, 'take_profit_risk_reward': 3.0, 'trend_strength_scaling': True
            },
            # Aggressive setups
            {
                'trend_ema_fast': 8, 'trend_ema_slow': 21, 'trend_strength_threshold': 0.01,
                'rsi_period': 10, 'volume_surge_multiplier': 1.2, 'volume_confirmation_required': False,
                'momentum_alignment_required': False, 'base_position_size': 0.03,
                'stop_loss_atr_multiplier': 1.5, 'take_profit_risk_reward': 2.0, 'trend_strength_scaling': True
            },
            # Sensitive setups
            {
                'trend_ema_fast': 16, 'trend_ema_slow': 32, 'trend_strength_threshold': 0.03,
                'rsi_period': 18, 'volume_surge_multiplier': 2.0, 'volume_confirmation_required': True,
                'momentum_alignment_required': True, 'base_position_size': 0.01,
                'stop_loss_atr_multiplier': 2.5, 'take_profit_risk_reward': 4.0, 'trend_strength_scaling': False
            },
            # Fast momentum
            {
                'trend_ema_fast': 8, 'trend_ema_slow': 21, 'trend_strength_threshold': 0.015,
                'rsi_period': 10, 'volume_surge_multiplier': 1.3, 'volume_confirmation_required': True,
                'momentum_alignment_required': True, 'base_position_size': 0.025,
                'stop_loss_atr_multiplier': 1.8, 'take_profit_risk_reward': 2.5, 'trend_strength_scaling': True
            },
            # Relaxed filters
            {
                'trend_ema_fast': 12, 'trend_ema_slow': 26, 'trend_strength_threshold': 0.01,
                'rsi_period': 14, 'volume_surge_multiplier': 1.2, 'volume_confirmation_required': False,
                'momentum_alignment_required': False, 'base_position_size': 0.02,
                'stop_loss_atr_multiplier': 2.0, 'take_profit_risk_reward': 3.0, 'trend_strength_scaling': True
            }
        ]
        
        return focused_combinations
    
    async def test_parameter_combination(self, params: Dict, historical_data: Dict) -> Dict:
        """Test a single parameter combination"""
        try:
            # Add required base parameters
            config = {
                'symbols': ['BTCUSDT'],
                **params
            }
            
            # Create strategy
            strategy = MomentumTradingStrategy(config)
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=params['base_position_size']
            )
            
            # Run backtest
            engine = BacktestEngine(backtest_config)
            
            # Set up date range
            all_timestamps = []
            for symbol, df in historical_data.items():
                if len(df) > 0:
                    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                    all_timestamps.extend(timestamps)
            
            if not all_timestamps:
                return {'error': 'No timestamp data available'}
            
            start_date = min(all_timestamps)
            end_date = max(all_timestamps)
            
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add parameter info to results
            results['parameters'] = params
            results['parameter_hash'] = hash(str(sorted(params.items())))
            
            return results
            
        except Exception as e:
            logger.error("Error testing parameters", params=params, error=str(e))
            return {'error': str(e), 'parameters': params}
    
    async def optimize_parameters(self) -> List[Dict]:
        """Run parameter optimization"""
        logger.info("Starting momentum strategy parameter optimization")
        
        # Load test data
        historical_data = await self.load_test_data(symbol='BTCUSDT', days=90)
        
        # Get parameter combinations
        param_combinations = self.get_parameter_combinations()
        logger.info("Testing parameter combinations", total=len(param_combinations))
        
        # Test each combination
        results = []
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}", params=params)
            
            result = await self.test_parameter_combination(params, historical_data)
            if 'error' not in result:
                results.append(result)
                
                logger.info("Backtest completed",
                           total_return=result['capital']['total_return_pct'],
                           total_trades=result['trades']['total'],
                           win_rate=result['trades']['win_rate_pct'],
                           sharpe=result['risk_metrics']['sharpe_ratio'])
            else:
                logger.warning("Backtest failed", error=result['error'])
        
        # Sort by total return
        results.sort(key=lambda x: x['capital']['total_return_pct'], reverse=True)
        
        logger.info("Optimization completed", 
                   successful_tests=len(results),
                   best_return=results[0]['capital']['total_return_pct'] if results else 0)
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results"""
        if not results:
            return {'error': 'No successful results to analyze'}
        
        # Find best performers
        top_5 = results[:5]
        
        analysis = {
            'total_tests': len(results),
            'best_performer': {
                'parameters': top_5[0]['parameters'],
                'total_return_pct': top_5[0]['capital']['total_return_pct'],
                'total_trades': top_5[0]['trades']['total'],
                'win_rate_pct': top_5[0]['trades']['win_rate_pct'],
                'sharpe_ratio': top_5[0]['risk_metrics']['sharpe_ratio'],
                'max_drawdown_pct': top_5[0]['risk_metrics']['max_drawdown_pct']
            },
            'top_5_performers': []
        }
        
        for i, result in enumerate(top_5):
            analysis['top_5_performers'].append({
                'rank': i + 1,
                'parameters': result['parameters'],
                'metrics': {
                    'total_return_pct': result['capital']['total_return_pct'],
                    'total_trades': result['trades']['total'],
                    'win_rate_pct': result['trades']['win_rate_pct'],
                    'sharpe_ratio': result['risk_metrics']['sharpe_ratio'],
                    'max_drawdown_pct': result['risk_metrics']['max_drawdown_pct']
                }
            })
        
        # Parameter analysis
        param_performance = {}
        for result in results:
            for param, value in result['parameters'].items():
                if param not in param_performance:
                    param_performance[param] = {}
                if value not in param_performance[param]:
                    param_performance[param][value] = []
                param_performance[param][value].append(result['capital']['total_return_pct'])
        
        # Average performance by parameter value
        param_averages = {}
        for param, values in param_performance.items():
            param_averages[param] = {}
            for value, returns in values.items():
                param_averages[param][value] = np.mean(returns)
        
        analysis['parameter_analysis'] = param_averages
        
        return analysis

async def main():
    """Main optimization function"""
    optimizer = MomentumOptimizer()
    
    # Run optimization
    results = await optimizer.optimize_parameters()
    
    if not results:
        print("❌ No successful optimization results")
        return
    
    # Analyze results
    analysis = optimizer.analyze_results(results)
    
    # Print results
    print("\n" + "="*80)
    print("🎯 MOMENTUM STRATEGY OPTIMIZATION RESULTS")
    print("="*80)
    
    best = analysis['best_performer']
    print(f"📊 Best Configuration:")
    print(f"   Total Return: {best['total_return_pct']:.2f}%")
    print(f"   Total Trades: {best['total_trades']}")
    print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
    print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {best['max_drawdown_pct']:.2f}%")
    
    print(f"\n🔧 Best Parameters:")
    for param, value in best['parameters'].items():
        print(f"   {param}: {value}")
    
    print(f"\n🏆 Top 5 Performers:")
    for performer in analysis['top_5_performers']:
        metrics = performer['metrics']
        print(f"   #{performer['rank']}: {metrics['total_return_pct']:.2f}% return, "
              f"{metrics['total_trades']} trades, {metrics['win_rate_pct']:.1f}% win rate")
    
    print(f"\n📈 Parameter Analysis:")
    for param, values in analysis['parameter_analysis'].items():
        best_value = max(values.items(), key=lambda x: x[1])
        print(f"   {param}: Best value = {best_value[0]} (avg return: {best_value[1]:.2f}%)")
    
    print("\n" + "="*80)
    print("✅ Optimization Complete!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())