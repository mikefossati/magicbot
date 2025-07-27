#!/usr/bin/env python3
"""
Comprehensive Momentum Strategy Parameter Optimization

This script runs extensive backtests with different parameter combinations
to find the optimal configuration for maximum profit.
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
import json
from concurrent.futures import ThreadPoolExecutor
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

logger = structlog.get_logger()

class ComprehensiveMomentumOptimizer:
    """Comprehensive momentum strategy parameter optimizer"""
    
    def __init__(self):
        self.optimization_results = []
        self.best_configs = []
        
    def generate_synthetic_trending_data(self, symbol: str = 'BTCUSDT', days: int = 180) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive synthetic data with multiple trend phases"""
        periods = days * 24  # Hourly data
        
        logger.info("Generating comprehensive synthetic data", periods=periods, days=days)
        
        data_points = []
        base_price = 50000
        current_price = base_price
        
        # Create different market phases
        phase_length = periods // 6
        
        for i in range(periods):
            timestamp = datetime.now() - timedelta(hours=periods-i)
            
            # Define market phases
            phase = i // phase_length
            
            if phase == 0:  # Strong uptrend
                trend = 0.004
                volatility = 0.015
                volume_multiplier = 1.5
            elif phase == 1:  # Moderate uptrend
                trend = 0.002
                volatility = 0.01
                volume_multiplier = 1.2
            elif phase == 2:  # Consolidation
                trend = 0.0
                volatility = 0.005
                volume_multiplier = 0.8
            elif phase == 3:  # Strong uptrend (breakout)
                trend = 0.005
                volatility = 0.02
                volume_multiplier = 2.0
            elif phase == 4:  # Moderate correction
                trend = -0.001
                volatility = 0.015
                volume_multiplier = 1.3
            else:  # Final rally
                trend = 0.003
                volatility = 0.012
                volume_multiplier = 1.4
            
            # Add noise
            noise = np.random.normal(0, volatility)
            price_change = trend + noise
            current_price *= (1 + price_change)
            
            # OHLCV data with realistic spreads
            open_price = current_price * (1 + np.random.normal(0, 0.001))
            high_price = current_price * (1 + abs(np.random.normal(0, 0.008)))
            low_price = current_price * (1 - abs(np.random.normal(0, 0.008)))
            close_price = current_price
            
            # Volume with phase-dependent patterns
            base_volume = 3000 * volume_multiplier
            volume = base_volume + np.random.uniform(0, base_volume * 0.5)
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            data_points.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data_points)
        
        logger.info("Generated comprehensive synthetic data", 
                   start_price=base_price, 
                   end_price=current_price,
                   total_return=((current_price / base_price) - 1) * 100)
        
        return {symbol: df}
    
    def get_parameter_grid(self) -> Dict[str, List]:
        """Define comprehensive parameter grid for optimization"""
        return {
            # Trend Detection - Test multiple EMA combinations
            'trend_ema_fast': [3, 5, 8, 12],
            'trend_ema_slow': [8, 10, 15, 21, 26],
            'trend_strength_threshold': [0.0005, 0.001, 0.002, 0.005, 0.01],
            
            # RSI Parameters - Test different periods and sensitivity
            'rsi_period': [5, 7, 10, 14],
            
            # Volume Parameters - Test different confirmation levels
            'volume_surge_multiplier': [1.05, 1.1, 1.2, 1.5],
            'volume_confirmation_required': [True, False],
            
            # Entry Parameters - Test different confirmation requirements
            'momentum_alignment_required': [True, False],
            'breakout_lookback': [3, 5, 10, 15, 20],
            
            # Position Sizing - Test different sizing strategies
            'base_position_size': [0.02, 0.03, 0.05, 0.07],
            'max_position_size': [0.05, 0.08, 0.1, 0.15],
            'trend_strength_scaling': [True, False],
            
            # Risk Management - Test different risk/reward profiles
            'stop_loss_atr_multiplier': [1.5, 2.0, 3.0, 5.0],
            'take_profit_risk_reward': [1.2, 1.5, 2.0, 3.0]
        }
    
    def generate_parameter_combinations(self, max_combinations: int = 500) -> List[Dict]:
        """Generate strategic parameter combinations for testing"""
        
        # Get full parameter grid
        param_grid = self.get_parameter_grid()
        
        # Generate strategic combinations instead of full grid
        combinations = []
        
        # Strategy 1: Ultra-Fast Momentum (maximum signals)
        for trend_fast in [3, 5]:
            for trend_slow in [8, 10]:
                if trend_fast < trend_slow:
                    combinations.append({
                        'name': f'ultra_fast_{trend_fast}_{trend_slow}',
                        'trend_ema_fast': trend_fast,
                        'trend_ema_slow': trend_slow,
                        'trend_strength_threshold': 0.0005,
                        'rsi_period': 5,
                        'volume_surge_multiplier': 1.05,
                        'volume_confirmation_required': False,
                        'momentum_alignment_required': False,
                        'breakout_lookback': 3,
                        'base_position_size': 0.05,
                        'max_position_size': 0.1,
                        'trend_strength_scaling': False,
                        'stop_loss_atr_multiplier': 3.0,
                        'take_profit_risk_reward': 1.2
                    })
        
        # Strategy 2: Balanced Momentum (quality vs quantity)
        for threshold in [0.001, 0.002]:
            for rsi_period in [7, 10]:
                for rr_ratio in [1.5, 2.0]:
                    combinations.append({
                        'name': f'balanced_{threshold}_{rsi_period}_{rr_ratio}',
                        'trend_ema_fast': 5,
                        'trend_ema_slow': 15,
                        'trend_strength_threshold': threshold,
                        'rsi_period': rsi_period,
                        'volume_surge_multiplier': 1.1,
                        'volume_confirmation_required': True,
                        'momentum_alignment_required': False,
                        'breakout_lookback': 5,
                        'base_position_size': 0.03,
                        'max_position_size': 0.08,
                        'trend_strength_scaling': True,
                        'stop_loss_atr_multiplier': 2.0,
                        'take_profit_risk_reward': rr_ratio
                    })
        
        # Strategy 3: Conservative Quality (high win rate)
        for volume_req in [True, False]:
            for momentum_req in [True, False]:
                combinations.append({
                    'name': f'conservative_{volume_req}_{momentum_req}',
                    'trend_ema_fast': 8,
                    'trend_ema_slow': 21,
                    'trend_strength_threshold': 0.005,
                    'rsi_period': 14,
                    'volume_surge_multiplier': 1.5,
                    'volume_confirmation_required': volume_req,
                    'momentum_alignment_required': momentum_req,
                    'breakout_lookback': 15,
                    'base_position_size': 0.02,
                    'max_position_size': 0.05,
                    'trend_strength_scaling': True,
                    'stop_loss_atr_multiplier': 2.0,
                    'take_profit_risk_reward': 3.0
                })
        
        # Strategy 4: Aggressive Growth (high returns)
        for position_size in [0.05, 0.07]:
            for stop_multiplier in [3.0, 5.0]:
                combinations.append({
                    'name': f'aggressive_{position_size}_{stop_multiplier}',
                    'trend_ema_fast': 3,
                    'trend_ema_slow': 8,
                    'trend_strength_threshold': 0.001,
                    'rsi_period': 5,
                    'volume_surge_multiplier': 1.05,
                    'volume_confirmation_required': False,
                    'momentum_alignment_required': False,
                    'breakout_lookback': 3,
                    'base_position_size': position_size,
                    'max_position_size': position_size * 2,
                    'trend_strength_scaling': False,
                    'stop_loss_atr_multiplier': stop_multiplier,
                    'take_profit_risk_reward': 1.5
                })
        
        # Strategy 5: Adaptive Scaling (dynamic sizing)
        for lookback in [5, 10, 20]:
            combinations.append({
                'name': f'adaptive_{lookback}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': 0.002,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.2,
                'volume_confirmation_required': False,
                'momentum_alignment_required': True,
                'breakout_lookback': lookback,
                'base_position_size': 0.03,
                'max_position_size': 0.1,
                'trend_strength_scaling': True,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 2.0
            })
        
        # Add some random combinations for exploration
        np.random.seed(42)  # For reproducibility
        for i in range(50):
            combination = {}
            for param, values in param_grid.items():
                combination[param] = np.random.choice(values)
                
            # Ensure valid EMA relationship
            if combination['trend_ema_fast'] >= combination['trend_ema_slow']:
                combination['trend_ema_slow'] = combination['trend_ema_fast'] + np.random.choice([3, 5, 8])
            
            # Ensure valid position sizing
            if combination['base_position_size'] > combination['max_position_size']:
                combination['max_position_size'] = combination['base_position_size'] + 0.02
                
            combination['name'] = f'random_{i}'
            combinations.append(combination)
        
        logger.info("Generated parameter combinations", total=len(combinations))
        return combinations[:max_combinations]
    
    async def test_configuration(self, config: Dict, historical_data: Dict, test_id: int) -> Dict:
        """Test a single parameter configuration"""
        
        try:
            # Create full config
            full_config = {
                'symbols': ['BTCUSDT'],
                **{k: v for k, v in config.items() if k != 'name'}
            }
            
            # Create strategy
            strategy = MomentumTradingStrategy(full_config)
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=config['base_position_size']
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
                return {'error': 'No timestamp data available', 'config': config}
            
            start_date = min(all_timestamps)
            end_date = max(all_timestamps)
            
            # Run backtest
            start_time = time.time()
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            backtest_time = time.time() - start_time
            
            # Calculate additional metrics
            results['config'] = config
            results['config_name'] = config.get('name', f'config_{test_id}')
            results['backtest_time'] = backtest_time
            results['test_id'] = test_id
            
            # Calculate profit per trade
            if results['trades']['total'] > 0:
                results['profit_per_trade'] = results['capital']['total_return_pct'] / results['trades']['total']
            else:
                results['profit_per_trade'] = 0
                
            # Calculate risk-adjusted return (Sharpe-like metric)
            if results['risk_metrics']['volatility_pct'] > 0:
                results['risk_adjusted_return'] = results['capital']['total_return_pct'] / results['risk_metrics']['volatility_pct']
            else:
                results['risk_adjusted_return'] = 0
            
            logger.info(f"Backtest {test_id} completed",
                       config_name=config.get('name'),
                       return_pct=results['capital']['total_return_pct'],
                       trades=results['trades']['total'],
                       time=f"{backtest_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest {test_id} failed", config=config.get('name'), error=str(e))
            return {'error': str(e), 'config': config, 'test_id': test_id}
    
    async def run_comprehensive_optimization(self, max_combinations: int = 500) -> Dict:
        """Run comprehensive parameter optimization"""
        
        logger.info("Starting comprehensive momentum strategy optimization", max_combinations=max_combinations)
        
        # Generate test data
        historical_data = self.generate_synthetic_trending_data(days=180)
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(max_combinations)
        
        logger.info("Running parameter optimization", total_combinations=len(combinations))
        
        # Run all backtests
        results = []
        failed_tests = []
        
        # Process in batches to avoid memory issues
        batch_size = 20
        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(combinations)-1)//batch_size + 1}")
            
            # Run batch of tests
            batch_tasks = [
                self.test_configuration(config, historical_data, i + j)
                for j, config in enumerate(batch)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, dict) and 'error' not in result:
                    results.append(result)
                else:
                    failed_tests.append(result)
        
        logger.info("Optimization completed", 
                   successful_tests=len(results),
                   failed_tests=len(failed_tests))
        
        if not results:
            return {'error': 'No successful backtest results'}
        
        # Analyze results
        analysis = self.analyze_optimization_results(results)
        
        # Save detailed results
        self.save_optimization_results(results, analysis)
        
        return analysis
    
    def analyze_optimization_results(self, results: List[Dict]) -> Dict:
        """Analyze and rank optimization results"""
        
        # Sort by different metrics
        by_total_return = sorted(results, key=lambda x: x['capital']['total_return_pct'], reverse=True)
        by_sharpe = sorted(results, key=lambda x: x['risk_metrics']['sharpe_ratio'], reverse=True)
        by_win_rate = sorted(results, key=lambda x: x['trades']['win_rate_pct'], reverse=True)
        by_profit_factor = sorted([r for r in results if r['trades']['profit_factor'] != float('inf')], 
                                 key=lambda x: x['trades']['profit_factor'], reverse=True)
        by_risk_adjusted = sorted(results, key=lambda x: x.get('risk_adjusted_return', 0), reverse=True)
        
        # Find overall best performers
        top_10_overall = by_total_return[:10]
        
        # Strategy category analysis
        strategy_performance = {}
        for result in results:
            config_name = result.get('config_name', 'unknown')
            strategy_type = config_name.split('_')[0] if '_' in config_name else 'unknown'
            
            if strategy_type not in strategy_performance:
                strategy_performance[strategy_type] = []
            strategy_performance[strategy_type].append(result)
        
        # Calculate average performance by strategy type
        strategy_averages = {}
        for strategy_type, strategy_results in strategy_performance.items():
            if strategy_results:
                avg_return = np.mean([r['capital']['total_return_pct'] for r in strategy_results])
                avg_trades = np.mean([r['trades']['total'] for r in strategy_results])
                avg_win_rate = np.mean([r['trades']['win_rate_pct'] for r in strategy_results])
                
                strategy_averages[strategy_type] = {
                    'avg_return': avg_return,
                    'avg_trades': avg_trades,
                    'avg_win_rate': avg_win_rate,
                    'count': len(strategy_results),
                    'best_result': max(strategy_results, key=lambda x: x['capital']['total_return_pct'])
                }
        
        analysis = {
            'total_tests': len(results),
            'best_overall': {
                'by_total_return': by_total_return[0],
                'by_sharpe_ratio': by_sharpe[0],
                'by_win_rate': by_win_rate[0],
                'by_profit_factor': by_profit_factor[0] if by_profit_factor else None,
                'by_risk_adjusted': by_risk_adjusted[0]
            },
            'top_10_configs': top_10_overall,
            'strategy_type_analysis': strategy_averages,
            'parameter_analysis': self.analyze_parameter_impact(results)
        }
        
        return analysis
    
    def analyze_parameter_impact(self, results: List[Dict]) -> Dict:
        """Analyze the impact of individual parameters on performance"""
        
        parameter_impact = {}
        
        # Analyze each parameter
        parameters_to_analyze = [
            'trend_ema_fast', 'trend_ema_slow', 'trend_strength_threshold',
            'rsi_period', 'volume_surge_multiplier', 'volume_confirmation_required',
            'momentum_alignment_required', 'breakout_lookback', 'base_position_size',
            'stop_loss_atr_multiplier', 'take_profit_risk_reward'
        ]
        
        for param in parameters_to_analyze:
            param_performance = {}
            
            for result in results:
                param_value = result['config'].get(param)
                if param_value is not None:
                    if param_value not in param_performance:
                        param_performance[param_value] = []
                    param_performance[param_value].append(result['capital']['total_return_pct'])
            
            # Calculate statistics for each parameter value
            param_stats = {}
            for value, returns in param_performance.items():
                if len(returns) >= 3:  # Only analyze values with sufficient data
                    param_stats[value] = {
                        'avg_return': np.mean(returns),
                        'median_return': np.median(returns),
                        'std_return': np.std(returns),
                        'max_return': np.max(returns),
                        'min_return': np.min(returns),
                        'count': len(returns)
                    }
            
            if param_stats:
                # Find best value for this parameter
                best_value = max(param_stats.items(), key=lambda x: x[1]['avg_return'])
                parameter_impact[param] = {
                    'best_value': best_value[0],
                    'best_avg_return': best_value[1]['avg_return'],
                    'all_values': param_stats
                }
        
        return parameter_impact
    
    def save_optimization_results(self, results: List[Dict], analysis: Dict):
        """Save optimization results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"momentum_optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = []
            for result in results:
                json_result = self._convert_for_json(result)
                json_results.append(json_result)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save analysis summary
        analysis_file = f"momentum_optimization_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json_analysis = self._convert_for_json(analysis)
            json.dump(json_analysis, f, indent=2, default=str)
        
        logger.info("Optimization results saved", 
                   results_file=results_file,
                   analysis_file=analysis_file)
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def print_optimization_summary(self, analysis: Dict):
        """Print comprehensive optimization summary"""
        
        print("\n" + "="*100)
        print("🚀 COMPREHENSIVE MOMENTUM STRATEGY OPTIMIZATION RESULTS")
        print("="*100)
        
        # Best overall performers
        best_return = analysis['best_overall']['by_total_return']
        print(f"\n🏆 BEST OVERALL PERFORMANCE:")
        print(f"   Configuration: {best_return['config_name']}")
        print(f"   Total Return: {best_return['capital']['total_return_pct']:.2f}%")
        print(f"   Total Trades: {best_return['trades']['total']}")
        print(f"   Win Rate: {best_return['trades']['win_rate_pct']:.1f}%")
        print(f"   Sharpe Ratio: {best_return['risk_metrics']['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {best_return['risk_metrics']['max_drawdown_pct']:.2f}%")
        
        # Best by different metrics
        print(f"\n📊 BEST BY DIFFERENT METRICS:")
        
        best_sharpe = analysis['best_overall']['by_sharpe_ratio']
        print(f"   Best Sharpe Ratio: {best_sharpe['config_name']} ({best_sharpe['risk_metrics']['sharpe_ratio']:.2f})")
        
        best_win_rate = analysis['best_overall']['by_win_rate']
        print(f"   Best Win Rate: {best_win_rate['config_name']} ({best_win_rate['trades']['win_rate_pct']:.1f}%)")
        
        if analysis['best_overall']['by_profit_factor']:
            best_pf = analysis['best_overall']['by_profit_factor']
            print(f"   Best Profit Factor: {best_pf['config_name']} ({best_pf['trades']['profit_factor']:.2f})")
        
        # Top 10 configurations
        print(f"\n🔝 TOP 10 CONFIGURATIONS:")
        for i, config in enumerate(analysis['top_10_configs'][:10], 1):
            print(f"   #{i}: {config['config_name']} - {config['capital']['total_return_pct']:.2f}% return, "
                  f"{config['trades']['total']} trades, {config['trades']['win_rate_pct']:.1f}% win rate")
        
        # Strategy type analysis
        print(f"\n📈 STRATEGY TYPE ANALYSIS:")
        strategy_analysis = analysis['strategy_type_analysis']
        sorted_strategies = sorted(strategy_analysis.items(), 
                                 key=lambda x: x[1]['avg_return'], reverse=True)
        
        for strategy_type, stats in sorted_strategies:
            print(f"   {strategy_type.upper()}: {stats['avg_return']:.2f}% avg return "
                  f"({stats['count']} tests, best: {stats['best_result']['capital']['total_return_pct']:.2f}%)")
        
        # Parameter impact analysis
        print(f"\n🔧 OPTIMAL PARAMETER VALUES:")
        param_analysis = analysis['parameter_analysis']
        
        for param, impact in param_analysis.items():
            print(f"   {param}: {impact['best_value']} (avg return: {impact['best_avg_return']:.2f}%)")
        
        # Best configuration details
        print(f"\n⚙️  OPTIMAL CONFIGURATION DETAILS:")
        best_config = best_return['config']
        for param, value in best_config.items():
            if param != 'name':
                print(f"   {param}: {value}")
        
        print(f"\n📊 OPTIMIZATION STATISTICS:")
        print(f"   Total Tests: {analysis['total_tests']}")
        print(f"   Best Return: {best_return['capital']['total_return_pct']:.2f}%")
        print(f"   Best Sharpe: {best_sharpe['risk_metrics']['sharpe_ratio']:.2f}")
        print(f"   Best Win Rate: {best_win_rate['trades']['win_rate_pct']:.1f}%")
        
        print("\n" + "="*100)
        print("✅ COMPREHENSIVE OPTIMIZATION COMPLETE!")
        print("="*100)

async def main():
    """Run comprehensive optimization"""
    
    optimizer = ComprehensiveMomentumOptimizer()
    
    print("🚀 Starting Comprehensive Momentum Strategy Optimization...")
    print("   This will test hundreds of parameter combinations")
    print("   Estimated time: 10-20 minutes")
    print("   Please wait...")
    
    start_time = time.time()
    
    # Run optimization with many combinations
    analysis = await optimizer.run_comprehensive_optimization(max_combinations=300)
    
    total_time = time.time() - start_time
    
    if 'error' in analysis:
        print(f"❌ Optimization failed: {analysis['error']}")
        return
    
    # Print results
    optimizer.print_optimization_summary(analysis)
    
    print(f"\n⏱️  Total optimization time: {total_time/60:.1f} minutes")
    
    # Return best configuration for use
    return analysis['best_overall']['by_total_return']['config']

if __name__ == "__main__":
    best_config = asyncio.run(main())