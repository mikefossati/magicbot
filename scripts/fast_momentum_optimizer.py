#!/usr/bin/env python3
"""
Fast Momentum Strategy Parameter Optimization

Efficient optimization with focused parameter combinations and smaller datasets.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import structlog
import json
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

logger = structlog.get_logger()

class FastMomentumOptimizer:
    """Fast and efficient momentum strategy optimizer"""
    
    def __init__(self):
        self.results = []
    
    def generate_test_data(self, days: int = 60) -> Dict[str, pd.DataFrame]:
        """Generate focused test data with clear trending patterns"""
        periods = days * 24  # Hourly data
        
        data_points = []
        base_price = 50000
        current_price = base_price
        
        # Create strong trending phases for better signal testing
        for i in range(periods):
            timestamp = datetime.now() - timedelta(hours=periods-i)
            
            # Define clear market phases
            phase = i // (periods // 4)
            
            if phase == 0:  # Strong uptrend
                trend = 0.003
                volatility = 0.01
                volume_base = 4000
            elif phase == 1:  # Consolidation  
                trend = 0.0005
                volatility = 0.005
                volume_base = 2500
            elif phase == 2:  # Breakout uptrend
                trend = 0.004
                volatility = 0.015
                volume_base = 6000
            else:  # Final rally
                trend = 0.002
                volatility = 0.008
                volume_base = 3500
            
            # Price movement
            noise = np.random.normal(0, volatility)
            price_change = trend + noise
            current_price *= (1 + price_change)
            
            # OHLCV with realistic patterns
            open_price = current_price * (1 + np.random.normal(0, 0.001))
            high_price = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = current_price * (1 - abs(np.random.normal(0, 0.005)))
            close_price = current_price
            volume = volume_base * (1 + np.random.uniform(-0.3, 0.8))
            
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
        logger.info("Generated test data", 
                   periods=periods,
                   start_price=base_price,
                   end_price=current_price,
                   return_pct=((current_price/base_price)-1)*100)
        
        return {'BTCUSDT': df}
    
    def get_focused_configurations(self) -> List[Dict]:
        """Get focused set of strategic configurations to test"""
        
        configs = []
        
        # Ultra-Fast Momentum (Current Default)
        configs.append({
            'name': 'current_default',
            'trend_ema_fast': 5,
            'trend_ema_slow': 10,
            'trend_strength_threshold': 0.001,
            'rsi_period': 7,
            'volume_surge_multiplier': 1.1,
            'volume_confirmation_required': False,
            'momentum_alignment_required': False,
            'breakout_lookback': 5,
            'base_position_size': 0.05,
            'max_position_size': 0.1,
            'trend_strength_scaling': False,
            'stop_loss_atr_multiplier': 5.0,
            'take_profit_risk_reward': 1.5
        })
        
        # Test different EMA combinations
        ema_combinations = [
            (3, 8), (5, 12), (8, 15), (5, 21), (12, 26)
        ]
        
        for fast, slow in ema_combinations:
            configs.append({
                'name': f'ema_{fast}_{slow}',
                'trend_ema_fast': fast,
                'trend_ema_slow': slow,
                'trend_strength_threshold': 0.001,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': 0.05,
                'max_position_size': 0.1,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 3.0,
                'take_profit_risk_reward': 2.0
            })
        
        # Test different trend thresholds
        thresholds = [0.0005, 0.001, 0.002, 0.005]
        for threshold in thresholds:
            configs.append({
                'name': f'threshold_{threshold}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': threshold,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': 0.03,
                'max_position_size': 0.08,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 2.5,
                'take_profit_risk_reward': 2.0
            })
        
        # Test different RSI periods
        rsi_periods = [5, 7, 10, 14]
        for rsi in rsi_periods:
            configs.append({
                'name': f'rsi_{rsi}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': 0.001,
                'rsi_period': rsi,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': 0.03,
                'max_position_size': 0.08,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 2.5,
                'take_profit_risk_reward': 2.0
            })
        
        # Test different risk/reward ratios
        rr_ratios = [1.2, 1.5, 2.0, 2.5, 3.0]
        for rr in rr_ratios:
            configs.append({
                'name': f'rr_{rr}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': 0.001,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': 0.03,
                'max_position_size': 0.08,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': rr
            })
        
        # Test different position sizes
        position_sizes = [(0.02, 0.05), (0.03, 0.08), (0.05, 0.1), (0.07, 0.15)]
        for base, max_size in position_sizes:
            configs.append({
                'name': f'size_{base}_{max_size}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': 0.001,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': base,
                'max_position_size': max_size,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 2.0
            })
        
        # Test different stop loss distances
        stop_multipliers = [1.5, 2.0, 3.0, 4.0, 5.0]
        for stop in stop_multipliers:
            configs.append({
                'name': f'stop_{stop}',
                'trend_ema_fast': 5,
                'trend_ema_slow': 12,
                'trend_strength_threshold': 0.001,
                'rsi_period': 7,
                'volume_surge_multiplier': 1.1,
                'volume_confirmation_required': False,
                'momentum_alignment_required': False,
                'breakout_lookback': 5,
                'base_position_size': 0.03,
                'max_position_size': 0.08,
                'trend_strength_scaling': False,
                'stop_loss_atr_multiplier': stop,
                'take_profit_risk_reward': 2.0
            })
        
        # Test filters enabled/disabled
        filter_combinations = [
            (True, True), (True, False), (False, True), (False, False)
        ]
        for vol_conf, mom_align in filter_combinations:
            configs.append({
                'name': f'filters_{vol_conf}_{mom_align}',
                'trend_ema_fast': 8,
                'trend_ema_slow': 21,
                'trend_strength_threshold': 0.002,
                'rsi_period': 14,
                'volume_surge_multiplier': 1.3,
                'volume_confirmation_required': vol_conf,
                'momentum_alignment_required': mom_align,
                'breakout_lookback': 10,
                'base_position_size': 0.02,
                'max_position_size': 0.06,
                'trend_strength_scaling': True,
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 2.5
            })
        
        logger.info("Generated focused configurations", total=len(configs))
        return configs
    
    async def test_configuration(self, config: Dict, historical_data: Dict) -> Dict:
        """Test a single configuration efficiently"""
        
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
            
            # Get timestamps
            timestamps = pd.to_datetime(historical_data['BTCUSDT']['timestamp'], unit='ms')
            start_date = timestamps.min()
            end_date = timestamps.max()
            
            # Run backtest
            start_time = time.time()
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add metadata
            results['config'] = config
            results['config_name'] = config['name']
            results['backtest_time'] = time.time() - start_time
            
            # Calculate additional metrics
            total_return = results['capital']['total_return_pct']
            total_trades = results['trades']['total']
            win_rate = results['trades']['win_rate_pct']
            
            if total_trades > 0:
                results['return_per_trade'] = total_return / total_trades
                results['trades_per_day'] = total_trades / 60  # 60 days
            else:
                results['return_per_trade'] = 0
                results['trades_per_day'] = 0
                
            # Risk-adjusted metrics
            volatility = results['risk_metrics']['volatility_pct']
            if volatility > 0:
                results['risk_adjusted_return'] = total_return / volatility
            else:
                results['risk_adjusted_return'] = 0
            
            return results
            
        except Exception as e:
            logger.error("Configuration test failed", config=config['name'], error=str(e))
            return {'error': str(e), 'config': config}
    
    async def run_optimization(self) -> Dict:
        """Run fast optimization"""
        
        print("🚀 Starting Fast Momentum Strategy Optimization...")
        
        # Generate test data
        historical_data = self.generate_test_data(days=60)
        
        # Get configurations to test
        configurations = self.get_focused_configurations()
        
        print(f"   Testing {len(configurations)} strategic configurations")
        print(f"   Using 60 days of synthetic data ({60*24} data points)")
        
        # Run tests
        results = []
        failed_tests = []
        
        for i, config in enumerate(configurations):
            print(f"   Testing {i+1}/{len(configurations)}: {config['name']}")
            
            result = await self.test_configuration(config, historical_data)
            
            if 'error' not in result:
                results.append(result)
            else:
                failed_tests.append(result)
        
        if not results:
            return {'error': 'No successful test results'}
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        return analysis
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results"""
        
        # Sort by different metrics
        by_return = sorted(results, key=lambda x: x['capital']['total_return_pct'], reverse=True)
        by_sharpe = sorted(results, key=lambda x: x['risk_metrics']['sharpe_ratio'], reverse=True)
        by_win_rate = sorted(results, key=lambda x: x['trades']['win_rate_pct'], reverse=True)
        by_trades_count = sorted(results, key=lambda x: x['trades']['total'], reverse=True)
        by_return_per_trade = sorted(results, key=lambda x: x.get('return_per_trade', 0), reverse=True)
        
        # Filter out infinite profit factors
        valid_pf = [r for r in results if r['trades']['profit_factor'] != float('inf') and r['trades']['profit_factor'] > 0]
        by_profit_factor = sorted(valid_pf, key=lambda x: x['trades']['profit_factor'], reverse=True)
        
        # Parameter impact analysis
        param_analysis = {}
        
        # Analyze EMA combinations
        ema_results = [r for r in results if 'ema_' in r['config_name']]
        if ema_results:
            best_ema = max(ema_results, key=lambda x: x['capital']['total_return_pct'])
            param_analysis['best_ema'] = {
                'config': best_ema['config_name'],
                'return': best_ema['capital']['total_return_pct'],
                'fast': best_ema['config']['trend_ema_fast'],
                'slow': best_ema['config']['trend_ema_slow']
            }
        
        # Analyze thresholds
        threshold_results = [r for r in results if 'threshold_' in r['config_name']]
        if threshold_results:
            best_threshold = max(threshold_results, key=lambda x: x['capital']['total_return_pct'])
            param_analysis['best_threshold'] = {
                'config': best_threshold['config_name'],
                'return': best_threshold['capital']['total_return_pct'],
                'value': best_threshold['config']['trend_strength_threshold']
            }
        
        # Analyze RSI periods
        rsi_results = [r for r in results if 'rsi_' in r['config_name']]
        if rsi_results:
            best_rsi = max(rsi_results, key=lambda x: x['capital']['total_return_pct'])
            param_analysis['best_rsi'] = {
                'config': best_rsi['config_name'],
                'return': best_rsi['capital']['total_return_pct'],
                'period': best_rsi['config']['rsi_period']
            }
        
        # Analyze risk/reward ratios
        rr_results = [r for r in results if 'rr_' in r['config_name']]
        if rr_results:
            best_rr = max(rr_results, key=lambda x: x['capital']['total_return_pct'])
            param_analysis['best_risk_reward'] = {
                'config': best_rr['config_name'],
                'return': best_rr['capital']['total_return_pct'],
                'ratio': best_rr['config']['take_profit_risk_reward']
            }
        
        analysis = {
            'total_tests': len(results),
            'best_overall': by_return[0],
            'best_by_metric': {
                'return': by_return[0],
                'sharpe': by_sharpe[0],
                'win_rate': by_win_rate[0],
                'trade_count': by_trades_count[0],
                'return_per_trade': by_return_per_trade[0],
                'profit_factor': by_profit_factor[0] if by_profit_factor else None
            },
            'top_10': by_return[:10],
            'parameter_analysis': param_analysis
        }
        
        return analysis
    
    def print_results(self, analysis: Dict):
        """Print optimization results"""
        
        print("\n" + "="*80)
        print("🎯 FAST MOMENTUM OPTIMIZATION RESULTS")
        print("="*80)
        
        best = analysis['best_overall']
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Name: {best['config_name']}")
        print(f"   Total Return: {best['capital']['total_return_pct']:.2f}%")
        print(f"   Total Trades: {best['trades']['total']}")
        print(f"   Win Rate: {best['trades']['win_rate_pct']:.1f}%")
        print(f"   Profit Factor: {best['trades']['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {best['risk_metrics']['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {best['risk_metrics']['max_drawdown_pct']:.2f}%")
        
        print(f"\n📊 BEST BY DIFFERENT METRICS:")
        metrics = analysis['best_by_metric']
        print(f"   Best Return: {metrics['return']['config_name']} ({metrics['return']['capital']['total_return_pct']:.2f}%)")
        print(f"   Best Sharpe: {metrics['sharpe']['config_name']} ({metrics['sharpe']['risk_metrics']['sharpe_ratio']:.2f})")
        print(f"   Best Win Rate: {metrics['win_rate']['config_name']} ({metrics['win_rate']['trades']['win_rate_pct']:.1f}%)")
        print(f"   Most Trades: {metrics['trade_count']['config_name']} ({metrics['trade_count']['trades']['total']} trades)")
        
        if metrics['profit_factor']:
            print(f"   Best Profit Factor: {metrics['profit_factor']['config_name']} ({metrics['profit_factor']['trades']['profit_factor']:.2f})")
        
        print(f"\n🔝 TOP 10 CONFIGURATIONS:")
        for i, config in enumerate(analysis['top_10'], 1):
            print(f"   #{i:2d}: {config['config_name']:20s} - {config['capital']['total_return_pct']:6.2f}% "
                  f"({config['trades']['total']:2d} trades, {config['trades']['win_rate_pct']:4.1f}% win)")
        
        # Parameter analysis
        param_analysis = analysis['parameter_analysis']
        if param_analysis:
            print(f"\n🔧 PARAMETER ANALYSIS:")
            
            if 'best_ema' in param_analysis:
                ema = param_analysis['best_ema']
                print(f"   Best EMA: {ema['fast']}/{ema['slow']} ({ema['return']:.2f}% return)")
            
            if 'best_threshold' in param_analysis:
                threshold = param_analysis['best_threshold']
                print(f"   Best Threshold: {threshold['value']} ({threshold['return']:.2f}% return)")
            
            if 'best_rsi' in param_analysis:
                rsi = param_analysis['best_rsi']
                print(f"   Best RSI Period: {rsi['period']} ({rsi['return']:.2f}% return)")
            
            if 'best_risk_reward' in param_analysis:
                rr = param_analysis['best_risk_reward']
                print(f"   Best Risk/Reward: {rr['ratio']} ({rr['return']:.2f}% return)")
        
        # Optimal configuration
        print(f"\n⚙️  OPTIMAL CONFIGURATION PARAMETERS:")
        best_config = best['config']
        for param, value in best_config.items():
            if param != 'name':
                print(f"   {param}: {value}")
        
        print("\n" + "="*80)
        print("✅ FAST OPTIMIZATION COMPLETE!")
        print("="*80)
        
        return best_config
    
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

async def main():
    """Run fast optimization"""
    
    optimizer = FastMomentumOptimizer()
    
    start_time = time.time()
    analysis = await optimizer.run_optimization()
    total_time = time.time() - start_time
    
    if 'error' in analysis:
        print(f"❌ Optimization failed: {analysis['error']}")
        return
    
    # Print results and get best config
    best_config = optimizer.print_results(analysis)
    
    print(f"\n⏱️  Total optimization time: {total_time:.1f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"fast_momentum_optimization_{timestamp}.json"
    
    # Convert for JSON
    json_analysis = optimizer._convert_for_json(analysis)
    with open(results_file, 'w') as f:
        json.dump(json_analysis, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {results_file}")
    
    return best_config

if __name__ == "__main__":
    best_config = asyncio.run(main())