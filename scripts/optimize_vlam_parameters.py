#!/usr/bin/env python3
"""
VLAM Strategy Parameter Optimization Script

This script performs systematic optimization of VLAM strategy parameters
to find the best performing combinations for maximum profitability.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import requests
import json
import time
from datetime import datetime, timedelta
from itertools import product
import pandas as pd

class VLAMParameterOptimizer:
    def __init__(self, base_url="http://localhost:8000/api/v1/backtesting"):
        self.base_url = base_url
        self.results = []
        
    def get_parameter_grid(self):
        """Define parameter grid for optimization"""
        return {
            'vlam_signal_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'consolidation_tolerance': [0.01, 0.02, 0.03, 0.04, 0.05],
            'spike_min_size': [0.8, 1.0, 1.2, 1.5, 2.0],
            'spike_volume_multiplier': [1.0, 1.2, 1.5, 2.0],
            'target_risk_reward': [1.5, 2.0, 2.5, 3.0],
            'entry_timeout_bars': [3, 5, 8, 10]
        }
    
    def get_base_parameters(self):
        """Get base parameter set"""
        return {
            "position_size": 0.02,
            "vlam_period": 10,
            "atr_period": 10,
            "volume_period": 15,
            "consolidation_min_length": 4,
            "consolidation_max_length": 20,
            "consolidation_tolerance": 0.03,
            "min_touches": 2,
            "spike_min_size": 1.0,
            "spike_volume_multiplier": 1.2,
            "vlam_signal_threshold": 0.3,
            "entry_timeout_bars": 8,
            "target_risk_reward": 2.0,
            "max_risk_per_trade": 0.02
        }
    
    async def run_backtest(self, parameters):
        """Run a single backtest with given parameters"""
        backtest_request = {
            "strategy_name": "vlam_consolidation_strategy",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "start_date": "2024-12-01T00:00:00",
            "end_date": "2025-01-15T00:00:00",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "parameters": parameters
        }
        
        try:
            # Start backtest
            response = requests.post(f"{self.base_url}/run", 
                                   json=backtest_request, 
                                   timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Failed to start backtest: {response.text}")
                return None
                
            session_data = response.json()
            session_id = session_data["session_id"]
            
            # Wait for completion
            max_wait = 60  # 60 seconds max
            for _ in range(max_wait):
                await asyncio.sleep(1)
                status_response = requests.get(f"{self.base_url}/status/{session_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status["status"] == "completed":
                        # Get results
                        results_response = requests.get(f"{self.base_url}/results/{session_id}")
                        if results_response.status_code == 200:
                            return results_response.json()
                        else:
                            print(f"❌ Failed to get results: {results_response.text}")
                            return None
                    elif status["status"] == "failed":
                        print(f"❌ Backtest failed: {status.get('message', 'Unknown error')}")
                        return None
            
            print(f"❌ Backtest timed out after {max_wait} seconds")
            return None
            
        except Exception as e:
            print(f"❌ Error running backtest: {e}")
            return None
    
    def evaluate_results(self, results):
        """Evaluate backtest results and return key metrics"""
        if not results:
            return None
            
        try:
            # Extract key metrics
            total_return = results.get('capital', {}).get('total_return_pct', -100)
            sharpe_ratio = results.get('risk_metrics', {}).get('sharpe_ratio', -10)
            max_drawdown = results.get('risk_metrics', {}).get('max_drawdown_pct', 100)
            
            trades_data = results.get('trades', {})
            total_trades = trades_data.get('total', 0)
            win_rate = trades_data.get('win_rate_pct', 0)
            avg_win = trades_data.get('avg_win', 0)
            avg_loss = trades_data.get('avg_loss', 0)
            profit_factor = trades_data.get('profit_factor', 0)
            
            # Calculate composite score
            # Prioritize: positive return, high Sharpe, low drawdown, reasonable trade count
            score = 0
            
            # Return component (40% weight)
            if total_return > 0:
                score += total_return * 0.4
            else:
                score += total_return * 0.8  # Penalize losses more
            
            # Sharpe ratio component (30% weight)
            if sharpe_ratio > 0:
                score += min(sharpe_ratio * 5, 15)  # Cap at 15 points
            else:
                score += sharpe_ratio * 3  # Penalize negative Sharpe
            
            # Drawdown component (20% weight) - penalty
            score -= max_drawdown * 0.5
            
            # Trade frequency component (10% weight)
            if total_trades >= 3:
                score += min(total_trades * 0.5, 5)  # Cap at 5 points
            elif total_trades > 0:
                score += total_trades * 0.2
            else:
                score -= 10  # Heavy penalty for no trades
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'score': score
            }
            
        except Exception as e:
            print(f"❌ Error evaluating results: {e}")
            return None
    
    async def optimize_single_parameter(self, param_name, values, fixed_params=None):
        """Optimize a single parameter while keeping others fixed"""
        print(f"\n🔬 OPTIMIZING PARAMETER: {param_name}")
        print("=" * 60)
        
        if fixed_params is None:
            fixed_params = self.get_base_parameters()
        
        results = []
        
        for i, value in enumerate(values):
            print(f"\n📊 Testing {param_name} = {value} ({i+1}/{len(values)})")
            
            # Create parameter set
            test_params = fixed_params.copy()
            test_params[param_name] = value
            
            # Run backtest
            backtest_results = await self.run_backtest(test_params)
            metrics = self.evaluate_results(backtest_results)
            
            if metrics:
                metrics['parameter_value'] = value
                metrics['parameters'] = test_params.copy()
                results.append(metrics)
                
                print(f"   Return: {metrics['total_return']:.2f}%, "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                      f"Trades: {metrics['total_trades']}, "
                      f"Score: {metrics['score']:.2f}")
            else:
                print(f"   ❌ Failed to get valid results")
        
        # Find best result
        if results:
            best = max(results, key=lambda x: x['score'])
            print(f"\n🏆 BEST {param_name.upper()}: {best['parameter_value']}")
            print(f"   Score: {best['score']:.2f}")
            print(f"   Return: {best['total_return']:.2f}%")
            print(f"   Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"   Trades: {best['total_trades']}")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            
            return best['parameter_value'], results
        else:
            print(f"❌ No valid results for {param_name}")
            return None, []
    
    async def optimize_sequential(self):
        """Sequential parameter optimization - optimize one parameter at a time"""
        print("🚀 STARTING SEQUENTIAL PARAMETER OPTIMIZATION")
        print("=" * 80)
        
        param_grid = self.get_parameter_grid()
        best_params = self.get_base_parameters()
        optimization_results = {}
        
        # Order of optimization (most impactful first)
        optimization_order = [
            'vlam_signal_threshold',
            'spike_min_size', 
            'consolidation_tolerance',
            'target_risk_reward',
            'spike_volume_multiplier',
            'entry_timeout_bars'
        ]
        
        for param_name in optimization_order:
            if param_name in param_grid:
                print(f"\n" + "="*80)
                print(f"OPTIMIZING: {param_name.upper()}")
                print("="*80)
                
                best_value, results = await self.optimize_single_parameter(
                    param_name, 
                    param_grid[param_name], 
                    best_params
                )
                
                if best_value is not None:
                    best_params[param_name] = best_value
                    optimization_results[param_name] = {
                        'best_value': best_value,
                        'results': results
                    }
                    
                    print(f"\n✅ Updated {param_name} to {best_value}")
                    print(f"📋 Current best parameters: {json.dumps(best_params, indent=2)}")
        
        return best_params, optimization_results
    
    async def run_final_validation(self, optimized_params):
        """Run final validation backtest with optimized parameters"""
        print("\n" + "="*80)
        print("🎯 FINAL VALIDATION WITH OPTIMIZED PARAMETERS")
        print("="*80)
        
        print(f"📋 Optimized Parameters:")
        for key, value in optimized_params.items():
            print(f"   {key}: {value}")
        
        # Run extended backtest
        validation_params = optimized_params.copy()
        
        # Test on longer period
        backtest_request = {
            "strategy_name": "vlam_consolidation_strategy",
            "symbol": "BTCUSDT", 
            "timeframe": "1h",
            "start_date": "2024-11-01T00:00:00",  # Longer period
            "end_date": "2025-01-15T00:00:00",
            "initial_capital": 10000.0,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "parameters": validation_params
        }
        
        print("\n📊 Running extended validation backtest...")
        
        try:
            response = requests.post(f"{self.base_url}/run", json=backtest_request, timeout=10)
            if response.status_code != 200:
                print(f"❌ Failed to start validation: {response.text}")
                return None
                
            session_data = response.json()
            session_id = session_data["session_id"]
            
            # Wait for completion
            for _ in range(120):  # 2 minutes max for longer backtest
                await asyncio.sleep(1)
                status_response = requests.get(f"{self.base_url}/status/{session_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status["status"] == "completed":
                        results_response = requests.get(f"{self.base_url}/results/{session_id}")
                        if results_response.status_code == 200:
                            results = results_response.json()
                            metrics = self.evaluate_results(results)
                            
                            print("\n🏆 FINAL VALIDATION RESULTS:")
                            print("-" * 40)
                            print(f"📈 Total Return: {metrics['total_return']:.2f}%")
                            print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                            print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2f}%")
                            print(f"🎯 Total Trades: {metrics['total_trades']}")
                            print(f"✅ Win Rate: {metrics['win_rate']:.1f}%")
                            print(f"💰 Profit Factor: {metrics['profit_factor']:.3f}")
                            print(f"🏅 Overall Score: {metrics['score']:.2f}")
                            
                            return results, metrics
                            
                    elif status["status"] == "failed":
                        print(f"❌ Validation failed: {status.get('message')}")
                        return None
            
            print("❌ Validation timed out")
            return None
            
        except Exception as e:
            print(f"❌ Error in validation: {e}")
            return None

async def main():
    """Main optimization function"""
    optimizer = VLAMParameterOptimizer()
    
    print("🔬 VLAM STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("This will systematically optimize parameters to improve strategy performance.")
    print("Expected runtime: 15-30 minutes depending on API response times.")
    
    # Run sequential optimization
    optimized_params, optimization_results = await optimizer.optimize_sequential()
    
    # Run final validation
    validation_results, validation_metrics = await optimizer.run_final_validation(optimized_params)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"vlam_optimization_results_{timestamp}.json"
    
    final_results = {
        'timestamp': timestamp,
        'optimized_parameters': optimized_params,
        'optimization_process': optimization_results,
        'validation_results': validation_metrics,
        'base_parameters': optimizer.get_base_parameters()
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎉 OPTIMIZATION COMPLETE!")
    
    return optimized_params, validation_metrics

if __name__ == "__main__":
    asyncio.run(main())