#!/usr/bin/env python3
"""
Quick VLAM Parameter Optimization - Focus on Key Parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import requests
import json
import time
from datetime import datetime

async def test_parameter_set(params, description):
    """Test a single parameter set"""
    print(f"\n🧪 Testing: {description}")
    
    backtest_request = {
        "strategy_name": "vlam_consolidation_strategy",
        "symbol": "BTCUSDT",
        "timeframe": "1h", 
        "start_date": "2024-12-15T00:00:00",  # Shorter period for speed
        "end_date": "2025-01-15T00:00:00",
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "parameters": params
    }
    
    try:
        # Start backtest
        response = requests.post("http://localhost:8000/api/v1/backtesting/run", 
                               json=backtest_request, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start: {response.text}")
            return None
            
        session_id = response.json()["session_id"]
        
        # Wait for completion
        for _ in range(30):  # 30 second timeout
            await asyncio.sleep(1)
            status_response = requests.get(f"http://localhost:8000/api/v1/backtesting/status/{session_id}")
            
            if status_response.status_code == 200:
                status = status_response.json()
                if status["status"] == "completed":
                    # Get results
                    results_response = requests.get(f"http://localhost:8000/api/v1/backtesting/results/{session_id}")
                    if results_response.status_code == 200:
                        results = results_response.json()
                        
                        # Extract key metrics
                        total_return = results.get('capital', {}).get('total_return_pct', -100)
                        trades = results.get('trades', {}).get('total', 0)
                        win_rate = results.get('trades', {}).get('win_rate_pct', 0)
                        sharpe = results.get('risk_metrics', {}).get('sharpe_ratio', -10)
                        
                        print(f"   📊 Return: {total_return:.2f}%, Trades: {trades}, Win Rate: {win_rate:.1f}%, Sharpe: {sharpe:.2f}")
                        
                        return {
                            'params': params,
                            'description': description,
                            'return': total_return,
                            'trades': trades,
                            'win_rate': win_rate,
                            'sharpe': sharpe,
                            'score': total_return + (sharpe * 2) + (trades * 0.5) - (50 - win_rate) * 0.1
                        }
                        
                elif status["status"] == "failed":
                    print(f"   ❌ Failed: {status.get('message')}")
                    return None
        
        print(f"   ⏱️ Timed out")
        return None
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

async def main():
    """Quick optimization focusing on key parameters"""
    print("🚀 QUICK VLAM PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    base_params = {
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
    
    test_configs = [
        # Original configuration
        (base_params.copy(), "Original Config"),
        
        # More sensitive VLAM threshold
        ({**base_params, "vlam_signal_threshold": 0.2}, "Lower VLAM Threshold (0.2)"),
        ({**base_params, "vlam_signal_threshold": 0.4}, "Higher VLAM Threshold (0.4)"),
        
        # Spike detection variations
        ({**base_params, "spike_min_size": 0.8}, "Lower Spike Size (0.8)"),
        ({**base_params, "spike_min_size": 1.5}, "Higher Spike Size (1.5)"),
        
        # Consolidation tolerance
        ({**base_params, "consolidation_tolerance": 0.02}, "Tighter Consolidation (2%)"),
        ({**base_params, "consolidation_tolerance": 0.04}, "Looser Consolidation (4%)"),
        
        # Risk-reward ratios
        ({**base_params, "target_risk_reward": 1.5}, "Lower Risk:Reward (1.5:1)"),
        ({**base_params, "target_risk_reward": 3.0}, "Higher Risk:Reward (3:1)"),
        
        # Combined optimizations
        ({**base_params, "vlam_signal_threshold": 0.2, "spike_min_size": 0.8, "consolidation_tolerance": 0.04}, 
         "More Sensitive Config"),
        
        ({**base_params, "vlam_signal_threshold": 0.4, "spike_min_size": 1.5, "target_risk_reward": 3.0}, 
         "More Selective Config"),
        
        # Volume-based variation
        ({**base_params, "spike_volume_multiplier": 1.0, "vlam_signal_threshold": 0.25}, 
         "Lower Volume Requirements"),
    ]
    
    results = []
    
    for params, description in test_configs:
        result = await test_parameter_set(params, description)
        if result:
            results.append(result)
    
    # Find best configurations
    if results:
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n📊 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. {result['description']}")
            print(f"   Return: {result['return']:.2f}%")
            print(f"   Trades: {result['trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Sharpe: {result['sharpe']:.2f}")
            print(f"   Score: {result['score']:.2f}")
        
        # Save best configuration
        best_config = results[0]
        print(f"\n🎯 BEST CONFIGURATION: {best_config['description']}")
        print("\n📋 Best Parameters:")
        for key, value in best_config['params'].items():
            if value != base_params[key]:
                print(f"   ✨ {key}: {value} (changed from {base_params[key]})")
            else:
                print(f"     {key}: {value}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"quick_vlam_optimization_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_config': best_config,
                'all_results': results,
                'base_params': base_params
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return best_config['params']
    
    else:
        print("❌ No valid results obtained")
        return base_params

if __name__ == "__main__":
    asyncio.run(main())