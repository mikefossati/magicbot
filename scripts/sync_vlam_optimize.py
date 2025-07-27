#!/usr/bin/env python3
"""
Synchronous VLAM Parameter Optimization - Direct Strategy Testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio

from src.strategies.vlam_consolidation_strategy import VLAMConsolidationStrategy
from src.exchanges.binance_exchange import BinanceBacktestingExchange
from src.data.historical_manager import HistoricalDataManager

class SyncVLAMOptimizer:
    def __init__(self):
        self.historical_data = None
        
    async def load_data(self):
        """Load historical data once"""
        print("📊 Loading historical data...")
        
        exchange = BinanceBacktestingExchange()
        await exchange.connect()
        data_manager = HistoricalDataManager(exchange)
        
        # Load data for optimization period
        end_date = datetime(2025, 1, 15)
        start_date = datetime(2024, 12, 1)
        
        self.historical_data = await data_manager.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        await exchange.disconnect()
        
        if self.historical_data is None or self.historical_data.empty:
            raise Exception("Failed to load historical data")
            
        print(f"✅ Loaded {len(self.historical_data)} bars of data")
        print(f"   Period: {self.historical_data.index[0]} to {self.historical_data.index[-1]}")
        print(f"   Price range: ${self.historical_data['low'].min():.0f} - ${self.historical_data['high'].max():.0f}")
    
    def test_parameters_sync(self, params):
        """Test parameters synchronously using strategy signal generation"""
        try:
            # Create strategy
            strategy_config = params.copy()
            strategy_config['symbols'] = ['BTCUSDT']
            strategy = VLAMConsolidationStrategy(strategy_config)
            
            # Simple signal-based evaluation
            signals = []
            total_bars = len(self.historical_data)
            
            # Test on sliding windows to simulate live trading
            window_size = 50  # 50-hour windows
            step_size = 10    # Move by 10 hours each time
            
            for start_idx in range(0, total_bars - window_size, step_size):
                end_idx = start_idx + window_size
                window_data = self.historical_data.iloc[start_idx:end_idx].copy()
                
                # Reset index for each window
                window_data.reset_index(drop=True, inplace=True)
                
                try:
                    # Test if strategy would generate a signal
                    indicators = strategy._calculate_indicators(window_data)
                    consolidation = strategy._detect_consolidation(window_data, indicators)
                    
                    if consolidation:
                        spike_event = strategy._detect_spike(window_data, indicators, consolidation)
                        
                        if spike_event:
                            entry_signal = strategy._check_vlam_entry_signal(window_data, indicators, consolidation, spike_event)
                            
                            if entry_signal:
                                # Calculate signal quality
                                signal_price = window_data['close'].iloc[-1]
                                
                                if entry_signal['action'] == 'BUY':
                                    stop_loss = spike_event['low']
                                    risk = signal_price - stop_loss
                                    target = signal_price + (risk * params.get('target_risk_reward', 2.0))
                                else:  # SELL
                                    stop_loss = spike_event['high']
                                    risk = stop_loss - signal_price
                                    target = signal_price - (risk * params.get('target_risk_reward', 2.0))
                                
                                signals.append({
                                    'price': signal_price,
                                    'action': entry_signal['action'],
                                    'stop_loss': stop_loss,
                                    'target': target,
                                    'risk': risk,
                                    'confidence': entry_signal['strength'],
                                    'window_start': start_idx
                                })
                                
                except Exception as e:
                    # Skip windows with errors
                    continue
            
            # Evaluate signals
            total_signals = len(signals)
            
            if total_signals == 0:
                return {
                    'params': params,
                    'total_signals': 0,
                    'estimated_return': 0,
                    'avg_confidence': 0,
                    'score': -20  # Heavy penalty for no signals
                }
            
            # Calculate estimated performance
            avg_confidence = np.mean([s['confidence'] for s in signals])
            avg_risk = np.mean([abs(s['risk']) for s in signals])
            
            # Estimate win rate based on confidence
            estimated_win_rate = min(avg_confidence * 60 + 40, 85)  # 40-85% range
            
            # Estimate returns
            avg_win = avg_risk * params.get('target_risk_reward', 2.0)
            avg_loss = avg_risk
            
            estimated_return_per_trade = (estimated_win_rate/100 * avg_win) - ((100-estimated_win_rate)/100 * avg_loss)
            estimated_total_return = estimated_return_per_trade * total_signals * 0.02 / 10000 * 100  # Rough estimate
            
            # Calculate score
            score = self.calculate_score(estimated_total_return, total_signals, avg_confidence, estimated_win_rate)
            
            return {
                'params': params,
                'total_signals': total_signals,
                'estimated_return': estimated_total_return,
                'avg_confidence': avg_confidence,
                'estimated_win_rate': estimated_win_rate,
                'avg_risk': avg_risk,
                'score': score
            }
            
        except Exception as e:
            print(f"❌ Error testing parameters: {e}")
            return None
    
    def calculate_score(self, estimated_return, total_signals, avg_confidence, estimated_win_rate):
        """Calculate composite score"""
        score = 0
        
        # Return component (40%)
        score += estimated_return * 2.0
        
        # Signal frequency (25%)
        if total_signals >= 8:
            score += 25
        elif total_signals >= 5:
            score += 15
        elif total_signals >= 2:
            score += 8
        elif total_signals == 1:
            score += 2
        else:
            score -= 20
        
        # Confidence component (20%)
        score += avg_confidence * 20
        
        # Win rate component (15%)
        score += (estimated_win_rate - 50) * 0.15
        
        return score
    
    def optimize_parameters(self):
        """Run parameter optimization"""
        print("\n🔬 STARTING SYNCHRONOUS PARAMETER OPTIMIZATION")
        print("=" * 70)
        
        # Base parameters
        base_params = {
            'position_size': 0.02,
            'vlam_period': 10,
            'atr_period': 10,
            'volume_period': 15,
            'consolidation_min_length': 4,
            'consolidation_max_length': 20,
            'consolidation_tolerance': 0.03,
            'min_touches': 2,
            'spike_min_size': 1.0,
            'spike_volume_multiplier': 1.2,
            'vlam_signal_threshold': 0.3,
            'entry_timeout_bars': 8,
            'target_risk_reward': 2.0,
            'max_risk_per_trade': 0.02
        }
        
        # Parameter ranges
        param_ranges = {
            'vlam_signal_threshold': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
            'spike_min_size': [0.6, 0.8, 1.0, 1.2, 1.5],
            'consolidation_tolerance': [0.02, 0.025, 0.03, 0.04, 0.05],
            'target_risk_reward': [1.5, 2.0, 2.5, 3.0],
            'spike_volume_multiplier': [1.0, 1.2, 1.5, 2.0],
            'entry_timeout_bars': [5, 8, 10, 12]
        }
        
        best_params = base_params.copy()
        best_overall_score = -999
        optimization_results = {}
        
        # Test baseline
        print(f"\n📊 Testing baseline configuration...")
        baseline_result = self.test_parameters_sync(base_params)
        if baseline_result:
            print(f"   Baseline: {baseline_result['total_signals']} signals, Score: {baseline_result['score']:.1f}")
            best_overall_score = baseline_result['score']
        
        # Optimize each parameter
        for param_name, param_values in param_ranges.items():
            print(f"\n🔬 Optimizing {param_name}...")
            
            best_score = best_overall_score
            best_value = best_params[param_name]
            param_results = []
            
            for value in param_values:
                test_params = best_params.copy()
                test_params[param_name] = value
                
                result = self.test_parameters_sync(test_params)
                if result:
                    param_results.append(result)
                    print(f"   {param_name}={value}: {result['total_signals']} signals, "
                          f"Score={result['score']:.1f}, Est.Return={result['estimated_return']:.2f}%")
                    
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_value = value
                        best_overall_score = result['score']
            
            # Update best parameters
            if best_value != best_params[param_name]:
                print(f"   ✅ Updated {param_name}: {best_params[param_name]} → {best_value}")
                best_params[param_name] = best_value
            else:
                print(f"   ➡️ Keeping {param_name}: {best_value}")
            
            optimization_results[param_name] = {
                'best_value': best_value,
                'best_score': best_score,
                'results': param_results
            }
        
        return best_params, optimization_results
    
    def run_final_test(self, params):
        """Run final comprehensive test"""
        print(f"\n🎯 FINAL VALIDATION TEST")
        print("-" * 50)
        
        result = self.test_parameters_sync(params)
        if result:
            print(f"📊 Final Results:")
            print(f"   Total Signals: {result['total_signals']}")
            print(f"   Estimated Return: {result['estimated_return']:.2f}%")
            print(f"   Average Confidence: {result['avg_confidence']:.2f}")
            print(f"   Estimated Win Rate: {result['estimated_win_rate']:.1f}%")
            print(f"   Overall Score: {result['score']:.2f}")
            
            return result
        else:
            print("❌ Final test failed")
            return None

async def main():
    optimizer = SyncVLAMOptimizer()
    
    # Load data
    await optimizer.load_data()
    
    # Run optimization
    best_params, optimization_results = optimizer.optimize_parameters()
    
    print("\n" + "="*70)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("="*70)
    
    print(f"\n📋 OPTIMIZED PARAMETERS:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Final validation
    final_result = optimizer.run_final_test(best_params)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sync_vlam_optimization_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'optimized_parameters': best_params,
            'optimization_process': optimization_results,
            'final_validation': final_result,
            'base_parameters': {
                'position_size': 0.02,
                'vlam_period': 10,
                'atr_period': 10,
                'volume_period': 15,
                'consolidation_min_length': 4,
                'consolidation_max_length': 20,
                'consolidation_tolerance': 0.03,
                'min_touches': 2,
                'spike_min_size': 1.0,
                'spike_volume_multiplier': 1.2,
                'vlam_signal_threshold': 0.3,
                'entry_timeout_bars': 8,
                'target_risk_reward': 2.0,
                'max_risk_per_trade': 0.02
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return best_params, final_result

if __name__ == "__main__":
    asyncio.run(main())