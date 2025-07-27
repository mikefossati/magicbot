#!/usr/bin/env python3
"""
Direct VLAM Parameter Optimization - Bypassing API for Speed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from itertools import product

from src.strategies.vlam_consolidation_strategy import VLAMConsolidationStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.exchanges.binance_exchange import BinanceBacktestingExchange
from src.data.historical_manager import HistoricalDataManager

class DirectVLAMOptimizer:
    def __init__(self):
        self.exchange = None
        self.data_manager = None
        self.historical_data = None
        
    async def initialize(self):
        """Initialize exchange and load historical data once"""
        print("📊 Initializing data connection...")
        
        self.exchange = BinanceBacktestingExchange()
        await self.exchange.connect()
        self.data_manager = HistoricalDataManager(self.exchange)
        
        # Load historical data once for all tests
        end_date = datetime(2025, 1, 15)
        start_date = datetime(2024, 12, 1)
        
        print(f"📈 Loading historical data: {start_date.date()} to {end_date.date()}")
        
        self.historical_data = await self.data_manager.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        if self.historical_data is None or self.historical_data.empty:
            raise Exception("Failed to load historical data")
            
        print(f"✅ Loaded {len(self.historical_data)} bars of data")
        
    async def cleanup(self):
        """Cleanup connections"""
        if self.exchange:
            await self.exchange.disconnect()
    
    def test_parameters(self, params):
        """Test a single parameter set directly"""
        try:
            # Create strategy instance
            strategy_config = params.copy()
            strategy_config['symbols'] = ['BTCUSDT']
            strategy = VLAMConsolidationStrategy(strategy_config)
            
            # Create backtest engine
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=params.get('position_size', 0.02)
            )
            
            engine = BacktestEngine(backtest_config)
            
            # Format data for engine
            formatted_data = {'BTCUSDT': self.historical_data}
            
            # Run synchronous backtest (engine.run_backtest is async but we'll handle it)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(engine.run_backtest(
                strategy=strategy,
                historical_data=formatted_data,
                start_date=datetime(2024, 12, 1),
                end_date=datetime(2025, 1, 15)
            ))
            
            loop.close()
            
            # Extract metrics
            total_return = results.get('total_return_pct', 0)
            total_trades = len(results.get('trades_detail', []))
            
            if total_trades > 0:
                winning_trades = [t for t in results.get('trades_detail', []) if t.pnl > 0]
                win_rate = (len(winning_trades) / total_trades) * 100
                avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
                losing_trades = [t for t in results.get('trades_detail', []) if t.pnl <= 0]
                avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
                profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            max_drawdown = results.get('max_drawdown_pct', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            
            # Calculate composite score
            score = self.calculate_score(total_return, sharpe_ratio, max_drawdown, total_trades, win_rate, profit_factor)
            
            return {
                'params': params,
                'total_return': total_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'score': score
            }
            
        except Exception as e:
            print(f"❌ Error testing parameters: {e}")
            return None
    
    def calculate_score(self, total_return, sharpe_ratio, max_drawdown, total_trades, win_rate, profit_factor):
        """Calculate composite score for parameter set"""
        score = 0
        
        # Return component (40% weight)
        if total_return > 0:
            score += total_return * 2.0
        else:
            score += total_return * 4.0  # Penalize losses more heavily
        
        # Sharpe ratio component (25% weight)
        if sharpe_ratio > 0:
            score += min(sharpe_ratio * 10, 25)  # Cap at 25 points
        else:
            score += sharpe_ratio * 5
        
        # Trade frequency component (15% weight)
        if total_trades >= 5:
            score += 15
        elif total_trades >= 3:
            score += 10
        elif total_trades >= 1:
            score += 5
        else:
            score -= 20  # Heavy penalty for no trades
        
        # Win rate component (10% weight)
        if total_trades > 0:
            score += (win_rate - 50) * 0.2  # Bonus/penalty relative to 50%
        
        # Profit factor component (10% weight)
        if profit_factor > 1:
            score += min((profit_factor - 1) * 10, 10)
        elif profit_factor > 0:
            score += (profit_factor - 1) * 20  # Penalty for PF < 1
        
        # Drawdown penalty
        score -= max_drawdown * 0.5
        
        return score
    
    async def optimize_grid_search(self):
        """Run grid search optimization on key parameters"""
        print("\n🔬 STARTING DIRECT PARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Define parameter grid - focusing on most impactful parameters
        param_grid = {
            'vlam_signal_threshold': [0.2, 0.3, 0.4, 0.5],
            'spike_min_size': [0.8, 1.0, 1.2, 1.5],
            'consolidation_tolerance': [0.02, 0.03, 0.04, 0.05],
            'target_risk_reward': [1.5, 2.0, 2.5, 3.0],
            'spike_volume_multiplier': [1.0, 1.2, 1.5],
            'entry_timeout_bars': [5, 8, 10]
        }
        
        # Base parameters
        base_params = {
            'position_size': 0.02,
            'vlam_period': 10,
            'atr_period': 10,
            'volume_period': 15,
            'consolidation_min_length': 4,
            'consolidation_max_length': 20,
            'min_touches': 2,
            'max_risk_per_trade': 0.02
        }
        
        # Generate all combinations (this will be manageable - ~1440 combinations)
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = np.prod([len(v) for v in param_values])
        print(f"📊 Testing {total_combinations} parameter combinations...")
        
        results = []
        
        for i, combination in enumerate(product(*param_values)):
            if i % 50 == 0:  # Progress update every 50 tests
                print(f"🧪 Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
            
            # Create parameter set
            test_params = base_params.copy()
            for param_name, param_value in zip(param_names, combination):
                test_params[param_name] = param_value
            
            # Test parameters
            result = self.test_parameters(test_params)
            if result:
                results.append(result)
                
                # Print top performers
                if result['score'] > 10:  # Only print good results
                    print(f"   🎯 Good result: Score={result['score']:.1f}, Return={result['total_return']:.2f}%, Trades={result['total_trades']}")
        
        return results
    
    async def optimize_sequential(self):
        """Sequential optimization - optimize one parameter at a time"""
        print("\n🎯 STARTING SEQUENTIAL OPTIMIZATION")
        print("=" * 60)
        
        # Base configuration
        best_params = {
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
        
        # Parameter ranges to test
        param_ranges = {
            'vlam_signal_threshold': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
            'spike_min_size': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
            'consolidation_tolerance': [0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
            'target_risk_reward': [1.2, 1.5, 2.0, 2.5, 3.0],
            'spike_volume_multiplier': [1.0, 1.1, 1.2, 1.5, 2.0],
            'entry_timeout_bars': [3, 5, 8, 10, 12]
        }
        
        optimization_results = {}
        
        for param_name, param_values in param_ranges.items():
            print(f"\n🔬 Optimizing {param_name}...")
            
            best_score = -999
            best_value = best_params[param_name]
            param_results = []
            
            for value in param_values:
                test_params = best_params.copy()
                test_params[param_name] = value
                
                result = self.test_parameters(test_params)
                if result:
                    param_results.append(result)
                    print(f"   {param_name}={value}: Score={result['score']:.1f}, Return={result['total_return']:.2f}%, Trades={result['total_trades']}")
                    
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_value = value
            
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

async def main():
    optimizer = DirectVLAMOptimizer()
    
    try:
        await optimizer.initialize()
        
        # Run sequential optimization (faster and often better)
        print("🚀 Running sequential parameter optimization...")
        best_params, optimization_results = await optimizer.optimize_sequential()
        
        print("\n" + "="*80)
        print("🏆 OPTIMIZATION COMPLETE!")
        print("="*80)
        
        print(f"\n📋 BEST PARAMETERS:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        # Test final configuration
        print(f"\n🎯 Testing final optimized configuration...")
        final_result = optimizer.test_parameters(best_params)
        
        if final_result:
            print(f"\n🏅 FINAL RESULTS:")
            print(f"   Total Return: {final_result['total_return']:.2f}%")
            print(f"   Total Trades: {final_result['total_trades']}")
            print(f"   Win Rate: {final_result['win_rate']:.1f}%")
            print(f"   Profit Factor: {final_result['profit_factor']:.2f}")
            print(f"   Max Drawdown: {final_result['max_drawdown']:.2f}%")
            print(f"   Sharpe Ratio: {final_result['sharpe_ratio']:.2f}")
            print(f"   Overall Score: {final_result['score']:.2f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"direct_vlam_optimization_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_parameters': best_params,
                'optimization_results': optimization_results,
                'final_test': final_result
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    finally:
        await optimizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())