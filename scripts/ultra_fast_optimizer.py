#!/usr/bin/env python3
"""
Ultra-Fast Momentum Optimization - Quick Results
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

# Add src to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.momentum_trading_strategy import MomentumTradingStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig

# Disable debug logging for speed
import logging
logging.getLogger().setLevel(logging.WARNING)

class UltraFastOptimizer:
    """Ultra-fast momentum optimizer with minimal logging"""
    
    def generate_minimal_data(self) -> Dict[str, pd.DataFrame]:
        """Generate minimal test data for quick results"""
        periods = 360  # 15 days hourly data
        data_points = []
        base_price = 50000
        current_price = base_price
        
        # Simple strong uptrend for testing
        for i in range(periods):
            timestamp = datetime.now() - timedelta(hours=periods-i)
            
            # Strong consistent trend with some noise
            trend = 0.002  # 0.2% per hour
            noise = np.random.normal(0, 0.005)
            price_change = trend + noise
            current_price *= (1 + price_change)
            
            # Simple OHLCV
            open_price = current_price * 0.999
            high_price = current_price * 1.003
            low_price = current_price * 0.997
            close_price = current_price
            volume = 3000 + np.random.uniform(-500, 1500)
            
            data_points.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return {'BTCUSDT': pd.DataFrame(data_points)}
    
    def get_key_configurations(self) -> List[Dict]:
        """Get key configurations to test"""
        return [
            # Current default
            {
                'name': 'current_default',
                'trend_ema_fast': 5, 'trend_ema_slow': 10, 'trend_strength_threshold': 0.001,
                'rsi_period': 7, 'volume_confirmation_required': False, 'momentum_alignment_required': False,
                'breakout_lookback': 5, 'base_position_size': 0.05, 'stop_loss_atr_multiplier': 5.0,
                'take_profit_risk_reward': 1.5
            },
            # Ultra fast
            {
                'name': 'ultra_fast',
                'trend_ema_fast': 3, 'trend_ema_slow': 8, 'trend_strength_threshold': 0.0005,
                'rsi_period': 5, 'volume_confirmation_required': False, 'momentum_alignment_required': False,
                'breakout_lookback': 3, 'base_position_size': 0.07, 'stop_loss_atr_multiplier': 3.0,
                'take_profit_risk_reward': 1.2
            },
            # Balanced
            {
                'name': 'balanced',
                'trend_ema_fast': 8, 'trend_ema_slow': 21, 'trend_strength_threshold': 0.002,
                'rsi_period': 14, 'volume_confirmation_required': True, 'momentum_alignment_required': False,
                'breakout_lookback': 10, 'base_position_size': 0.03, 'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 2.5
            },
            # High RR
            {
                'name': 'high_rr',
                'trend_ema_fast': 5, 'trend_ema_slow': 12, 'trend_strength_threshold': 0.001,
                'rsi_period': 7, 'volume_confirmation_required': False, 'momentum_alignment_required': False,
                'breakout_lookback': 5, 'base_position_size': 0.04, 'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 3.0
            },
            # Conservative
            {
                'name': 'conservative',
                'trend_ema_fast': 12, 'trend_ema_slow': 26, 'trend_strength_threshold': 0.005,
                'rsi_period': 14, 'volume_confirmation_required': True, 'momentum_alignment_required': True,
                'breakout_lookback': 20, 'base_position_size': 0.02, 'stop_loss_atr_multiplier': 2.0,
                'take_profit_risk_reward': 3.0
            }
        ]
    
    async def test_config(self, config: Dict, data: Dict) -> Dict:
        """Test single config quickly"""
        try:
            full_config = {'symbols': ['BTCUSDT'], **{k: v for k, v in config.items() if k != 'name'}}
            full_config.update({
                'volume_surge_multiplier': 1.1,
                'max_position_size': full_config['base_position_size'] * 2,
                'trend_strength_scaling': False,
                'max_risk_per_trade': 0.02
            })
            
            strategy = MomentumTradingStrategy(full_config)
            engine = BacktestEngine(BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=config['base_position_size']
            ))
            
            timestamps = pd.to_datetime(data['BTCUSDT']['timestamp'], unit='ms')
            result = await engine.run_backtest(strategy, data, timestamps.min(), timestamps.max())
            
            return {
                'name': config['name'],
                'config': config,
                'return_pct': result['capital']['total_return_pct'],
                'trades': result['trades']['total'],
                'win_rate': result['trades']['win_rate_pct'],
                'sharpe': result['risk_metrics']['sharpe_ratio'],
                'max_dd': result['risk_metrics']['max_drawdown_pct'],
                'profit_factor': result['trades']['profit_factor']
            }
        except Exception as e:
            return {'name': config['name'], 'error': str(e)}
    
    async def run_quick_optimization(self):
        """Run ultra-fast optimization"""
        print("🚀 Ultra-Fast Momentum Optimization")
        
        # Generate minimal data
        data = self.generate_minimal_data() 
        end_price = data['BTCUSDT']['close'].iloc[-1]
        start_price = data['BTCUSDT']['close'].iloc[0]
        market_return = ((end_price / start_price) - 1) * 100
        
        print(f"   Market return: {market_return:.1f}% (15 days)")
        
        # Test configurations
        configs = self.get_key_configurations()
        print(f"   Testing {len(configs)} configurations...")
        
        results = []
        for config in configs:
            print(f"     {config['name']}...", end="")
            result = await self.test_config(config, data)
            
            if 'error' not in result:
                print(f" {result['return_pct']:.1f}% ({result['trades']} trades)")
                results.append(result)
            else:
                print(f" ERROR: {result['error']}")
        
        if not results:
            print("❌ No successful results")
            return
        
        # Find best
        best = max(results, key=lambda x: x['return_pct'])
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print(f"   Market Return: {market_return:.1f}%")
        print(f"   Best Strategy: {best['name']}")
        print(f"   Best Return: {best['return_pct']:.1f}%")
        print(f"   Trades: {best['trades']}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   Max DD: {best['max_dd']:.1f}%")
        
        print(f"\n📊 ALL RESULTS:")
        for r in sorted(results, key=lambda x: x['return_pct'], reverse=True):
            print(f"   {r['name']:15s}: {r['return_pct']:6.1f}% ({r['trades']:2d} trades, "
                  f"{r['win_rate']:4.1f}% win, {r['sharpe']:4.2f} sharpe)")
        
        print(f"\n⚙️  BEST CONFIGURATION:")
        for k, v in best['config'].items():
            if k != 'name':
                print(f"   {k}: {v}")
        
        return best['config']

async def main():
    optimizer = UltraFastOptimizer()
    start_time = time.time()
    
    best_config = await optimizer.run_quick_optimization()
    
    print(f"\n⏱️  Completed in {time.time() - start_time:.1f} seconds")
    return best_config

if __name__ == "__main__":
    best = asyncio.run(main())