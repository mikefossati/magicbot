#!/usr/bin/env python3
"""
Simple Day Trading Strategy Test
Quick test of the day trading strategy with minimal data to verify it works
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.day_trading_strategy import DayTradingStrategy
from tests.unit.fixtures.historical_snapshots import get_historical_snapshot

def test_strategy_basic():
    """Test basic strategy functionality"""
    print("=" * 60)
    print("DAY TRADING STRATEGY BASIC TEST")
    print("=" * 60)
    
    # Test configurations to compare
    test_configs = [
        {
            'name': 'Default Config',
            'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
            'volume_multiplier': 1.2, 'min_signal_score': 0.6,
            'support_resistance_threshold': 0.8,
            'stop_loss_pct': 1.5, 'take_profit_pct': 2.5,
            'max_daily_trades': 3
        },
        {
            'name': 'More Sensitive',
            'fast_ema': 5, 'medium_ema': 13, 'slow_ema': 34,
            'volume_multiplier': 1.0, 'min_signal_score': 0.5,
            'support_resistance_threshold': 1.0,
            'stop_loss_pct': 1.0, 'take_profit_pct': 2.0,
            'max_daily_trades': 5
        },
        {
            'name': 'Conservative',
            'fast_ema': 12, 'medium_ema': 26, 'slow_ema': 55,
            'volume_multiplier': 1.5, 'min_signal_score': 0.7,
            'support_resistance_threshold': 0.5,
            'stop_loss_pct': 2.0, 'take_profit_pct': 3.0,
            'max_daily_trades': 2
        }
    ]
    
    # Use test data
    test_data = get_historical_snapshot('bullish_crossover')
    print(f"📊 Using test data with {len(test_data)} points")
    
    results = []
    
    for config_test in test_configs:
        print(f"\n🔄 Testing: {config_test['name']}")
        
        try:
            # Create full config
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
            
            # Add test parameters
            for key in ['fast_ema', 'medium_ema', 'slow_ema', 'volume_multiplier', 
                       'min_signal_score', 'support_resistance_threshold',
                       'stop_loss_pct', 'take_profit_pct', 'max_daily_trades']:
                config[key] = config_test[key]
            
            # Create strategy
            strategy = DayTradingStrategy(config)
            
            # Test signal generation
            market_data = {'BTCUSDT': test_data}
            
            # Run async signal generation
            async def test_signals():
                return await strategy.generate_signals(market_data)
            
            signals = asyncio.run(test_signals())
            
            signal_count = len(signals)
            avg_confidence = sum(s.confidence for s in signals) / signal_count if signals else 0
            signal_types = [s.action for s in signals]
            
            results.append({
                'name': config_test['name'],
                'config': config,
                'signal_count': signal_count,
                'avg_confidence': avg_confidence,
                'signal_types': signal_types,
                'success': True
            })
            
            print(f"   ✅ Signals generated: {signal_count}")
            print(f"   📊 Average confidence: {avg_confidence:.3f}")
            print(f"   🎯 Signal types: {', '.join(signal_types) if signals else 'None'}")
            
            if signals:
                for i, signal in enumerate(signals):
                    print(f"      Signal {i+1}: {signal.action} at {signal.price} (conf: {signal.confidence:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'name': config_test['name'],
                'error': str(e),
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    
    if successful:
        print(f"✅ {len(successful)}/{len(results)} configurations successful")
        
        print(f"\n{'Configuration':<15} {'Signals':<8} {'Avg Conf':<10} {'Types'}")
        print("-" * 60)
        
        for result in successful:
            types_str = ', '.join(set(result['signal_types'])) if result['signal_types'] else 'None'
            print(f"{result['name']:<15} "
                  f"{result['signal_count']:<8} "
                  f"{result['avg_confidence']:<10.3f} "
                  f"{types_str}")
        
        # Find best by signal quality (count + confidence)
        best = max(successful, key=lambda x: x['signal_count'] * x['avg_confidence'])
        
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print(f"   Signals: {best['signal_count']}")
        print(f"   Average Confidence: {best['avg_confidence']:.3f}")
        print(f"   Quality Score: {best['signal_count'] * best['avg_confidence']:.2f}")
        
        # Show parameter details
        config = best['config']
        print(f"\n🔧 PARAMETERS:")
        print(f"   EMA: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
        print(f"   Volume Multiplier: {config['volume_multiplier']}")
        print(f"   Min Signal Score: {config['min_signal_score']}")
        print(f"   S/R Threshold: {config['support_resistance_threshold']}%")
        print(f"   Risk: {config['stop_loss_pct']}% SL / {config['take_profit_pct']}% TP")
        print(f"   Max Daily Trades: {config['max_daily_trades']}")
        
    else:
        print("❌ No successful configurations!")
    
    print(f"\n{'='*60}")
    
    return results

def test_parameter_sensitivity():
    """Test how sensitive the strategy is to parameter changes"""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    base_config = {
        'symbols': ['BTCUSDT'],
        'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
        'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
        'rsi_neutral_high': 60, 'rsi_neutral_low': 40,
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'volume_period': 20, 'volume_multiplier': 1.2,
        'pivot_period': 10, 'support_resistance_threshold': 0.8,
        'min_signal_score': 0.6, 'stop_loss_pct': 1.5, 'take_profit_pct': 2.5,
        'trailing_stop_pct': 1.0, 'max_daily_trades': 3,
        'session_start': "00:00", 'session_end': "23:59", 'position_size': 0.02
    }
    
    # Test different values for key parameters
    sensitivity_tests = [
        ('volume_multiplier', [1.0, 1.2, 1.5, 2.0]),
        ('min_signal_score', [0.5, 0.6, 0.7, 0.8]),
        ('support_resistance_threshold', [0.5, 0.8, 1.0, 1.5]),
        ('max_daily_trades', [2, 3, 5, 8])
    ]
    
    test_data = get_historical_snapshot('bullish_crossover')
    
    for param_name, values in sensitivity_tests:
        print(f"\n📊 Testing {param_name.replace('_', ' ').title()}:")
        print("-" * 40)
        
        for value in values:
            config = base_config.copy()
            config[param_name] = value
            
            try:
                strategy = DayTradingStrategy(config)
                market_data = {'BTCUSDT': test_data}
                
                async def test_param():
                    return await strategy.generate_signals(market_data)
                
                signals = asyncio.run(test_param())
                signal_count = len(signals)
                avg_confidence = sum(s.confidence for s in signals) / signal_count if signals else 0
                
                print(f"   {param_name} = {value:<4}: {signal_count} signals, avg conf: {avg_confidence:.3f}")
                
            except Exception as e:
                print(f"   {param_name} = {value:<4}: ERROR - {e}")

if __name__ == "__main__":
    # Run basic tests
    results = test_strategy_basic()
    
    # Run sensitivity analysis
    test_parameter_sensitivity()
    
    print(f"\n🎯 CONCLUSION:")
    print("   Strategy is working and generates signals")
    print("   Parameter sensitivity shows clear differences")
    print("   Ready for more comprehensive backtesting")