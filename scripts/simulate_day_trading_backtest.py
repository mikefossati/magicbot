#!/usr/bin/env python3
"""
Simulate Day Trading Backtest
Test the optimized day trading strategy with simulated price movements
"""

import asyncio
import sys
import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.day_trading_strategy import DayTradingStrategy

def generate_realistic_price_data(initial_price=50000, num_points=200, volatility=0.02):
    """Generate realistic cryptocurrency price data"""
    prices = [initial_price]
    volumes = []
    
    for i in range(num_points - 1):
        # Random walk with slight upward bias
        change = np.random.normal(0.0001, volatility)  # Small upward bias
        new_price = prices[-1] * (1 + change)
        
        # Ensure price doesn't go below 10% of initial
        new_price = max(new_price, initial_price * 0.1)
        prices.append(new_price)
        
        # Generate volume (inversely correlated with price stability)
        base_volume = 1000000
        volatility_factor = abs(change) * 1000000
        volume = base_volume + volatility_factor + random.uniform(-100000, 100000)
        volumes.append(max(volume, 10000))
    
    # Add final volume
    volumes.append(volumes[-1] + random.uniform(-50000, 50000))
    
    # Create timestamps (15-minute intervals)
    start_time = datetime.now() - timedelta(minutes=15 * num_points)
    timestamps = []
    for i in range(num_points):
        timestamps.append(start_time + timedelta(minutes=15 * i))
    
    # Create OHLC data
    data = []
    for i in range(num_points):
        price = prices[i]
        # Generate realistic OHLC from price
        noise = random.uniform(-0.01, 0.01)
        
        open_price = price * (1 + noise * 0.5)
        high_price = price * (1 + abs(noise))
        low_price = price * (1 - abs(noise))
        close_price = price
        
        data.append({
            'timestamp': int(timestamps[i].timestamp() * 1000),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volumes[i]
        })
    
    return data

def simulate_trading(signals, initial_capital=10000, commission=0.001):
    """Simulate trading based on signals"""
    
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]
    
    for signal in signals:
        price = float(signal.price)
        action = signal.action
        
        if action == 'BUY' and position is None:
            # Open long position
            shares = (capital * float(signal.quantity)) / price
            cost = shares * price * (1 + commission)
            
            if cost <= capital:
                position = {
                    'type': 'LONG',
                    'shares': shares,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'stop_loss': signal.metadata.get('stop_loss'),
                    'take_profit': signal.metadata.get('take_profit')
                }
                capital -= cost
                
        elif action == 'SELL' and position and position['type'] == 'LONG':
            # Close long position
            proceeds = position['shares'] * price * (1 - commission)
            trade_return = proceeds - (position['shares'] * position['entry_price'])
            
            trades.append({
                'type': 'LONG',
                'entry_price': position['entry_price'],
                'exit_price': price,
                'shares': position['shares'],
                'return': trade_return,
                'return_pct': (trade_return / (position['shares'] * position['entry_price'])) * 100,
                'duration': datetime.now() - position['entry_time']
            })
            
            capital += proceeds
            position = None
            
        elif action == 'SELL' and position is None:
            # Open short position (simplified)
            shares = (capital * float(signal.quantity)) / price
            proceeds = shares * price * (1 - commission)
            
            position = {
                'type': 'SHORT',
                'shares': shares,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop_loss': signal.metadata.get('stop_loss'),
                'take_profit': signal.metadata.get('take_profit')
            }
            capital += proceeds
        
        equity_curve.append(capital)
    
    return {
        'final_capital': capital,
        'trades': trades,
        'equity_curve': equity_curve,
        'position': position
    }

def analyze_backtest_results(results, initial_capital):
    """Analyze simulated backtest results"""
    
    final_capital = results['final_capital']
    trades = results['trades']
    
    if not trades:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate_pct': 0,
            'avg_return_pct': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0
        }
    
    # Calculate metrics
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    total_trades = len(trades)
    
    winning_trades = [t for t in trades if t['return'] > 0]
    losing_trades = [t for t in trades if t['return'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if trades else 0
    
    avg_return = sum(t['return_pct'] for t in trades) / total_trades if trades else 0
    
    # Profit factor
    total_profit = sum(t['return'] for t in winning_trades)
    total_loss = abs(sum(t['return'] for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Max drawdown (simplified)
    equity_curve = results['equity_curve']
    running_max = equity_curve[0]
    max_drawdown = 0
    
    for value in equity_curve:
        if value > running_max:
            running_max = value
        drawdown = ((running_max - value) / running_max) * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return_pct': total_return,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'avg_return_pct': avg_return,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

async def test_optimized_strategy():
    """Test the optimized day trading strategy"""
    
    print("=" * 70)
    print("OPTIMIZED DAY TRADING STRATEGY SIMULATION")
    print("=" * 70)
    
    # Optimized configuration based on our findings
    optimized_configs = [
        {
            'name': 'Fast Responsive (5/13/34)',
            'fast_ema': 5, 'medium_ema': 13, 'slow_ema': 34,
            'volume_multiplier': 1.0, 'min_signal_score': 0.5,
            'support_resistance_threshold': 1.0,
            'stop_loss_pct': 1.0, 'take_profit_pct': 2.0,
            'max_daily_trades': 5
        },
        {
            'name': 'Balanced Quality (8/21/50)',
            'fast_ema': 8, 'medium_ema': 21, 'slow_ema': 50,
            'volume_multiplier': 1.2, 'min_signal_score': 0.6,
            'support_resistance_threshold': 0.8,
            'stop_loss_pct': 1.5, 'take_profit_pct': 2.5,
            'max_daily_trades': 3
        },
        {
            'name': 'High Volume Confidence',
            'fast_ema': 6, 'medium_ema': 15, 'slow_ema': 40,
            'volume_multiplier': 1.8, 'min_signal_score': 0.6,
            'support_resistance_threshold': 0.8,
            'stop_loss_pct': 1.2, 'take_profit_pct': 2.4,
            'max_daily_trades': 4
        }
    ]
    
    # Generate test data
    print("📊 Generating realistic price data...")
    price_data = generate_realistic_price_data(initial_price=50000, num_points=200, volatility=0.025)
    print(f"   Generated {len(price_data)} price points")
    
    results_summary = []
    
    for config_test in optimized_configs:
        print(f"\n🔄 Testing: {config_test['name']}")
        print("-" * 50)
        
        try:
            # Create full strategy config
            config = {
                'symbols': ['BTCUSDT'],
                'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
                'rsi_neutral_high': 60, 'rsi_neutral_low': 40,
                'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                'volume_period': 20, 'pivot_period': 10,
                'trailing_stop_pct': 1.0, 'session_start': "00:00", 'session_end': "23:59",
                'position_size': 0.02
            }
            
            # Add test parameters
            for key in ['fast_ema', 'medium_ema', 'slow_ema', 'volume_multiplier', 
                       'min_signal_score', 'support_resistance_threshold',
                       'stop_loss_pct', 'take_profit_pct', 'max_daily_trades']:
                config[key] = config_test[key]
            
            # Create strategy and generate signals
            strategy = DayTradingStrategy(config)
            market_data = {'BTCUSDT': price_data}
            
            signals = await strategy.generate_signals(market_data)
            
            print(f"   📈 Signals generated: {len(signals)}")
            
            if signals:
                signal_types = [s.action for s in signals]
                avg_confidence = sum(s.confidence for s in signals) / len(signals)
                print(f"   🎯 Signal types: {', '.join(set(signal_types))}")
                print(f"   📊 Average confidence: {avg_confidence:.3f}")
                
                # Simulate trading
                trading_results = simulate_trading(signals, initial_capital=10000)
                metrics = analyze_backtest_results(trading_results, initial_capital=10000)
                
                print(f"   💰 Total Return: {metrics['total_return_pct']:.2f}%")
                print(f"   🔄 Total Trades: {metrics['total_trades']}")
                print(f"   ✅ Win Rate: {metrics['win_rate_pct']:.1f}%")
                print(f"   📉 Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
                print(f"   ⚖️ Profit Factor: {metrics['profit_factor']:.2f}")
                
                # Calculate score
                score = (
                    metrics['total_return_pct'] * 0.3 +
                    metrics['win_rate_pct'] * 0.3 +
                    (1 / (1 + metrics['max_drawdown_pct'])) * 20 * 0.2 +
                    len(signals) * 0.2
                )
                
                results_summary.append({
                    'name': config_test['name'],
                    'config': config,
                    'signals': len(signals),
                    'avg_confidence': avg_confidence,
                    'total_return_pct': metrics['total_return_pct'],
                    'win_rate_pct': metrics['win_rate_pct'],
                    'total_trades': metrics['total_trades'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'profit_factor': metrics['profit_factor'],
                    'score': score
                })
                
            else:
                print(f"   ⚠️ No signals generated")
                results_summary.append({
                    'name': config_test['name'],
                    'signals': 0,
                    'score': 0
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Analysis and recommendations
    print(f"\n{'='*70}")
    print("SIMULATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    successful_results = [r for r in results_summary if r.get('signals', 0) > 0]
    
    if successful_results:
        # Sort by score
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'Configuration':<25} {'Signals':<8} {'Return%':<8} {'Win%':<6} {'Trades':<7} {'Score':<8}")
        print("-" * 70)
        
        for result in successful_results:
            print(f"{result['name']:<25} "
                  f"{result['signals']:<8} "
                  f"{result['total_return_pct']:<8.2f} "
                  f"{result['win_rate_pct']:<6.1f} "
                  f"{result['total_trades']:<7} "
                  f"{result['score']:<8.2f}")
        
        # Best configuration
        best = successful_results[0]
        print(f"\n🏆 BEST PERFORMING CONFIGURATION: {best['name']}")
        print("-" * 50)
        
        config = best['config']
        print(f"📊 Parameters:")
        print(f"   EMA Periods: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
        print(f"   Volume Multiplier: {config['volume_multiplier']}")
        print(f"   Min Signal Score: {config['min_signal_score']}")
        print(f"   S/R Threshold: {config['support_resistance_threshold']}%")
        print(f"   Risk Management: {config['stop_loss_pct']}% SL / {config['take_profit_pct']}% TP")
        print(f"   Max Daily Trades: {config['max_daily_trades']}")
        
        print(f"\n🎯 Performance:")
        print(f"   Signals Generated: {best['signals']}")
        print(f"   Average Confidence: {best['avg_confidence']:.3f}")
        print(f"   Total Return: {best['total_return_pct']:.2f}%")
        print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
        print(f"   Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        
        # Save optimal configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = f"optimal_day_trading_config_{timestamp}.json"
        
        with open(config_file, 'w') as f:
            json.dump(best['config'], f, indent=2)
        
        print(f"\n💾 Optimal configuration saved to: {config_file}")
        
        # Final recommendations
        print(f"\n🎖️ RECOMMENDATIONS:")
        if best['total_return_pct'] > 2:
            print("✅ Excellent performance - ready for live testing")
        elif best['total_return_pct'] > 0:
            print("⚠️ Positive performance - consider paper trading first")
        else:
            print("❌ Needs improvement - revise parameters")
            
        if best['win_rate_pct'] > 60:
            print("✅ High win rate - good signal quality")
        elif best['win_rate_pct'] > 45:
            print("⚠️ Moderate win rate - acceptable")
        else:
            print("❌ Low win rate - improve signal filtering")
        
        return best['config']
    
    else:
        print("❌ No successful configurations found!")
        return None

if __name__ == "__main__":
    optimal_config = asyncio.run(test_optimized_strategy())
    
    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    if optimal_config:
        print("   Optimal parameters identified")
        print("   Ready for paper trading or live implementation")
    else:
        print("   Strategy needs further optimization")
    
    print(f"{'='*70}")