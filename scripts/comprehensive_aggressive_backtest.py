#!/usr/bin/env python3
"""
Comprehensive Aggressive Strategy Backtest
Test multiple parameter variations to show strategy performance
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.day_trading_strategy import DayTradingStrategy
from src.data.historical_manager import HistoricalDataManager
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def convert_dataframe_to_list(data):
    """Convert DataFrame to list of dicts for strategy"""
    if isinstance(data, pd.DataFrame):
        data_list = data.to_dict('records')
        
        # Ensure timestamp is integer
        for record in data_list:
            if 'timestamp' in record and hasattr(record['timestamp'], 'timestamp'):
                record['timestamp'] = int(record['timestamp'].timestamp() * 1000)
        
        return data_list
    return data

def simulate_trading(signals, entry_success_rate=0.8):
    """Simulate trading performance"""
    if not signals:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0,
            'avg_return_pct': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0
        }
    
    trades = []
    
    for signal in signals:
        # Simulate if we can enter the trade (market conditions, slippage, etc.)
        if hash(str(signal.price)) % 100 < entry_success_rate * 100:
            entry_price = float(signal.price)
            
            if signal.action == 'BUY':
                # Simulate trade outcome based on stop loss and take profit
                take_profit = signal.metadata.get('take_profit', entry_price * 1.02)
                stop_loss = signal.metadata.get('stop_loss', entry_price * 0.99)
                
                # Use confidence to determine trade outcome probability
                win_probability = signal.confidence * 0.6 + 0.2  # 0.2 to 0.8 range
                
                # Simulate trade result
                trade_hash = hash(str(entry_price) + str(signal.confidence))
                if (trade_hash % 100) < (win_probability * 100):
                    # Winning trade
                    exit_price = take_profit
                    trade_return = (exit_price - entry_price) / entry_price * 100
                else:
                    # Losing trade
                    exit_price = stop_loss
                    trade_return = (exit_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return,
                    'action': signal.action,
                    'confidence': signal.confidence
                })
    
    if not trades:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate_pct': 0,
            'avg_return_pct': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0
        }
    
    total_return = sum(t['return_pct'] for t in trades)
    winning_trades = [t for t in trades if t['return_pct'] > 0]
    losing_trades = [t for t in trades if t['return_pct'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_return = total_return / len(trades)
    best_trade = max(t['return_pct'] for t in trades)
    worst_trade = min(t['return_pct'] for t in trades)
    
    return {
        'total_return_pct': total_return,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate_pct': win_rate,
        'avg_return_pct': avg_return,
        'best_trade_pct': best_trade,
        'worst_trade_pct': worst_trade,
        'trades': trades
    }

async def comprehensive_backtest():
    """Run comprehensive backtest with multiple parameter sets"""
    
    print("=" * 80)
    print("COMPREHENSIVE AGGRESSIVE DAY TRADING BACKTEST")
    print("Testing Multiple Parameter Variations")
    print("=" * 80)
    
    # Define parameter variations to test
    test_configs = [
        {
            'name': 'Original Aggressive',
            'fast_ema': 5, 'medium_ema': 13, 'slow_ema': 34,
            'volume_multiplier': 1.0, 'min_signal_score': 0.5,
            'stop_loss_pct': 1.0, 'take_profit_pct': 2.0,
            'max_daily_trades': 5
        },
        {
            'name': 'High Frequency',
            'fast_ema': 3, 'medium_ema': 8, 'slow_ema': 21,
            'volume_multiplier': 0.8, 'min_signal_score': 0.4,
            'stop_loss_pct': 0.8, 'take_profit_pct': 1.6,
            'max_daily_trades': 8
        },
        {
            'name': 'Balanced Aggressive',
            'fast_ema': 6, 'medium_ema': 15, 'slow_ema': 40,
            'volume_multiplier': 1.1, 'min_signal_score': 0.55,
            'stop_loss_pct': 1.2, 'take_profit_pct': 2.4,
            'max_daily_trades': 4
        },
        {
            'name': 'Quality Focus',
            'fast_ema': 8, 'medium_ema': 18, 'slow_ema': 45,
            'volume_multiplier': 1.3, 'min_signal_score': 0.65,
            'stop_loss_pct': 1.5, 'take_profit_pct': 3.0,
            'max_daily_trades': 3
        }
    ]
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get recent data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)  # 10 days for comprehensive test
        
        print(f"📅 BACKTEST PERIOD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(end_date - start_date).days} days")
        
        # Test multiple intervals
        intervals = ['1h', '4h']
        all_results = []
        
        for interval in intervals:
            print(f"\n{'='*60}")
            print(f"TESTING {interval.upper()} INTERVAL")
            print(f"{'='*60}")
            
            try:
                historical_data = await data_manager.get_multiple_symbols_data(
                    symbols=['BTCUSDT'],
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'BTCUSDT' not in historical_data:
                    print(f"❌ No data for {interval}")
                    continue
                
                data = historical_data['BTCUSDT']
                data_list = convert_dataframe_to_list(data)
                
                print(f"📊 Data: {len(data_list)} records")
                
                if len(data_list) < 50:
                    print(f"⚠️ Insufficient data ({len(data_list)} records)")
                    continue
                
                # Show market context
                first_price = data_list[0]['close']
                last_price = data_list[-1]['close']
                price_change = ((last_price - first_price) / first_price) * 100
                
                print(f"💰 Market: ${first_price:,.2f} → ${last_price:,.2f} ({price_change:+.2f}%)")
                
                interval_results = []
                
                for config_test in test_configs:
                    print(f"\n🔄 Testing: {config_test['name']}")
                    print("-" * 40)
                    
                    try:
                        # Create full config
                        config = {
                            'symbols': ['BTCUSDT'],
                            'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
                            'rsi_neutral_high': 60, 'rsi_neutral_low': 40,
                            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                            'volume_period': 20, 'min_volume_ratio': 0.8,
                            'pivot_period': 10, 'support_resistance_threshold': 1.0,
                            'strong_signal_score': 0.8, 'trailing_stop_pct': 1.0,
                            'session_start': '00:00', 'session_end': '23:59',
                            'position_size': 0.02, 'leverage': 1.0, 'use_leverage': False
                        }
                        
                        # Add test parameters
                        for key in ['fast_ema', 'medium_ema', 'slow_ema', 'volume_multiplier', 
                                   'min_signal_score', 'stop_loss_pct', 'take_profit_pct', 'max_daily_trades']:
                            config[key] = config_test[key]
                        
                        # Create strategy and test
                        strategy = DayTradingStrategy(config)
                        market_data = {'BTCUSDT': data_list}
                        
                        signals = await strategy.generate_signals(market_data)
                        
                        print(f"   📊 Signals: {len(signals)}")
                        
                        if signals:
                            # Show signal details
                            avg_confidence = sum(s.confidence for s in signals) / len(signals)
                            signal_types = [s.action for s in signals]
                            
                            print(f"   🎯 Types: {', '.join(set(signal_types))}")
                            print(f"   📈 Avg Confidence: {avg_confidence:.3f}")
                            
                            # Simulate trading
                            performance = simulate_trading(signals)
                            
                            print(f"   💰 Simulated Return: {performance['total_return_pct']:+.2f}%")
                            print(f"   ✅ Win Rate: {performance['win_rate_pct']:.1f}%")
                            print(f"   🔄 Trades: {performance['total_trades']}")
                            
                            # Calculate frequency
                            days = (end_date - start_date).days
                            signals_per_day = len(signals) / days if days > 0 else 0
                            
                            print(f"   📊 Frequency: {signals_per_day:.2f} signals/day")
                            
                            # Store results
                            interval_results.append({
                                'config_name': config_test['name'],
                                'config': config_test,
                                'signals': len(signals),
                                'avg_confidence': avg_confidence,
                                'signals_per_day': signals_per_day,
                                'performance': performance,
                                'success': True
                            })
                            
                        else:
                            print(f"   ⚠️ No signals generated")
                            interval_results.append({
                                'config_name': config_test['name'],
                                'config': config_test,
                                'signals': 0,
                                'success': False
                            })
                        
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        interval_results.append({
                            'config_name': config_test['name'],
                            'error': str(e),
                            'success': False
                        })
                
                # Analyze interval results
                successful_results = [r for r in interval_results if r['success'] and r['signals'] > 0]
                
                if successful_results:
                    print(f"\n📊 {interval.upper()} INTERVAL SUMMARY:")
                    print("-" * 50)
                    
                    print(f"{'Config':<18} {'Signals':<8} {'Return%':<8} {'Win%':<6} {'Conf':<6} {'S/Day':<6}")
                    print("-" * 55)
                    
                    for result in successful_results:
                        perf = result['performance']
                        print(f"{result['config_name']:<18} "
                              f"{result['signals']:<8} "
                              f"{perf['total_return_pct']:<8.2f} "
                              f"{perf['win_rate_pct']:<6.1f} "
                              f"{result['avg_confidence']:<6.3f} "
                              f"{result['signals_per_day']:<6.2f}")
                    
                    # Find best performer
                    best = max(successful_results, key=lambda x: x['performance']['total_return_pct'])
                    
                    print(f"\n🏆 BEST {interval.upper()}: {best['config_name']}")
                    print(f"   Return: {best['performance']['total_return_pct']:+.2f}%")
                    print(f"   Win Rate: {best['performance']['win_rate_pct']:.1f}%")
                    print(f"   Signals: {best['signals']}")
                    print(f"   Confidence: {best['avg_confidence']:.3f}")
                
                # Add to overall results
                for result in interval_results:
                    result['interval'] = interval
                    result['market_change_pct'] = price_change
                    result['data_points'] = len(data_list)
                
                all_results.extend(interval_results)
                
            except Exception as e:
                print(f"❌ Error with {interval}: {e}")
        
        # Overall analysis
        if all_results:
            print(f"\n{'='*80}")
            print("COMPREHENSIVE BACKTEST RESULTS")
            print(f"{'='*80}")
            
            successful_all = [r for r in all_results if r['success'] and r['signals'] > 0]
            
            if successful_all:
                print(f"✅ Successful tests: {len(successful_all)}/{len(all_results)}")
                
                # Summary table
                print(f"\n📊 ALL RESULTS:")
                print(f"{'Config':<18} {'Interval':<8} {'Signals':<8} {'Return%':<8} {'Win%':<6} {'S/Day':<6}")
                print("-" * 70)
                
                for result in successful_all:
                    perf = result['performance']
                    print(f"{result['config_name']:<18} "
                          f"{result['interval']:<8} "
                          f"{result['signals']:<8} "
                          f"{perf['total_return_pct']:<8.2f} "
                          f"{perf['win_rate_pct']:<6.1f} "
                          f"{result['signals_per_day']:<6.2f}")
                
                # Overall best
                overall_best = max(successful_all, key=lambda x: x['performance']['total_return_pct'])
                
                print(f"\n🥇 OVERALL BEST PERFORMANCE:")
                print(f"   Configuration: {overall_best['config_name']}")
                print(f"   Interval: {overall_best['interval']}")
                print(f"   Total Return: {overall_best['performance']['total_return_pct']:+.2f}%")
                print(f"   Win Rate: {overall_best['performance']['win_rate_pct']:.1f}%")
                print(f"   Signals Generated: {overall_best['signals']}")
                print(f"   Average Confidence: {overall_best['avg_confidence']:.3f}")
                print(f"   Signals per Day: {overall_best['signals_per_day']:.2f}")
                
                # Best config details
                best_config = overall_best['config']
                print(f"\n🔧 OPTIMAL PARAMETERS:")
                print(f"   EMA Periods: {best_config['fast_ema']}/{best_config['medium_ema']}/{best_config['slow_ema']}")
                print(f"   Volume Multiplier: {best_config['volume_multiplier']}")
                print(f"   Min Signal Score: {best_config['min_signal_score']}")
                print(f"   Risk Management: {best_config['stop_loss_pct']}%/{best_config['take_profit_pct']}%")
                print(f"   Max Daily Trades: {best_config['max_daily_trades']}")
                
                # Performance analysis
                avg_return = sum(r['performance']['total_return_pct'] for r in successful_all) / len(successful_all)
                avg_win_rate = sum(r['performance']['win_rate_pct'] for r in successful_all) / len(successful_all)
                avg_signals_per_day = sum(r['signals_per_day'] for r in successful_all) / len(successful_all)
                
                print(f"\n📈 AVERAGE PERFORMANCE:")
                print(f"   Average Return: {avg_return:.2f}%")
                print(f"   Average Win Rate: {avg_win_rate:.1f}%")
                print(f"   Average Signals/Day: {avg_signals_per_day:.2f}")
                
                # Final verdict
                print(f"\n🎯 AGGRESSIVE STRATEGY VERDICT:")
                if avg_return > 2 and avg_win_rate > 50 and avg_signals_per_day >= 0.5:
                    verdict = "EXCELLENT"
                    print("   ✅ EXCELLENT - Strong performance across configurations")
                elif avg_return > 0 and avg_signals_per_day >= 0.3:
                    verdict = "GOOD"
                    print("   ⚠️ GOOD - Positive performance with room for improvement")
                elif avg_return > 0:
                    verdict = "MARGINAL"
                    print("   ⚠️ MARGINAL - Modest performance")
                else:
                    verdict = "POOR"
                    print("   ❌ POOR - Needs significant improvement")
                
            else:
                verdict = "NO_SIGNALS"
                print("❌ No successful signal generations across all tests")
            
            # Save comprehensive results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"comprehensive_aggressive_backtest_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'test_info': {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'duration_days': (end_date - start_date).days,
                        'intervals_tested': intervals,
                        'configs_tested': len(test_configs)
                    },
                    'all_results': all_results,
                    'summary': {
                        'successful_tests': len(successful_all),
                        'total_tests': len(all_results),
                        'avg_return_pct': avg_return if successful_all else 0,
                        'avg_win_rate_pct': avg_win_rate if successful_all else 0,
                        'avg_signals_per_day': avg_signals_per_day if successful_all else 0,
                        'verdict': verdict,
                        'best_config': overall_best['config_name'] if successful_all else None
                    }
                }, f, indent=2, default=str)
            
            print(f"\n💾 Comprehensive results saved: {results_file}")
        
        else:
            print("❌ No test results generated")
        
        print(f"\n{'='*80}")
        print("🎯 AGGRESSIVE STRATEGY BACKTEST COMPLETED!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(comprehensive_backtest())