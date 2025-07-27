#!/usr/bin/env python3
"""
Simple Aggressive Strategy Backtest
Direct testing of the aggressive strategy without complex backtest engine
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta

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

def calculate_trade_performance(signals, historical_data, initial_capital=10000):
    """Simple trade performance calculation"""
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]
    
    # Create price lookup
    price_data = {data['timestamp']: data for data in historical_data}
    
    for signal in signals:
        signal_time = None
        signal_price = float(signal.price)
        
        # Find closest price data point
        for timestamp, data in price_data.items():
            if abs(data['close'] - signal_price) < signal_price * 0.01:  # Within 1%
                signal_time = timestamp
                break
        
        if signal.action == 'BUY' and position is None:
            # Open long position
            position_size = capital * float(signal.quantity)
            shares = position_size / signal_price
            
            position = {
                'type': 'LONG',
                'shares': shares,
                'entry_price': signal_price,
                'entry_time': signal_time,
                'stop_loss': signal.metadata.get('stop_loss'),
                'take_profit': signal.metadata.get('take_profit')
            }
            capital -= position_size
            
        elif signal.action == 'SELL' and position and position['type'] == 'LONG':
            # Close long position
            proceeds = position['shares'] * signal_price
            trade_return = proceeds - (position['shares'] * position['entry_price'])
            
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': signal_price,
                'return_amount': trade_return,
                'return_pct': (trade_return / (position['shares'] * position['entry_price'])) * 100,
                'confidence': signal.confidence
            })
            
            capital += proceeds
            position = None
        
        equity_curve.append(capital)
    
    return {
        'final_capital': capital,
        'trades': trades,
        'equity_curve': equity_curve
    }

def analyze_results(results, initial_capital):
    """Analyze trading results"""
    final_capital = results['final_capital']
    trades = results['trades']
    
    if not trades:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate_pct': 0,
            'avg_return_pct': 0,
            'best_trade_pct': 0,
            'worst_trade_pct': 0
        }
    
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    winning_trades = [t for t in trades if t['return_amount'] > 0]
    losing_trades = [t for t in trades if t['return_amount'] <= 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
    avg_return = sum(t['return_pct'] for t in trades) / len(trades) if trades else 0
    best_trade = max(t['return_pct'] for t in trades) if trades else 0
    worst_trade = min(t['return_pct'] for t in trades) if trades else 0
    
    return {
        'total_return_pct': total_return,
        'total_trades': len(trades),
        'win_rate_pct': win_rate,
        'avg_return_pct': avg_return,
        'best_trade_pct': best_trade,
        'worst_trade_pct': worst_trade,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades)
    }

async def simple_aggressive_backtest():
    """Run simple backtest of aggressive strategy"""
    
    print("=" * 70)
    print("SIMPLE AGGRESSIVE STRATEGY BACKTEST")
    print("Direct Signal Generation and Performance Analysis")
    print("=" * 70)
    
    # Aggressive configuration
    config = {
        'symbols': ['BTCUSDT'],
        'fast_ema': 5,
        'medium_ema': 13,
        'slow_ema': 34,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'rsi_neutral_high': 60,
        'rsi_neutral_low': 40,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'volume_period': 20,
        'volume_multiplier': 1.0,
        'min_volume_ratio': 0.8,
        'pivot_period': 10,
        'support_resistance_threshold': 1.0,
        'min_signal_score': 0.5,
        'strong_signal_score': 0.8,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0,
        'trailing_stop_pct': 1.0,
        'max_daily_trades': 5,
        'session_start': '00:00',
        'session_end': '23:59',
        'position_size': 0.02,
        'leverage': 1.0,
        'use_leverage': False
    }
    
    print(f"🔧 AGGRESSIVE PARAMETERS:")
    print(f"   EMA: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
    print(f"   Volume Multiplier: {config['volume_multiplier']}")
    print(f"   Min Signal Score: {config['min_signal_score']}")
    print(f"   Risk: {config['stop_loss_pct']}%/{config['take_profit_pct']}%")
    print(f"   Max Daily Trades: {config['max_daily_trades']}")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get recent data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # 7 days for quick test
        
        print(f"\n📅 Test Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        intervals = ['1h', '4h']
        results_summary = []
        
        for interval in intervals:
            print(f"\n{'='*50}")
            print(f"TESTING {interval.upper()} INTERVAL")
            print(f"{'='*50}")
            
            try:
                # Fetch data
                historical_data = await data_manager.get_multiple_symbols_data(
                    symbols=['BTCUSDT'],
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if 'BTCUSDT' not in historical_data:
                    print(f"❌ No data available for {interval}")
                    continue
                
                data = historical_data['BTCUSDT']
                print(f"📊 Data points: {len(data)}")
                
                if len(data) < 50:
                    print(f"⚠️ Insufficient data for analysis")
                    continue
                
                # Show data info
                first_price = data[0]['close']
                last_price = data[-1]['close']
                price_change = ((last_price - first_price) / first_price) * 100
                
                print(f"💰 Price: ${first_price:,.2f} → ${last_price:,.2f} ({price_change:+.2f}%)")
                
                # Create strategy and generate signals
                strategy = DayTradingStrategy(config)
                market_data = {'BTCUSDT': data}
                
                signals = await strategy.generate_signals(market_data)
                
                print(f"🎯 Signals generated: {len(signals)}")
                
                if signals:
                    # Show signal details
                    for i, signal in enumerate(signals):
                        print(f"   Signal {i+1}: {signal.action} at ${signal.price:.2f} (conf: {signal.confidence:.3f})")
                    
                    # Calculate performance
                    performance = calculate_trade_performance(signals, data)
                    metrics = analyze_results(performance, 10000)
                    
                    print(f"\n📈 PERFORMANCE:")
                    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
                    print(f"   Total Trades: {metrics['total_trades']}")
                    print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
                    print(f"   Avg Trade: {metrics['avg_return_pct']:.2f}%")
                    print(f"   Best Trade: {metrics['best_trade_pct']:.2f}%")
                    print(f"   Worst Trade: {metrics['worst_trade_pct']:.2f}%")
                    
                    # Calculate frequency
                    days_tested = (end_date - start_date).days
                    signals_per_day = len(signals) / days_tested if days_tested > 0 else 0
                    
                    print(f"\n📊 FREQUENCY:")
                    print(f"   Signals per day: {signals_per_day:.2f}")
                    print(f"   Trades per day: {metrics['total_trades'] / days_tested:.2f}")
                    
                    # Store results
                    results_summary.append({
                        'interval': interval,
                        'data_points': len(data),
                        'signals': len(signals),
                        'total_return_pct': metrics['total_return_pct'],
                        'win_rate_pct': metrics['win_rate_pct'],
                        'total_trades': metrics['total_trades'],
                        'signals_per_day': signals_per_day,
                        'avg_confidence': sum(s.confidence for s in signals) / len(signals),
                        'price_change_pct': price_change
                    })
                    
                else:
                    print(f"⚠️ No signals generated")
                    results_summary.append({
                        'interval': interval,
                        'data_points': len(data),
                        'signals': 0,
                        'price_change_pct': price_change
                    })
                
            except Exception as e:
                print(f"❌ Error with {interval}: {e}")
        
        # Summary
        if results_summary:
            print(f"\n{'='*70}")
            print("AGGRESSIVE STRATEGY RESULTS SUMMARY")
            print(f"{'='*70}")
            
            successful_tests = [r for r in results_summary if r['signals'] > 0]
            
            if successful_tests:
                print(f"{'Interval':<8} {'Signals':<8} {'Return%':<8} {'Win%':<6} {'Conf':<6} {'Sig/Day':<7}")
                print("-" * 50)
                
                for result in successful_tests:
                    print(f"{result['interval']:<8} "
                          f"{result['signals']:<8} "
                          f"{result['total_return_pct']:<8.2f} "
                          f"{result['win_rate_pct']:<6.1f} "
                          f"{result['avg_confidence']:<6.3f} "
                          f"{result['signals_per_day']:<7.2f}")
                
                # Best performing
                best = max(successful_tests, key=lambda x: x['total_return_pct'])
                
                print(f"\n🏆 BEST PERFORMANCE: {best['interval']}")
                print(f"   Return: {best['total_return_pct']:.2f}%")
                print(f"   Signals: {best['signals']}")
                print(f"   Win Rate: {best['win_rate_pct']:.1f}%")
                print(f"   Confidence: {best['avg_confidence']:.3f}")
                print(f"   Frequency: {best['signals_per_day']:.2f} signals/day")
                
                # Overall assessment
                avg_return = sum(r['total_return_pct'] for r in successful_tests) / len(successful_tests)
                avg_signals_per_day = sum(r['signals_per_day'] for r in successful_tests) / len(successful_tests)
                avg_confidence = sum(r['avg_confidence'] for r in successful_tests) / len(successful_tests)
                
                print(f"\n🎯 OVERALL ASSESSMENT:")
                print(f"   Average Return: {avg_return:.2f}%")
                print(f"   Average Signals/Day: {avg_signals_per_day:.2f}")
                print(f"   Average Confidence: {avg_confidence:.3f}")
                
                if avg_return > 1 and avg_signals_per_day >= 0.5 and avg_confidence >= 0.6:
                    print("   ✅ STRONG PERFORMANCE - Strategy is effective")
                elif avg_return > 0 and avg_signals_per_day >= 0.2:
                    print("   ⚠️ MODERATE PERFORMANCE - Has potential")
                else:
                    print("   ❌ NEEDS IMPROVEMENT - Consider parameter adjustment")
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"simple_aggressive_results_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump({
                        'configuration': config,
                        'test_period': {
                            'start': start_date.isoformat(),
                            'end': end_date.isoformat(),
                            'days': (end_date - start_date).days
                        },
                        'results': results_summary,
                        'summary': {
                            'avg_return_pct': avg_return,
                            'avg_signals_per_day': avg_signals_per_day,
                            'avg_confidence': avg_confidence
                        }
                    }, f, indent=2, default=str)
                
                print(f"\n💾 Results saved: {results_file}")
                
            else:
                print("❌ No signals generated in any interval")
                print("   Strategy may need parameter adjustment")
        
        print(f"\n{'='*70}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error("Simple backtest error", error=str(e))
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(simple_aggressive_backtest())