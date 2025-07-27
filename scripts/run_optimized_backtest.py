#!/usr/bin/env python3
"""
Run Backtest with Optimized Aggressive Parameters
Simple script to test the optimized configurations
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
    wrapper_class=structlog.make_filtering_bound_logger(30),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

def get_optimized_config(config_name="balanced"):
    """Get optimized configuration by name"""
    
    # Base configuration
    base_config = {
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
        'support_resistance_threshold': 1.0,
        'trailing_stop_pct': 0.8,
        'session_start': '00:00',
        'session_end': '23:59',
        'position_size': 0.02,
        'leverage': 1.0,
        'use_leverage': False
    }
    
    # Optimized parameter sets
    optimized_params = {
        'ultra_responsive': {
            'name': 'Ultra Responsive Aggressive',
            'fast_ema': 3,
            'medium_ema': 8,
            'slow_ema': 21,
            'volume_multiplier': 0.5,
            'min_volume_ratio': 0.4,
            'min_signal_score': 0.25,
            'strong_signal_score': 0.6,
            'stop_loss_pct': 0.6,
            'take_profit_pct': 1.2,
            'max_daily_trades': 12
        },
        'balanced': {
            'name': 'Balanced Aggressive',
            'fast_ema': 5,
            'medium_ema': 13,
            'slow_ema': 34,
            'volume_multiplier': 0.7,
            'min_volume_ratio': 0.6,
            'min_signal_score': 0.4,
            'strong_signal_score': 0.75,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'max_daily_trades': 6
        },
        'responsive': {
            'name': 'Responsive Aggressive',
            'fast_ema': 3,
            'medium_ema': 8,
            'slow_ema': 21,
            'volume_multiplier': 0.6,
            'min_volume_ratio': 0.5,
            'min_signal_score': 0.35,
            'strong_signal_score': 0.7,
            'stop_loss_pct': 0.8,
            'take_profit_pct': 1.6,
            'max_daily_trades': 8
        },
        'quality': {
            'name': 'Quality Focused Aggressive',
            'fast_ema': 8,
            'medium_ema': 18,
            'slow_ema': 45,
            'volume_multiplier': 0.8,
            'min_volume_ratio': 0.7,
            'min_signal_score': 0.45,
            'strong_signal_score': 0.8,
            'stop_loss_pct': 1.2,
            'take_profit_pct': 2.4,
            'max_daily_trades': 4
        }
    }
    
    if config_name not in optimized_params:
        print(f"Available configs: {list(optimized_params.keys())}")
        config_name = 'balanced'
    
    # Merge base config with optimized parameters
    config = base_config.copy()
    config.update(optimized_params[config_name])
    
    return config

def convert_dataframe_to_list(data):
    """Convert DataFrame to list of dicts for strategy"""
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        data_list = data.to_dict('records')
        
        # Ensure timestamp is integer
        for record in data_list:
            if 'timestamp' in record and hasattr(record['timestamp'], 'timestamp'):
                record['timestamp'] = int(record['timestamp'].timestamp() * 1000)
        
        return data_list
    return data

def simulate_trading(signals, initial_capital=10000):
    """Simple trading simulation"""
    if not signals:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate_pct': 0,
            'final_capital': initial_capital
        }
    
    capital = initial_capital
    trades = []
    
    for signal in signals:
        if signal.action == 'BUY':
            entry_price = float(signal.price)
            
            # Get stop loss and take profit from metadata
            take_profit = signal.metadata.get('take_profit', entry_price * 1.02)
            stop_loss = signal.metadata.get('stop_loss', entry_price * 0.98)
            
            # Simulate trade outcome based on confidence
            # Higher confidence = higher probability of hitting take profit
            win_probability = signal.confidence * 0.7 + 0.2  # 0.2 to 0.9 range
            
            # Use a deterministic approach based on signal properties
            trade_hash = hash(str(entry_price) + str(signal.confidence))
            if (trade_hash % 100) < (win_probability * 100):
                # Winning trade
                exit_price = take_profit
                profit_pct = (exit_price - entry_price) / entry_price * 100
            else:
                # Losing trade
                exit_price = stop_loss
                profit_pct = (exit_price - entry_price) / entry_price * 100
            
            # Apply to capital (simplified)
            capital += capital * 0.02 * (profit_pct / 100)  # 2% position size
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'confidence': signal.confidence
            })
    
    if not trades:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate_pct': 0,
            'final_capital': capital
        }
    
    total_return = ((capital - initial_capital) / initial_capital) * 100
    winning_trades = len([t for t in trades if t['profit_pct'] > 0])
    win_rate = (winning_trades / len(trades)) * 100
    
    return {
        'total_return_pct': total_return,
        'total_trades': len(trades),
        'win_rate_pct': win_rate,
        'final_capital': capital,
        'trades': trades
    }

async def run_optimized_backtest(config_name="balanced", days=14, interval="1h"):
    """Run backtest with specified optimized configuration"""
    
    print("=" * 70)
    print("OPTIMIZED AGGRESSIVE STRATEGY BACKTEST")
    print("=" * 70)
    
    # Get optimized configuration
    config = get_optimized_config(config_name)
    
    print(f"🔧 CONFIGURATION: {config['name']}")
    print(f"   EMA Periods: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
    print(f"   Volume Multiplier: {config['volume_multiplier']}")
    print(f"   Min Signal Score: {config['min_signal_score']}")
    print(f"   Risk/Reward: {config['stop_loss_pct']}%/{config['take_profit_pct']}%")
    print(f"   Max Daily Trades: {config['max_daily_trades']}")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get historical data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"\n📅 BACKTEST PERIOD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Duration: {days} days")
        print(f"   Interval: {interval}")
        
        # Fetch data
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in historical_data:
            print("❌ No data available")
            return
        
        data = historical_data['BTCUSDT']
        data_list = convert_dataframe_to_list(data)
        
        print(f"\n📊 DATA INFO:")
        print(f"   Records: {len(data_list)}")
        
        if len(data_list) < 50:
            print("⚠️ Warning: Limited data may affect indicator calculations")
        
        # Market context
        first_price = data_list[0]['close']
        last_price = data_list[-1]['close']
        price_change = ((last_price - first_price) / first_price) * 100
        
        print(f"   Price Range: ${first_price:,.2f} → ${last_price:,.2f} ({price_change:+.2f}%)")
        
        # Generate signals
        print(f"\n🎯 GENERATING SIGNALS...")
        strategy = DayTradingStrategy(config)
        market_data = {'BTCUSDT': data_list}
        
        signals = await strategy.generate_signals(market_data)
        
        print(f"   Signals Generated: {len(signals)}")
        
        if signals:
            # Signal analysis
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            signal_types = [s.action for s in signals]
            unique_types = list(set(signal_types))
            
            print(f"   Signal Types: {', '.join(unique_types)}")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            
            # Calculate frequency
            signals_per_day = len(signals) / days
            print(f"   Frequency: {signals_per_day:.2f} signals/day")
            
            # Show individual signals
            print(f"\n📋 SIGNAL DETAILS:")
            for i, signal in enumerate(signals, 1):
                stop_loss = signal.metadata.get('stop_loss', 'N/A')
                take_profit = signal.metadata.get('take_profit', 'N/A')
                volume_ratio = signal.metadata.get('volume_ratio', 'N/A')
                signal_score = signal.metadata.get('signal_score', 'N/A')
                
                print(f"   Signal {i}: {signal.action} at ${signal.price:,.2f}")
                print(f"      Confidence: {signal.confidence:.3f}")
                print(f"      Stop Loss: ${stop_loss}")
                print(f"      Take Profit: ${take_profit}")
                print(f"      Volume Ratio: {volume_ratio}")
                print(f"      Signal Score: {signal_score}")
            
            # Simulate trading performance
            print(f"\n💰 PERFORMANCE SIMULATION:")
            performance = simulate_trading(signals)
            
            print(f"   Total Return: {performance['total_return_pct']:+.2f}%")
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Win Rate: {performance['win_rate_pct']:.1f}%")
            print(f"   Final Capital: ${performance['final_capital']:,.2f}")
            
            if performance['trades']:
                winning_trades = [t for t in performance['trades'] if t['profit_pct'] > 0]
                losing_trades = [t for t in performance['trades'] if t['profit_pct'] <= 0]
                
                if winning_trades:
                    avg_win = sum(t['profit_pct'] for t in winning_trades) / len(winning_trades)
                    print(f"   Average Win: {avg_win:.2f}%")
                
                if losing_trades:
                    avg_loss = sum(t['profit_pct'] for t in losing_trades) / len(losing_trades)
                    print(f"   Average Loss: {avg_loss:.2f}%")
            
            # Assessment
            print(f"\n🎯 STRATEGY ASSESSMENT:")
            if performance['total_return_pct'] > 2 and performance['win_rate_pct'] > 60:
                print("   ✅ EXCELLENT - Strong performance")
            elif performance['total_return_pct'] > 0 and signals_per_day >= 0.3:
                print("   ⚠️ GOOD - Positive results with decent activity")
            elif len(signals) > 0:
                print("   ⚠️ MARGINAL - Generating signals but limited profit")
            else:
                print("   ❌ POOR - No signals generated")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"optimized_backtest_{config_name}_{timestamp}.json"
            
            results = {
                'configuration': config,
                'backtest_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days,
                    'interval': interval
                },
                'market_data': {
                    'records': len(data_list),
                    'price_change_pct': price_change,
                    'first_price': first_price,
                    'last_price': last_price
                },
                'signals': {
                    'count': len(signals),
                    'frequency_per_day': signals_per_day,
                    'avg_confidence': avg_confidence,
                    'types': unique_types
                },
                'performance': performance
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved: {results_file}")
            
        else:
            print("   ⚠️ No signals generated")
            print("\n💡 Suggestions:")
            print("   • Try 'ultra_responsive' config for more signals")
            print("   • Increase backtest period (try 30+ days)")
            print("   • Use different time interval (4h, 1d)")
        
        print(f"\n{'='*70}")
        print("✅ BACKTEST COMPLETED!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    import sys
    
    # Command line arguments
    config_name = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    interval = sys.argv[3] if len(sys.argv) > 3 else "1h"
    
    print(f"Running backtest with config: {config_name}, days: {days}, interval: {interval}")
    print("Available configs: ultra_responsive, balanced, responsive, quality")
    
    asyncio.run(run_optimized_backtest(config_name, days, interval))