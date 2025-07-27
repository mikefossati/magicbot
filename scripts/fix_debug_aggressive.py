#!/usr/bin/env python3
"""
Fixed Debug Aggressive Strategy
Properly handle pandas DataFrame data format
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies.day_trading_strategy import DayTradingStrategy
from src.data.historical_manager import HistoricalDataManager
import structlog
import traceback

# Configure detailed logging
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

async def fixed_debug_strategy():
    """Debug with proper data format handling"""
    
    print("=" * 60)
    print("FIXED AGGRESSIVE STRATEGY DEBUG & BACKTEST")
    print("=" * 60)
    
    # Aggressive config
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
    
    print("🔧 AGGRESSIVE PARAMETERS:")
    print(f"   EMA: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
    print(f"   Min Signal Score: {config['min_signal_score']}")
    print(f"   Volume Multiplier: {config['volume_multiplier']}")
    print(f"   Risk: {config['stop_loss_pct']}%/{config['take_profit_pct']}%")
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get recent data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # 5 days
        
        print(f"\n📅 Fetching data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in historical_data:
            print("❌ No BTCUSDT data returned")
            return
        
        data = historical_data['BTCUSDT']
        print(f"📊 Data type: {type(data)}")
        print(f"📊 Data length: {len(data)}")
        
        # Handle different data formats
        if isinstance(data, pd.DataFrame):
            print("📋 Data is DataFrame - converting to list of dicts")
            data_list = data.to_dict('records')
            
            # Add timestamp as integer if it's datetime
            for record in data_list:
                if 'timestamp' in record and hasattr(record['timestamp'], 'timestamp'):
                    record['timestamp'] = int(record['timestamp'].timestamp() * 1000)
            
            print(f"📋 Converted to {len(data_list)} records")
            print(f"📋 Sample record: {data_list[0] if data_list else 'No data'}")
            
        elif isinstance(data, list):
            print("📋 Data is already list of dicts")
            data_list = data
        else:
            print(f"❌ Unexpected data format: {type(data)}")
            return
        
        if len(data_list) < 50:
            print(f"⚠️ Only {len(data_list)} records - may be insufficient")
        
        # Show price range
        if data_list:
            first_price = data_list[0]['close']
            last_price = data_list[-1]['close']
            price_change = ((last_price - first_price) / first_price) * 100
            
            print(f"💰 Price range: ${first_price:,.2f} → ${last_price:,.2f} ({price_change:+.2f}%)")
        
        # Create strategy and test
        print(f"\n🔄 Creating strategy and generating signals...")
        strategy = DayTradingStrategy(config)
        
        market_data = {'BTCUSDT': data_list}
        signals = await strategy.generate_signals(market_data)
        
        print(f"🎯 SIGNALS GENERATED: {len(signals)}")
        
        if signals:
            print(f"\n📊 SIGNAL DETAILS:")
            for i, signal in enumerate(signals, 1):
                print(f"   Signal {i}:")
                print(f"      Action: {signal.action}")
                print(f"      Price: ${signal.price:,.2f}")
                print(f"      Confidence: {signal.confidence:.3f}")
                print(f"      Quantity: {signal.quantity}")
                
                if signal.metadata:
                    stop_loss = signal.metadata.get('stop_loss', 'N/A')
                    take_profit = signal.metadata.get('take_profit', 'N/A')
                    volume_ratio = signal.metadata.get('volume_ratio', 'N/A')
                    signal_score = signal.metadata.get('signal_score', 'N/A')
                    
                    print(f"      Stop Loss: ${stop_loss}")
                    print(f"      Take Profit: ${take_profit}")
                    print(f"      Volume Ratio: {volume_ratio}")
                    print(f"      Signal Score: {signal_score}")
            
            # Calculate simple performance metrics
            print(f"\n📈 PERFORMANCE ANALYSIS:")
            
            # Simulate simple trading
            total_return = 0
            wins = 0
            losses = 0
            
            for signal in signals:
                # Simplified: assume trades are executed at signal price
                if signal.action == 'BUY':
                    # Assume next signal is exit or use take profit
                    entry_price = float(signal.price)
                    take_profit = signal.metadata.get('take_profit', entry_price * 1.02)
                    
                    # Simulate 2% gain (simplified)
                    trade_return = (take_profit - entry_price) / entry_price * 100
                    total_return += trade_return
                    
                    if trade_return > 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    print(f"   Simulated trade: ${entry_price:,.2f} → ${take_profit:,.2f} ({trade_return:+.2f}%)")
            
            if signals:
                avg_return = total_return / len(signals)
                win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
                
                print(f"\n📊 SIMULATION RESULTS:")
                print(f"   Total Simulated Return: {total_return:.2f}%")
                print(f"   Average Return per Trade: {avg_return:.2f}%")
                print(f"   Winning Trades: {wins}")
                print(f"   Losing Trades: {losses}")
                print(f"   Win Rate: {win_rate:.1f}%")
                
                # Frequency analysis
                days_tested = (end_date - start_date).days
                signals_per_day = len(signals) / days_tested if days_tested > 0 else 0
                
                print(f"\n🔄 FREQUENCY ANALYSIS:")
                print(f"   Days tested: {days_tested}")
                print(f"   Signals per day: {signals_per_day:.2f}")
                print(f"   Total signals: {len(signals)}")
                
                # Assessment
                print(f"\n🎯 AGGRESSIVE STRATEGY ASSESSMENT:")
                
                if total_return > 2 and win_rate > 50 and signals_per_day >= 0.5:
                    print("   ✅ EXCELLENT - High returns with good frequency")
                elif total_return > 0 and signals_per_day >= 0.3:
                    print("   ⚠️ GOOD - Positive returns with decent activity")
                elif total_return > 0:
                    print("   ⚠️ MARGINAL - Profitable but low activity")
                else:
                    print("   ❌ NEEDS IMPROVEMENT - Poor performance")
                
                # Specific feedback for aggressive strategy
                if signals_per_day < 0.5:
                    print("   💡 Consider lowering min_signal_score to increase frequency")
                if win_rate < 45:
                    print("   💡 Consider increasing min_signal_score for better quality")
                if total_return < 1:
                    print("   💡 Review risk/reward ratio and stop loss settings")
                
        else:
            print("⚠️ No signals generated")
            print("💡 Possible improvements:")
            print("   • Lower min_signal_score (try 0.4)")
            print("   • Reduce volume_multiplier (try 0.8)")
            print("   • Check if data has sufficient history for EMA calculations")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'configuration': config,
            'test_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'data_info': {
                'records': len(data_list),
                'price_change_pct': price_change if data_list else 0
            },
            'signals': len(signals),
            'performance': {
                'total_return_pct': total_return if signals else 0,
                'signals_per_day': signals_per_day if signals else 0,
                'win_rate_pct': win_rate if signals else 0
            }
        }
        
        results_file = f"aggressive_debug_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved: {results_file}")
        print(f"\n✅ AGGRESSIVE STRATEGY BACKTEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(fixed_debug_strategy())