#!/usr/bin/env python3
"""
Dynamic Aggressive Strategy
Automatically adjusts parameters based on market conditions
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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

class DynamicAggressiveStrategy:
    """Dynamic strategy that adapts parameters based on market conditions"""
    
    def __init__(self, config_file_path=None):
        # Load optimized configurations
        if config_file_path:
            with open(config_file_path, 'r') as f:
                self.configs = json.load(f)
        else:
            # Default optimized config
            self.configs = {
                "configurations": {
                    "ultra_responsive": {
                        "fast_ema": 3, "medium_ema": 8, "slow_ema": 21,
                        "volume_multiplier": 0.5, "min_volume_ratio": 0.4,
                        "min_signal_score": 0.25, "strong_signal_score": 0.6,
                        "stop_loss_pct": 0.6, "take_profit_pct": 1.2,
                        "max_daily_trades": 12
                    }
                },
                "base_config": {
                    "symbols": ["BTCUSDT"], "rsi_period": 14,
                    "rsi_overbought": 70, "rsi_oversold": 30,
                    "rsi_neutral_high": 60, "rsi_neutral_low": 40,
                    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                    "volume_period": 20, "pivot_period": 10,
                    "support_resistance_threshold": 1.0,
                    "trailing_stop_pct": 0.8,
                    "session_start": "00:00", "session_end": "23:59",
                    "position_size": 0.02, "leverage": 1.0, "use_leverage": False
                }
            }
    
    def analyze_market_conditions(self, data):
        """Analyze market conditions to determine optimal parameters"""
        df = pd.DataFrame(data)
        
        # Calculate volatility (20-period rolling std of returns)
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(24)  # Annualized
        
        # Calculate trend strength (price vs moving averages)
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
        current_price = df['close'].iloc[-1]
        
        trend_strength = ((current_price - ma_20) / ma_20) * 100
        
        # Calculate volume trend
        volume_ma = df['volume'].rolling(20).mean()
        recent_volume_trend = (df['volume'].iloc[-5:].mean() / volume_ma.iloc[-1]) if len(volume_ma) > 0 else 1.0
        
        # Determine market regime
        market_regime = self._classify_market_regime(volatility, trend_strength, recent_volume_trend)
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'volume_trend': recent_volume_trend,
            'regime': market_regime,
            'current_price': current_price,
            'ma_20': ma_20,
            'ma_50': ma_50
        }
    
    def _classify_market_regime(self, volatility, trend_strength, volume_trend):
        """Classify current market regime"""
        
        # High volatility threshold (equivalent to crypto VIX > 80)
        high_vol_threshold = 0.8
        low_vol_threshold = 0.3
        
        # Strong trend threshold
        strong_trend_threshold = 5.0  # 5% from MA
        
        conditions = []
        
        if volatility > high_vol_threshold:
            conditions.append('high_volatility')
        elif volatility < low_vol_threshold:
            conditions.append('low_volatility')
        
        if abs(trend_strength) > strong_trend_threshold:
            conditions.append('trending_market')
        else:
            conditions.append('sideways_market')
        
        if volume_trend > 1.2:
            conditions.append('high_volume')
        elif volume_trend < 0.8:
            conditions.append('low_volume')
        
        return conditions
    
    def adapt_parameters(self, base_config, market_analysis):
        """Adapt strategy parameters based on market conditions"""
        adapted_config = base_config.copy()
        
        regime = market_analysis['regime']
        volatility = market_analysis['volatility']
        trend_strength = market_analysis['trend_strength']
        
        # Base adjustments for high volatility
        if 'high_volatility' in regime:
            adapted_config['volume_multiplier'] = max(0.3, adapted_config['volume_multiplier'] - 0.1)
            adapted_config['min_signal_score'] = max(0.15, adapted_config['min_signal_score'] - 0.05)
            adapted_config['stop_loss_pct'] = adapted_config['stop_loss_pct'] + 0.2
            adapted_config['take_profit_pct'] = adapted_config['take_profit_pct'] + 0.4
            adapted_config['max_daily_trades'] = min(15, adapted_config['max_daily_trades'] + 2)
        
        # Base adjustments for low volatility
        elif 'low_volatility' in regime:
            adapted_config['volume_multiplier'] = min(1.0, adapted_config['volume_multiplier'] + 0.1)
            adapted_config['min_signal_score'] = min(0.6, adapted_config['min_signal_score'] + 0.05)
            adapted_config['stop_loss_pct'] = max(0.4, adapted_config['stop_loss_pct'] - 0.1)
            adapted_config['take_profit_pct'] = max(0.8, adapted_config['take_profit_pct'] - 0.2)
            adapted_config['max_daily_trades'] = max(2, adapted_config['max_daily_trades'] - 1)
        
        # Trending market adjustments
        if 'trending_market' in regime:
            adapted_config['min_signal_score'] = max(0.15, adapted_config['min_signal_score'] - 0.1)
            adapted_config['max_daily_trades'] = min(15, adapted_config['max_daily_trades'] + 3)
            adapted_config['take_profit_pct'] = adapted_config['take_profit_pct'] + 0.5
        
        # Sideways market adjustments
        elif 'sideways_market' in regime:
            adapted_config['min_signal_score'] = min(0.6, adapted_config['min_signal_score'] + 0.1)
            adapted_config['stop_loss_pct'] = max(0.4, adapted_config['stop_loss_pct'] - 0.2)
            adapted_config['take_profit_pct'] = max(0.8, adapted_config['take_profit_pct'] - 0.3)
            adapted_config['max_daily_trades'] = max(2, adapted_config['max_daily_trades'] - 2)
        
        # Extreme volatility protection
        if volatility > 1.5:  # Very high volatility
            adapted_config['min_signal_score'] = max(0.4, adapted_config['min_signal_score'])
            adapted_config['stop_loss_pct'] = min(1.5, adapted_config['stop_loss_pct'] + 0.3)
        
        return adapted_config
    
    def get_optimized_config(self, config_name="ultra_responsive"):
        """Get a specific optimized configuration"""
        base = self.configs["base_config"].copy()
        specific = self.configs["configurations"][config_name].copy()
        
        # Merge configurations
        base.update(specific)
        return base

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

async def test_dynamic_strategy():
    """Test the dynamic aggressive strategy"""
    
    print("=" * 80)
    print("DYNAMIC AGGRESSIVE STRATEGY TESTING")
    print("Real-time parameter adaptation based on market conditions")
    print("=" * 80)
    
    # Initialize dynamic strategy
    dynamic_strategy = DynamicAggressiveStrategy()
    
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Get recent data for analysis
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)  # 2 weeks for analysis
        
        print(f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Test multiple intervals
        intervals = ['1h', '4h']
        results = []
        
        for interval in intervals:
            print(f"\n{'='*60}")
            print(f"TESTING {interval.upper()} INTERVAL WITH DYNAMIC ADAPTATION")
            print(f"{'='*60}")
            
            try:
                # Fetch historical data
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
                
                print(f"📊 Data points: {len(data_list)}")
                
                if len(data_list) < 100:
                    print(f"⚠️ Insufficient data for market analysis")
                    continue
                
                # Analyze market conditions
                print("\n🔍 MARKET ANALYSIS:")
                market_analysis = dynamic_strategy.analyze_market_conditions(data_list)
                
                print(f"   Volatility: {market_analysis['volatility']:.3f}")
                print(f"   Trend Strength: {market_analysis['trend_strength']:+.2f}%")
                print(f"   Volume Trend: {market_analysis['volume_trend']:.2f}x")
                print(f"   Market Regime: {', '.join(market_analysis['regime'])}")
                print(f"   Current Price: ${market_analysis['current_price']:,.2f}")
                
                # Test different base configurations
                config_names = ['ultra_responsive', 'balanced_aggressive', 'quality_focused']
                
                for config_name in config_names:
                    if config_name in dynamic_strategy.configs["configurations"]:
                        print(f"\n🔄 Testing: {config_name.replace('_', ' ').title()}")
                        print("-" * 50)
                        
                        try:
                            # Get base optimized config
                            base_config = dynamic_strategy.get_optimized_config(config_name)
                            
                            # Adapt parameters based on market conditions
                            adapted_config = dynamic_strategy.adapt_parameters(base_config, market_analysis)
                            
                            print(f"   Original Config:")
                            print(f"      Min Signal Score: {base_config['min_signal_score']}")
                            print(f"      Volume Multiplier: {base_config['volume_multiplier']}")
                            print(f"      Stop Loss: {base_config['stop_loss_pct']}%")
                            print(f"      Take Profit: {base_config['take_profit_pct']}%")
                            print(f"      Max Daily Trades: {base_config['max_daily_trades']}")
                            
                            print(f"   Adapted Config:")
                            print(f"      Min Signal Score: {adapted_config['min_signal_score']}")
                            print(f"      Volume Multiplier: {adapted_config['volume_multiplier']}")
                            print(f"      Stop Loss: {adapted_config['stop_loss_pct']}%")
                            print(f"      Take Profit: {adapted_config['take_profit_pct']}%")
                            print(f"      Max Daily Trades: {adapted_config['max_daily_trades']}")
                            
                            # Test signal generation
                            strategy = DayTradingStrategy(adapted_config)
                            market_data = {'BTCUSDT': data_list}
                            
                            signals = await strategy.generate_signals(market_data)
                            
                            print(f"   📊 Signals Generated: {len(signals)}")
                            
                            if signals:
                                avg_confidence = sum(s.confidence for s in signals) / len(signals)
                                signal_types = [s.action for s in signals]
                                
                                print(f"   🎯 Signal Types: {', '.join(set(signal_types))}")
                                print(f"   📈 Avg Confidence: {avg_confidence:.3f}")
                                
                                # Calculate frequency
                                days = (end_date - start_date).days
                                signals_per_day = len(signals) / days if days > 0 else 0
                                print(f"   📊 Frequency: {signals_per_day:.2f} signals/day")
                                
                                # Store results
                                results.append({
                                    'interval': interval,
                                    'config_name': config_name,
                                    'market_regime': market_analysis['regime'],
                                    'signals': len(signals),
                                    'avg_confidence': avg_confidence,
                                    'signals_per_day': signals_per_day,
                                    'adapted_params': {
                                        'min_signal_score': adapted_config['min_signal_score'],
                                        'volume_multiplier': adapted_config['volume_multiplier'],
                                        'stop_loss_pct': adapted_config['stop_loss_pct'],
                                        'take_profit_pct': adapted_config['take_profit_pct'],
                                        'max_daily_trades': adapted_config['max_daily_trades']
                                    }
                                })
                            else:
                                print(f"   ⚠️ No signals generated")
                        
                        except Exception as e:
                            print(f"   ❌ Error: {e}")
                
            except Exception as e:
                print(f"❌ Error with {interval}: {e}")
        
        # Summary
        if results:
            print(f"\n{'='*80}")
            print("DYNAMIC STRATEGY RESULTS SUMMARY")
            print(f"{'='*80}")
            
            successful_results = [r for r in results if r['signals'] > 0]
            
            if successful_results:
                print(f"✅ Successful configurations: {len(successful_results)}/{len(results)}")
                
                print(f"\n📊 PERFORMANCE TABLE:")
                print(f"{'Config':<20} {'Interval':<8} {'Signals':<8} {'Conf':<6} {'S/Day':<6} {'Score':<6} {'Vol':<6}")
                print("-" * 75)
                
                for result in successful_results:
                    params = result['adapted_params']
                    print(f"{result['config_name']:<20} "
                          f"{result['interval']:<8} "
                          f"{result['signals']:<8} "
                          f"{result['avg_confidence']:<6.3f} "
                          f"{result['signals_per_day']:<6.2f} "
                          f"{params['min_signal_score']:<6.2f} "
                          f"{params['volume_multiplier']:<6.2f}")
                
                # Best performing configuration
                best = max(successful_results, key=lambda x: x['signals_per_day'] * x['avg_confidence'])
                
                print(f"\n🏆 BEST ADAPTIVE CONFIGURATION:")
                print(f"   Config: {best['config_name']}")
                print(f"   Interval: {best['interval']}")
                print(f"   Market Regime: {', '.join(best['market_regime'])}")
                print(f"   Signals: {best['signals']}")
                print(f"   Avg Confidence: {best['avg_confidence']:.3f}")
                print(f"   Frequency: {best['signals_per_day']:.2f} signals/day")
                print(f"   Adapted Parameters:")
                params = best['adapted_params']
                for param, value in params.items():
                    print(f"      {param}: {value}")
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"dynamic_aggressive_results_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump({
                        'test_info': {
                            'start_date': start_date.isoformat(),
                            'end_date': end_date.isoformat(),
                            'market_analysis': market_analysis
                        },
                        'results': results,
                        'best_config': best,
                        'summary': {
                            'successful_tests': len(successful_results),
                            'total_tests': len(results),
                            'avg_signals_per_day': sum(r['signals_per_day'] for r in successful_results) / len(successful_results),
                            'avg_confidence': sum(r['avg_confidence'] for r in successful_results) / len(successful_results)
                        }
                    }, f, indent=2, default=str)
                
                print(f"\n💾 Results saved: {results_file}")
                
            else:
                print("❌ No successful signal generations")
        
        print(f"\n{'='*80}")
        print("🎯 DYNAMIC AGGRESSIVE STRATEGY TESTING COMPLETED!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(test_dynamic_strategy())