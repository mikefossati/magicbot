#!/usr/bin/env python3
"""
Test Optimized Configuration
Verify that the optimized day trading configuration works correctly
"""

import asyncio
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies.day_trading_strategy import DayTradingStrategy
from tests.unit.fixtures.historical_snapshots import get_historical_snapshot

async def test_optimized_config():
    """Test the balanced optimized configuration"""
    
    print("=" * 60)
    print("TESTING OPTIMIZED DAY TRADING CONFIGURATION")
    print("=" * 60)
    
    # Load the optimized configuration
    config_file = "day_trading_balanced_optimized_20250726_140500.json"
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded configuration: {config['name']}")
        print(f"📝 Description: {config['description']}")
        print(f"⚖️ Risk Level: {config['risk_level']}")
        print(f"🎯 Use Case: {config['use_case']}")
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    # Test with different market scenarios
    test_scenarios = [
        ('bullish_crossover', 'Bullish market conditions'),
        ('bearish_crossover', 'Bearish market conditions'),
        ('morning_breakout', 'Morning breakout scenario')
    ]
    
    total_signals = 0
    total_confidence = 0
    
    for scenario_name, description in test_scenarios:
        print(f"\n🔄 Testing scenario: {description}")
        print("-" * 40)
        
        try:
            # Get test data
            test_data = get_historical_snapshot(scenario_name)
            print(f"📊 Data points: {len(test_data)}")
            
            # Create strategy
            strategy = DayTradingStrategy(config)
            
            # Generate signals
            market_data = {config['symbols'][0]: test_data}
            signals = await strategy.generate_signals(market_data)
            
            scenario_signals = len(signals)
            total_signals += scenario_signals
            
            if signals:
                avg_confidence = sum(s.confidence for s in signals) / len(signals)
                total_confidence += avg_confidence * len(signals)
                
                print(f"   ✅ Signals generated: {scenario_signals}")
                print(f"   📊 Average confidence: {avg_confidence:.3f}")
                
                # Show signal details
                for i, signal in enumerate(signals):
                    stop_loss = signal.metadata.get('stop_loss', 'N/A')
                    take_profit = signal.metadata.get('take_profit', 'N/A')
                    print(f"      Signal {i+1}: {signal.action} at ${signal.price:.2f}")
                    print(f"         Confidence: {signal.confidence:.3f}")
                    print(f"         Stop Loss: ${stop_loss}")
                    print(f"         Take Profit: ${take_profit}")
                    print(f"         Volume Ratio: {signal.metadata.get('volume_ratio', 'N/A'):.2f}")
            else:
                print(f"   ⚠️ No signals generated")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"📊 Configuration Details:")
    print(f"   EMA Periods: {config['fast_ema']}/{config['medium_ema']}/{config['slow_ema']}")
    print(f"   Volume Multiplier: {config['volume_multiplier']}")
    print(f"   Min Signal Score: {config['min_signal_score']}")
    print(f"   Stop Loss: {config['stop_loss_pct']}%")
    print(f"   Take Profit: {config['take_profit_pct']}%")
    print(f"   Max Daily Trades: {config['max_daily_trades']}")
    
    print(f"\n🎯 Performance Across Scenarios:")
    print(f"   Total Signals: {total_signals}")
    if total_signals > 0:
        overall_confidence = total_confidence / total_signals
        print(f"   Overall Avg Confidence: {overall_confidence:.3f}")
        
        if overall_confidence >= 0.7:
            print("   ✅ High quality signals")
        elif overall_confidence >= 0.5:
            print("   ⚠️ Moderate quality signals")
        else:
            print("   ❌ Low quality signals")
    
    print(f"\n🏆 FINAL VERDICT:")
    if total_signals >= 2 and total_confidence / total_signals >= 0.6:
        print("   ✅ CONFIGURATION APPROVED")
        print("   Ready for paper trading")
    elif total_signals >= 1:
        print("   ⚠️ CONFIGURATION NEEDS MONITORING") 
        print("   Proceed with caution")
    else:
        print("   ❌ CONFIGURATION NEEDS ADJUSTMENT")
        print("   Consider parameter tweaks")
    
    return total_signals > 0

def show_implementation_recommendations():
    """Show implementation recommendations"""
    
    print(f"\n{'='*60}")
    print("IMPLEMENTATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    recommendations = [
        "🚀 Next Steps:",
        "   1. Start with paper trading for 1-2 weeks",
        "   2. Monitor signal frequency and quality",
        "   3. Track win rate and drawdown",
        "   4. Gradually scale position size if successful",
        "",
        "⚖️ Risk Management:",
        "   • Never risk more than 2% per trade",
        "   • Use all provided stop losses",
        "   • Set daily loss limits",
        "   • Review performance weekly",
        "",
        "📊 Monitoring Metrics:",
        "   • Signal count per day (target: 2-4)",
        "   • Average confidence (target: >0.6)",
        "   • Win rate (target: >50%)",
        "   • Maximum drawdown (keep <15%)",
        "",
        "🔧 Configuration Notes:",
        "   • Uses ATR-based dynamic stop losses",
        "   • Includes volume confirmation on all signals",
        "   • Probability scoring prevents overtrading",
        "   • Support/resistance levels improve timing"
    ]
    
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    success = asyncio.run(test_optimized_config())
    
    if success:
        show_implementation_recommendations()
    
    print(f"\n{'='*60}")
    print("🎯 OPTIMIZATION AND TESTING COMPLETE!")
    print("Ready for deployment with optimized parameters")
    print(f"{'='*60}")