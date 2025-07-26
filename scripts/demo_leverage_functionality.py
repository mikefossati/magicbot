#!/usr/bin/env python3
"""
Demo Leverage Functionality
Demonstrate how leverage affects position sizing and risk management
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.strategies import create_strategy

def create_sample_data():
    """Create sample market data for testing"""
    # Generate 100 data points with realistic price movement
    dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=100, freq='15min')
    base_price = 50000
    
    # Generate price data with some volatility
    price_changes = np.random.normal(0, 0.002, 100)  # 0.2% average volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    data = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    # Set timestamp as index
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    
    return data

def demo_leverage_functionality():
    """Demonstrate leverage functionality with different configurations"""
    
    print("=" * 80)
    print("LEVERAGE FUNCTIONALITY DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("\n📊 Creating sample market data...")
    sample_data = create_sample_data()
    print(f"✅ Generated {len(sample_data)} data points")
    
    # Test different leverage configurations
    leverage_configs = [
        {"name": "No Leverage", "use_leverage": False, "leverage": 1.0, "risk_factor": 1.0},
        {"name": "2x Leverage", "use_leverage": True, "leverage": 2.0, "risk_factor": 0.7},
        {"name": "3x Leverage", "use_leverage": True, "leverage": 3.0, "risk_factor": 0.6},
        {"name": "5x Leverage", "use_leverage": True, "leverage": 5.0, "risk_factor": 0.4},
    ]
    
    current_price = sample_data['close'].iloc[-1]
    print(f"\n💰 Current Price: ${current_price:.2f}")
    
    print(f"\n{'='*80}")
    print("LEVERAGE IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    results = []
    
    for config in leverage_configs:
        print(f"\n🔧 {config['name']} Configuration:")
        print("-" * 40)
        
        # Create strategy with specific leverage settings
        strategy_config = {
            'symbols': ['BTCUSDT'],
            'fast_ema': 8,
            'medium_ema': 21,
            'slow_ema': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'rsi_neutral_high': 60,
            'rsi_neutral_low': 40,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_period': 20,
            'volume_multiplier': 1.5,
            'pivot_period': 10,
            'support_resistance_threshold': 0.2,
            'stop_loss_pct': 1.5,
            'take_profit_pct': 2.5,
            'trailing_stop_pct': 1.0,
            'max_daily_trades': 3,
            'session_start': "00:00",
            'session_end': "23:59",
            'position_size': 0.02,  # 2% base position
            'use_leverage': config['use_leverage'],
            'leverage': config['leverage'],
            'max_leverage': 10.0,
            'leverage_risk_factor': config['risk_factor']
        }
        
        strategy = create_strategy('day_trading_strategy', strategy_config)
        
        # Test leverage calculations with different confidence levels
        confidence_levels = [0.5, 0.7, 0.9]
        
        base_position_float = float(strategy.position_size)  # Convert Decimal to float
        print(f"Base Position Size: {base_position_float:.3f} (2%)")
        print(f"Leverage: {strategy.leverage:.1f}x")
        print(f"Risk Factor: {strategy.leverage_risk_factor:.2f}")
        
        for confidence in confidence_levels:
            # Calculate position size
            position_size = strategy._calculate_leveraged_position_size(current_price, confidence)
            
            # Calculate stops
            atr = current_price * 0.01  # Simulate 1% ATR
            stops = strategy._calculate_leveraged_stops(current_price, 'BUY', atr)
            
            # Calculate potential returns
            leverage_multiplier = strategy.leverage if strategy.use_leverage else 1.0
            
            potential_profit_pct = (stops['take_profit'] - current_price) / current_price * 100
            potential_loss_pct = (current_price - stops['stop_loss']) / current_price * 100
            
            # With leverage
            leveraged_profit = potential_profit_pct * leverage_multiplier
            leveraged_loss = potential_loss_pct * leverage_multiplier
            
            print(f"\n  📈 Confidence {confidence:.0%}:")
            print(f"    Position Size: {position_size:.4f} ({position_size/base_position_float:.2f}x base)")
            print(f"    Stop Loss: ${stops['stop_loss']:.2f} (-{potential_loss_pct:.2f}%)")
            print(f"    Take Profit: ${stops['take_profit']:.2f} (+{potential_profit_pct:.2f}%)")
            print(f"    Leveraged P&L: +{leveraged_profit:.2f}% / -{leveraged_loss:.2f}%")
            print(f"    Risk/Reward: 1:{leveraged_profit/leveraged_loss:.2f}")
        
        # Store summary results
        avg_position = strategy._calculate_leveraged_position_size(current_price, 0.7)  # 70% confidence
        stops = strategy._calculate_leveraged_stops(current_price, 'BUY', current_price * 0.01)
        leverage_multiplier = strategy.leverage if strategy.use_leverage else 1.0
        
        profit_pct = (stops['take_profit'] - current_price) / current_price * 100 * leverage_multiplier
        loss_pct = (current_price - stops['stop_loss']) / current_price * 100 * leverage_multiplier
        
        results.append({
            'name': config['name'],
            'leverage': config['leverage'],
            'position_multiplier': avg_position / base_position_float,
            'profit_potential': profit_pct,
            'loss_risk': loss_pct,
            'risk_reward': profit_pct / loss_pct if loss_pct > 0 else 0
        })
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("LEVERAGE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Strategy':<15} {'Leverage':<10} {'Pos Multi':<10} {'Profit%':<10} {'Loss%':<10} {'R/R':<8}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['name']:<15} "
              f"{result['leverage']:<10.1f} "
              f"{result['position_multiplier']:<10.2f} "
              f"{result['profit_potential']:<10.2f} "
              f"{result['loss_risk']:<10.2f} "
              f"{result['risk_reward']:<8.2f}")
    
    # Risk analysis
    print(f"\n🎯 LEVERAGE RISK ANALYSIS:")
    print("-" * 50)
    
    base_case = results[0]  # No leverage
    
    for result in results[1:]:  # Skip base case
        leverage_factor = result['leverage']
        position_increase = result['position_multiplier'] / base_case['position_multiplier']
        profit_increase = result['profit_potential'] / base_case['profit_potential']
        loss_increase = result['loss_risk'] / base_case['loss_risk']
        
        print(f"\n{result['name']}:")
        print(f"  • {leverage_factor:.1f}x leverage increases position exposure by {position_increase:.2f}x")
        print(f"  • Potential profits increase by {profit_increase:.2f}x")
        print(f"  • Potential losses increase by {loss_increase:.2f}x")
        
        if loss_increase > profit_increase:
            print(f"  ⚠️  Risk increases faster than reward!")
        else:
            print(f"  ✅ Balanced risk/reward scaling")
    
    # Recommendations
    print(f"\n🏆 LEVERAGE RECOMMENDATIONS:")
    print("-" * 50)
    
    print("💡 Key Insights:")
    print("  • Higher leverage amplifies both gains and losses")
    print("  • Position sizes are reduced with leverage to manage risk")
    print("  • Stop losses become tighter with higher leverage")
    print("  • Confidence-based position sizing provides additional risk control")
    
    print("\n📋 Best Practices:")
    print("  • Start with 2x-3x leverage for experience")
    print("  • Use tight stop losses with leveraged positions")
    print("  • Reduce position sizes when using leverage")
    print("  • Monitor margin requirements carefully")
    print("  • Consider market volatility when choosing leverage")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    demo_leverage_functionality()