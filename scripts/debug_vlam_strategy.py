#!/usr/bin/env python3
"""
VLAM Strategy Component Debugging Script

This script systematically tests each component of the VLAM Consolidation Strategy
to identify where signal generation is failing and optimize parameters.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import requests
import json

from src.strategies.vlam_consolidation_strategy import VLAMConsolidationStrategy
from src.exchanges.binance_exchange import BinanceBacktestingExchange
from src.data.historical_manager import HistoricalDataManager

async def debug_vlam_strategy():
    """Main debugging function"""
    print("🔍 VLAM STRATEGY COMPREHENSIVE DEBUGGING")
    print("=" * 60)
    
    # Test with real market data
    await test_with_real_data()
    
    # Test with synthetic data
    await test_with_synthetic_data()
    
    # Parameter sensitivity analysis
    await parameter_sensitivity_analysis()

async def test_with_real_data():
    """Test strategy components with real BTCUSDT data"""
    print("\n📊 TESTING WITH REAL MARKET DATA")
    print("-" * 40)
    
    try:
        # Initialize exchange and get real data
        exchange = BinanceBacktestingExchange()
        await exchange.connect()
        data_manager = HistoricalDataManager(exchange)
        
        # Get 1 week of 1h data for analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"📅 Fetching data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        historical_data = await data_manager.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_date=start_date,
            end_date=end_date
        )
        
        if historical_data is None or historical_data.empty:
            print("❌ No historical data received")
            return
            
        print(f"✅ Data received: {len(historical_data)} bars")
        print(f"   Price range: ${historical_data['low'].min():.0f} - ${historical_data['high'].max():.0f}")
        print(f"   Volume range: {historical_data['volume'].min():.0f} - {historical_data['volume'].max():.0f}")
        
        # Test strategy components
        await debug_strategy_components(historical_data, "Real Market Data")
        
        await exchange.disconnect()
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        import traceback
        traceback.print_exc()

async def test_with_synthetic_data():
    """Test strategy components with synthetic data designed to trigger signals"""
    print("\n🧪 TESTING WITH SYNTHETIC DATA")
    print("-" * 35)
    
    # Create perfect consolidation + spike + reversion pattern
    synthetic_data = create_perfect_pattern()
    await debug_strategy_components(synthetic_data, "Synthetic Data")

def create_perfect_pattern():
    """Create synthetic data with ideal consolidation + spike + reversion pattern"""
    print("🔨 Creating perfect consolidation pattern...")
    
    base_price = 50000
    data = []
    
    # Phase 1: Pre-consolidation (15 bars) - trending toward consolidation area
    for i in range(15):
        # Gradually move toward consolidation area
        trend_factor = i / 15.0  # 0 to 1
        price = base_price - 500 + (trend_factor * 500)  # Move from 49500 to 50000
        timestamp = datetime.now() - timedelta(hours=35-i)
        data.append(create_simple_bar(timestamp, price, 1000))
    
    # Phase 2: Clean consolidation (8 bars) - very tight range
    consolidation_high = base_price + 100  # 50100
    consolidation_low = base_price - 100   # 49900
    
    for i in range(8):
        # Alternate between support and resistance
        if i % 2 == 0:
            price = consolidation_low + 25   # Near support at 49925
        else:
            price = consolidation_high - 25  # Near resistance at 50075
            
        timestamp = datetime.now() - timedelta(hours=16-i)
        data.append(create_simple_bar(timestamp, price, 1000))
    
    # Phase 3: Clear spike breakout (1 bar) - breaking above consolidation
    spike_price = consolidation_high + 600  # 50700 - clear breakout
    timestamp = datetime.now() - timedelta(hours=7)
    spike_bar = {
        'timestamp': timestamp,
        'open': consolidation_high - 25,  # 50075
        'high': spike_price,             # 50700
        'low': consolidation_high - 50,   # 50050
        'close': spike_price - 100,      # 50600
        'volume': 2500  # High volume spike
    }
    data.append(spike_bar)
    
    # Phase 4: Pullback bars (6 bars) - reversion back toward consolidation
    current_price = spike_bar['close']  # Start at 50600
    for i in range(6):
        # Gradual decline back toward consolidation
        decline = (i + 1) * 80  # 80, 160, 240, 320, 400, 480
        price = current_price - decline
        timestamp = datetime.now() - timedelta(hours=6-i)
        data.append(create_simple_bar(timestamp, price, 1200))
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Synthetic pattern created: {len(df)} bars")
    print(f"   Consolidation: ${consolidation_low:.0f} - ${consolidation_high:.0f}")
    print(f"   Spike high: ${spike_price:.0f}")
    print(f"   Current price: ${df['close'].iloc[-1]:.0f}")
    print(f"   Price range: ${df['low'].min():.0f} - ${df['high'].max():.0f}")
    
    return df

def create_simple_bar(timestamp, price, volume):
    """Create a simple OHLC bar with minimal noise"""
    return {
        'timestamp': timestamp,
        'open': price,
        'high': price + 25,
        'low': price - 25,
        'close': price + 10,
        'volume': volume
    }

def create_ohlcv_bar(timestamp, price, volume):
    """Create a single OHLCV bar with realistic data"""
    volatility = 0.005  # 0.5%
    open_price = price
    high_price = price * (1 + np.random.uniform(0, volatility))
    low_price = price * (1 - np.random.uniform(0, volatility))
    close_price = price + np.random.uniform(-price*volatility/2, price*volatility/2)
    
    return {
        'timestamp': timestamp,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume + np.random.uniform(-100, 200)
    }

async def debug_strategy_components(data, data_type):
    """Debug each strategy component systematically"""
    print(f"\n🔍 DEBUGGING COMPONENTS - {data_type}")
    print("-" * 50)
    
    # Create strategy with relaxed parameters
    config = {
        'symbols': ['BTCUSDT'],
        'position_size': 0.02,
        'vlam_period': 10,
        'atr_period': 10,
        'volume_period': 15,
        'consolidation_min_length': 4,
        'consolidation_max_length': 20,
        'consolidation_tolerance': 0.03,  # 3%
        'min_touches': 2,
        'spike_min_size': 1.0,
        'spike_volume_multiplier': 1.2,
        'vlam_signal_threshold': 0.3,
        'entry_timeout_bars': 8,
        'target_risk_reward': 2.0,
        'max_risk_per_trade': 0.02
    }
    
    try:
        strategy = VLAMConsolidationStrategy(config)
        
        # Step 1: Test indicator calculations
        print("1️⃣ Testing indicator calculations...")
        indicators = strategy._calculate_indicators(data)
        
        print(f"   ✅ Indicators calculated: {len(indicators)} indicators")
        print(f"   📊 ATR range: {indicators['atr'].min():.2f} - {indicators['atr'].max():.2f}")
        print(f"   📊 VLAM range: {indicators['vlam'].min():.2f} - {indicators['vlam'].max():.2f}")
        print(f"   📊 Volume ratio range: {indicators['volume_ratio'].min():.2f} - {indicators['volume_ratio'].max():.2f}")
        
        # Check for NaN values
        for name, indicator in indicators.items():
            if isinstance(indicator, pd.Series):
                nan_count = indicator.isna().sum()
                if nan_count > 0:
                    print(f"   ⚠️  {name} has {nan_count} NaN values")
        
        # Step 2: Test consolidation detection
        print("\n2️⃣ Testing consolidation detection...")
        consolidation = strategy._detect_consolidation(data, indicators)
        
        if consolidation:
            print(f"   ✅ Consolidation detected:")
            print(f"      Length: {consolidation['length']} bars")
            print(f"      Range: ${consolidation['low']:.0f} - ${consolidation['high']:.0f}")
            print(f"      Size: ${consolidation['size']:.0f} ({consolidation['size']/consolidation['mid']*100:.2f}%)")
            print(f"      Support touches: {consolidation['support_touches']}")
            print(f"      Resistance touches: {consolidation['resistance_touches']}")
        else:
            print("   ❌ No consolidation detected")
            
            # Debug why consolidation wasn't detected
            print("   🔍 Debugging consolidation detection...")
            for length in range(config['consolidation_min_length'], min(config['consolidation_max_length'], len(data))):
                recent_highs = data['high'].iloc[-length:]
                recent_lows = data['low'].iloc[-length:]
                range_high = recent_highs.max()
                range_low = recent_lows.min()
                range_size = range_high - range_low
                range_mid = (range_high + range_low) / 2
                tolerance = range_mid * config['consolidation_tolerance']
                
                if range_size <= tolerance:
                    print(f"      Length {length}: Range ${range_size:.0f} <= Tolerance ${tolerance:.0f} ✓")
                else:
                    print(f"      Length {length}: Range ${range_size:.0f} > Tolerance ${tolerance:.0f} ✗")
                    
                if length >= 8:  # Only check first few
                    break
        
        # Step 3: Test spike detection (if consolidation found)
        if consolidation:
            print("\n3️⃣ Testing spike detection...")
            spike_event = strategy._detect_spike(data, indicators, consolidation)
            
            if spike_event:
                print(f"   ✅ Spike detected:")
                print(f"      Direction: {spike_event['direction']}")
                print(f"      Strength: {spike_event['strength']:.2f}x ATR")
                print(f"      Volume ratio: {spike_event['volume_ratio']:.2f}")
                print(f"      Bars since: {spike_event['bars_since']}")
            else:
                print("   ❌ No spike detected")
                
                # Debug spike detection
                print("   🔍 Debugging spike detection...")
                atr = indicators['atr']
                volume_ratio = indicators['volume_ratio']
                
                # Check recent bars for spikes
                search_start = max(0, len(data) - 10)
                for i in range(search_start, len(data)):
                    bar = data.iloc[i]
                    current_atr = atr.iloc[i]
                    current_vol_ratio = volume_ratio.iloc[i]
                    
                    upward_spike = (bar['high'] - consolidation['high']) / current_atr
                    downward_spike = (consolidation['low'] - bar['low']) / current_atr
                    
                    print(f"      Bar {i}: Up={upward_spike:.2f}, Down={downward_spike:.2f}, Vol={current_vol_ratio:.2f}")
            
            # Step 4: Test VLAM entry signal (if spike found)
            if spike_event:
                print("\n4️⃣ Testing VLAM entry signal...")
                entry_signal = strategy._check_vlam_entry_signal(data, indicators, consolidation, spike_event)
                
                if entry_signal:
                    print(f"   ✅ Entry signal confirmed:")
                    print(f"      Action: {entry_signal['action']}")
                    print(f"      Direction: {entry_signal['direction']}")
                    print(f"      Strength: {entry_signal['strength']:.2f}")
                    print(f"      VLAM value: {entry_signal['vlam_value']:.2f}")
                else:
                    print("   ❌ No entry signal")
                    
                    # Debug entry signal
                    print("   🔍 Debugging entry signal...")
                    vlam = indicators['vlam']
                    current_vlam = vlam.iloc[-1]
                    
                    expected_direction = 'bearish' if spike_event['direction'] == 'up' else 'bullish'
                    signal_direction = 'bullish' if current_vlam > 0 else 'bearish'
                    signal_strength = abs(current_vlam)
                    
                    print(f"      Expected direction: {expected_direction}")
                    print(f"      VLAM direction: {signal_direction}")
                    print(f"      VLAM strength: {signal_strength:.2f} (threshold: {config['vlam_signal_threshold']})")
                    print(f"      Direction match: {signal_direction == expected_direction}")
                    print(f"      Strength sufficient: {signal_strength >= config['vlam_signal_threshold']}")
                    
                    # Check timeout
                    print(f"      Bars since spike: {spike_event['bars_since']} (timeout: {config['entry_timeout_bars']})")
                
                # Step 5: Test complete signal generation
                if entry_signal:
                    print("\n5️⃣ Testing complete signal generation...")
                    try:
                        signal = strategy._create_vlam_signal('BTCUSDT', data, indicators, consolidation, spike_event, entry_signal)
                        print(f"   ✅ Complete signal created:")
                        print(f"      Action: {signal.action}")
                        print(f"      Price: ${signal.price:.2f}")
                        print(f"      Stop loss: ${signal.metadata['stop_loss']:.2f}")
                        print(f"      Take profit: ${signal.metadata['take_profit']:.2f}")
                        print(f"      Risk:Reward: {signal.metadata['risk_reward_ratio']:.2f}:1")
                        print(f"      Confidence: {signal.confidence:.2f}")
                    except Exception as e:
                        print(f"   ❌ Signal creation failed: {e}")
        
        print(f"\n📋 SUMMARY - {data_type}")
        print("-" * 30)
        has_consolidation = consolidation is not None
        has_spike = consolidation and 'spike_event' in locals() and spike_event is not None
        has_entry = has_spike and 'entry_signal' in locals() and entry_signal is not None
        
        print(f"   Consolidation: {'✅' if has_consolidation else '❌'}")
        print(f"   Spike: {'✅' if has_spike else '❌'}")
        print(f"   Entry Signal: {'✅' if has_entry else '❌'}")
        print(f"   Complete Signal: {'✅' if has_entry and 'signal' in locals() else '❌'}")
        
    except Exception as e:
        print(f"❌ Error in component debugging: {e}")
        import traceback
        traceback.print_exc()

async def parameter_sensitivity_analysis():
    """Test parameter sensitivity to find optimal ranges"""
    print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    # Create test data
    test_data = create_perfect_pattern()
    
    # Test different VLAM thresholds
    await test_parameter_range(test_data, 'vlam_signal_threshold', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Test different consolidation tolerances
    await test_parameter_range(test_data, 'consolidation_tolerance', [0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Test different spike sizes
    await test_parameter_range(test_data, 'spike_min_size', [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])

async def test_parameter_range(data, param_name, values):
    """Test a range of values for a specific parameter"""
    print(f"\n📊 Testing {param_name}:")
    
    base_config = {
        'symbols': ['BTCUSDT'],
        'position_size': 0.02,
        'vlam_period': 10,
        'atr_period': 10,
        'volume_period': 15,
        'consolidation_min_length': 4,
        'consolidation_max_length': 20,
        'consolidation_tolerance': 0.03,
        'min_touches': 2,
        'spike_min_size': 1.0,
        'spike_volume_multiplier': 1.2,
        'vlam_signal_threshold': 0.3,
        'entry_timeout_bars': 8,
        'target_risk_reward': 2.0,
        'max_risk_per_trade': 0.02
    }
    
    results = []
    
    for value in values:
        config = base_config.copy()
        config[param_name] = value
        
        try:
            strategy = VLAMConsolidationStrategy(config)
            signal = await strategy.analyze_market_data('BTCUSDT', data)
            
            status = "✅ SIGNAL" if signal else "❌ No signal"
            results.append((value, status))
            print(f"   {param_name} = {value}: {status}")
            
        except Exception as e:
            print(f"   {param_name} = {value}: ❌ Error - {e}")
            results.append((value, f"Error: {e}"))
    
    # Find optimal range
    signal_values = [v for v, status in results if "SIGNAL" in status]
    if signal_values:
        print(f"   🎯 Signal-generating range: {min(signal_values)} - {max(signal_values)}")
    else:
        print(f"   ⚠️  No values generated signals")

if __name__ == "__main__":
    asyncio.run(debug_vlam_strategy())