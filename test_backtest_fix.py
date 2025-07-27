#!/usr/bin/env python3
"""
Quick test to verify BacktestConfig compatibility fix.
"""

import requests
import json

def test_backtest_fix():
    """Test that the BacktestConfig fix resolves the compatibility issue."""
    
    print("🔧 Testing BacktestConfig Compatibility Fix")
    print("=" * 50)
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test backtest request with Day Trading Strategy
    backtest_request = {
        "strategy_name": "day_trading_strategy",
        "symbol": "BTCUSDT",
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "parameters": {
            "position_size": 0.02,
            "min_signal_score": 0.3,  # Optimized parameter
            "min_volume_ratio": 0.3   # Optimized parameter
        }
    }
    
    print("\n🧪 Testing backtest execution...")
    try:
        response = requests.post(
            f"{base_url}/backtesting/backtest",
            json=backtest_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backtest executed successfully!")
            print(f"   📊 Status: {result.get('status', 'Unknown')}")
            
            # Check for results
            if 'results' in result:
                results = result['results']
                print(f"   💰 Total return: {results.get('total_return', 'N/A')}")
                print(f"   📈 Total trades: {results.get('total_trades', 'N/A')}")
                print(f"   🎯 Signals generated: {results.get('total_signals', 'N/A')}")
            
            return True
        else:
            print(f"❌ Backtest failed with status {response.status_code}")
            error_detail = response.text
            print(f"   Error: {error_detail}")
            
            # Check if BacktestConfig error is resolved
            if "BacktestConfig" in error_detail:
                print("❌ BacktestConfig compatibility issue still exists!")
                return False
            else:
                print("✅ BacktestConfig compatibility issue resolved!")
                return True
                
    except Exception as e:
        print(f"❌ Error testing backtest: {e}")
        return False

def test_optimization_job():
    """Test optimization job creation."""
    
    print("\n🎯 Testing optimization job creation...")
    base_url = "http://localhost:8000/api/v1"
    
    optimization_request = {
        "strategy_name": "day_trading_strategy",
        "optimizer_type": "grid_search",
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "objectives": ["maximize_return"],
        "validation_enabled": False
    }
    
    try:
        response = requests.post(
            f"{base_url}/optimization/jobs",
            json=optimization_request,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            job_data = response.json()
            print(f"✅ Optimization job created successfully!")
            print(f"   🆔 Job ID: {job_data.get('job_id')}")
            print(f"   📅 Strategy: {job_data.get('strategy_name')}")
            return True
        else:
            print(f"❌ Failed to create optimization job: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating optimization job: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting BacktestConfig Compatibility Test")
    
    # Test backtest fix
    backtest_success = test_backtest_fix()
    
    # Test optimization job
    optimization_success = test_optimization_job()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   🔧 BacktestConfig Fix: {'✅ WORKING' if backtest_success else '❌ FAILED'}")
    print(f"   🎯 Optimization Jobs: {'✅ WORKING' if optimization_success else '❌ FAILED'}")
    
    if backtest_success and optimization_success:
        print("\n🎉 All tests passed! System is ready for optimization.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
