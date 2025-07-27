#!/usr/bin/env python3
"""
Test script to verify the optimization frontend is sending correct parameters.

This script simulates frontend requests to test parameter validation.
"""

import requests
import json
from datetime import datetime, timedelta

def test_optimization_api():
    """Test optimization API with various parameter combinations"""
    
    base_url = "http://localhost:8000/api/v1/optimization"
    
    # Test 1: Valid request with correct strategy name
    print("🧪 Test 1: Valid request with day_trading_strategy")
    valid_request = {
        "strategy_name": "day_trading_strategy",
        "optimizer_type": "grid_search",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "objectives": ["maximize_return"],
        "validation_enabled": True
    }
    
    response = requests.post(f"{base_url}/jobs", json=valid_request)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: Job ID {result['job_id']}")
    else:
        print(f"❌ Error: {response.text}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Invalid strategy name (old frontend format)
    print("🧪 Test 2: Invalid request with old strategy name format")
    invalid_request = {
        "strategy_name": "day_trading",  # Old format from frontend
        "optimizer_type": "grid_search",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31"
    }
    
    response = requests.post(f"{base_url}/jobs", json=invalid_request)
    print(f"Status: {response.status_code}")
    if response.status_code == 422:
        error = response.json()
        print(f"✅ Expected validation error: {error['detail']}")
    else:
        print(f"❌ Unexpected response: {response.text}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Test all available strategies
    print("🧪 Test 3: Testing all available strategy names")
    available_strategies = [
        "ma_crossover",
        "rsi_strategy", 
        "bollinger_bands",
        "breakout_strategy",
        "macd_strategy",
        "momentum_strategy",
        "stochastic_strategy",
        "mean_reversion_rsi",
        "ema_scalping_strategy",
        "day_trading_strategy",
        "vlam_consolidation_strategy",
        "momentum_trading_strategy"
    ]
    
    for strategy in available_strategies:
        test_request = {
            "strategy_name": strategy,
            "optimizer_type": "grid_search",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07"  # Short period for testing
        }
        
        response = requests.post(f"{base_url}/jobs", json=test_request)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {strategy}: Job ID {result['job_id']}")
        else:
            print(f"❌ {strategy}: {response.status_code} - {response.text}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Check recent jobs
    print("🧪 Test 4: Checking recent optimization jobs")
    response = requests.get(f"{base_url}/runs?limit=5")
    if response.status_code == 200:
        runs = response.json()
        print(f"✅ Found {len(runs['runs'])} recent runs")
        for run in runs['runs']:
            print(f"  - {run['strategy_name']}: {run['status']}")
    else:
        print(f"❌ Error fetching runs: {response.text}")

if __name__ == "__main__":
    print("🚀 Testing Optimization API Frontend Parameters\n")
    test_optimization_api()
    print("\n✅ Frontend parameter testing complete!")
