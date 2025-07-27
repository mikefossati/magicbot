#!/usr/bin/env python3
import requests
import json
import time

def test_config(params, name):
    print(f'🧪 Testing {name}...')
    req = {
        'strategy_name': 'vlam_consolidation_strategy',
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'start_date': '2024-12-01T00:00:00',
        'end_date': '2025-01-15T00:00:00',
        'initial_capital': 10000.0,
        'parameters': params
    }
    
    resp = requests.post('http://localhost:8000/api/v1/backtesting/run', json=req)
    if resp.status_code != 200:
        print(f'  ❌ Failed: {resp.text}')
        return None
        
    session_id = resp.json()['session_id']
    
    # Wait for completion
    for _ in range(60):
        time.sleep(1)
        status = requests.get(f'http://localhost:8000/api/v1/backtesting/status/{session_id}')
        if status.status_code == 200:
            s = status.json()
            if s['status'] == 'completed':
                results = requests.get(f'http://localhost:8000/api/v1/backtesting/results/{session_id}')
                if results.status_code == 200:
                    r = results.json()
                    ret = r.get('capital', {}).get('total_return_pct', 0)
                    trades = r.get('trades', {}).get('total', 0)
                    win_rate = r.get('trades', {}).get('win_rate_pct', 0)
                    sharpe = r.get('risk_metrics', {}).get('sharpe_ratio', 0)
                    print(f'  📊 Return: {ret:.2f}%, Trades: {trades}, Win Rate: {win_rate:.1f}%, Sharpe: {sharpe:.2f}')
                    return {'name': name, 'return': ret, 'trades': trades, 'win_rate': win_rate, 'sharpe': sharpe}
            elif s['status'] == 'failed':
                print(f'  ❌ Failed: {s.get("message", "")}')
                return None
    
    print('  ⏱️ Timed out')
    return None

def main():
    print('🚀 MANUAL VLAM PARAMETER OPTIMIZATION')
    print('='*50)
    
    # Base configuration
    base = {
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

    configs = [
        (base.copy(), 'Original Config'),
        ({**base, 'vlam_signal_threshold': 0.25}, 'Lower VLAM Threshold (0.25)'),
        ({**base, 'spike_min_size': 0.8}, 'Lower Spike Size (0.8)'),
        ({**base, 'consolidation_tolerance': 0.04}, 'Looser Consolidation (4%)'),
        ({**base, 'target_risk_reward': 1.5}, 'Lower Risk:Reward (1.5:1)'),
    ]

    results = []
    for params, name in configs:
        result = test_config(params, name)
        if result:
            results.append(result)

    print('\n🏆 OPTIMIZATION RESULTS:')
    print('='*50)
    for r in sorted(results, key=lambda x: x['return'], reverse=True):
        print(f'{r["name"]}: {r["return"]:.2f}% return, {r["trades"]} trades, {r["win_rate"]:.1f}% win rate')

if __name__ == '__main__':
    main()