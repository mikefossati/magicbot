#!/bin/bash

# Simple script to run optimized backtests
echo "=========================================="
echo "OPTIMIZED AGGRESSIVE STRATEGY BACKTESTS"
echo "=========================================="

if [ $# -eq 0 ]; then
    echo "Usage: ./run_backtest.sh [config] [days] [interval]"
    echo ""
    echo "Available configurations:"
    echo "  ultra_responsive  - Maximum signal generation (EMA 3/8/21, score 0.25)"
    echo "  balanced         - Balanced approach (EMA 5/13/34, score 0.4)"
    echo "  responsive       - Fast response (EMA 3/8/21, score 0.35)"
    echo "  quality          - Quality focused (EMA 8/18/45, score 0.45)"
    echo ""
    echo "Examples:"
    echo "  ./run_backtest.sh balanced 21 1h"
    echo "  ./run_backtest.sh ultra_responsive 30 4h"
    echo "  ./run_backtest.sh quality 14 1h"
    echo ""
    echo "Running default: balanced config, 14 days, 1h interval"
    python scripts/run_optimized_backtest.py balanced 14 1h
else
    CONFIG=${1:-balanced}
    DAYS=${2:-14}
    INTERVAL=${3:-1h}
    
    echo "Configuration: $CONFIG"
    echo "Days: $DAYS"
    echo "Interval: $INTERVAL"
    echo ""
    
    python scripts/run_optimized_backtest.py $CONFIG $DAYS $INTERVAL
fi