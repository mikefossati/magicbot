#!/usr/bin/env python3
"""
Multi-Strategy Backtesting Example
Demonstrates how to compare and combine multiple trading strategies
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def run_multi_strategy_example():
    """Example of multi-strategy backtesting workflow"""
    
    print("🚀 MagicBot Multi-Strategy Example")
    print("=" * 50)
    
    # Example 1: Compare all strategies
    print("\n1️⃣  COMPARING ALL STRATEGIES")
    print("Command: python scripts/compare_strategies.py --all")
    print("This will test all 4 strategies and show performance comparison")
    
    # Example 2: Compare specific strategies
    print("\n2️⃣  COMPARING SPECIFIC STRATEGIES")
    print("Command: python scripts/compare_strategies.py rsi_strategy bollinger_bands")
    print("This compares RSI vs Bollinger Bands strategies")
    
    # Example 3: Portfolio backtest
    print("\n3️⃣  MULTI-STRATEGY PORTFOLIO")
    print("Command: python scripts/run_multi_strategy_backtest.py \\")
    print("  --strategies ma_crossover rsi_strategy bollinger_bands")
    print("This runs individual backtests AND portfolio combination")
    
    # Example 4: Custom parameters
    print("\n4️⃣  CUSTOM PARAMETERS")
    print("Command: python scripts/compare_strategies.py \\")
    print("  ma_crossover breakout_strategy \\")
    print("  --symbols BTCUSDT ETHUSDT ADAUSDT \\")
    print("  --start-date 2024-01-01 \\")
    print("  --capital 50000 \\")
    print("  --interval 4h")
    
    print("\n📊 EXPECTED OUTPUT:")
    print("""
🔍 COMPARING 4 STRATEGIES
📅 Period: 2024-07-01 to 2024-10-25
💰 Initial Capital: $10,000
📊 Symbols: BTCUSDT, ETHUSDT
⏱️  Timeframe: 1h

⚡ Testing Strategy 1/4: ma_crossover
   Return: +15.2% | Trades: 67 | Win Rate: 52.2%

⚡ Testing Strategy 2/4: rsi_strategy  
   Return: +24.5% | Trades: 147 | Win Rate: 58.5%

⚡ Testing Strategy 3/4: bollinger_bands
   Return: +18.3% | Trades: 203 | Win Rate: 55.2%

⚡ Testing Strategy 4/4: breakout_strategy
   Return: +28.7% | Trades: 89 | Win Rate: 64.0%

📊 STRATEGY COMPARISON RESULTS
================================================================================
Strategy             Return %   Trades   Win %    Sharpe   Max DD %   Final $  
--------------------------------------------------------------------------------
breakout_strategy        +28.7      89    64.0      2.12      -6.8    $12,870
rsi_strategy            +24.5     147    58.5      1.85      -8.3    $12,450
bollinger_bands         +18.3     203    55.2      1.67      -9.1    $11,830
ma_crossover            +15.2      67    52.2      1.43     -12.4    $11,520

🏆 BEST PERFORMERS
----------------------------------------
💰 Highest Return:   breakout_strategy (+28.7%)
📈 Best Sharpe:      breakout_strategy (2.12)  
🎯 Best Win Rate:    breakout_strategy (64.0%)

💡 RECOMMENDATIONS
----------------------------------------
🎯 Consider combining trend-following and mean-reversion strategies
📊 Portfolio of top 2 strategies could yield ~26.6% return
    """)
    
    print("\n📁 OUTPUT FILES:")
    print("- backtest_results/strategy_comparison_YYYYMMDD_HHMMSS.json")
    print("- backtest_results/comparison_report_YYYYMMDD_HHMMSS.txt")
    
    print("\n💡 NEXT STEPS:")
    print("1. Run the comparison to see which strategies work best")
    print("2. Combine top performers in a portfolio")
    print("3. Optimize parameters for best strategies")
    print("4. Test on different timeframes and symbols")
    
    print("\n✅ Ready to start? Run one of the commands above!")

if __name__ == "__main__":
    asyncio.run(run_multi_strategy_example())