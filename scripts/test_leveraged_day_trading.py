#!/usr/bin/env python3
"""
Test Leveraged Day Trading Strategy
Test the day trading strategy with different leverage settings
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchanges.binance_exchange import BinanceExchange
from src.strategies import create_strategy
from src.data.historical_manager import HistoricalDataManager
from src.backtesting.engine import BacktestEngine, BacktestConfig
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

async def test_leveraged_strategies():
    """Test different leverage configurations"""
    
    print("=" * 80)
    print("LEVERAGED DAY TRADING STRATEGY TEST")
    print("=" * 80)
    
    # Initialize exchange
    exchange = BinanceExchange()
    await exchange.connect()
    
    try:
        # Fetch test data
        data_manager = HistoricalDataManager(exchange)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print("📊 Fetching historical data...")
        historical_data = await data_manager.get_multiple_symbols_data(
            symbols=['BTCUSDT'],
            interval='15m',
            start_date=start_date,
            end_date=end_date
        )
        
        if 'BTCUSDT' not in historical_data or len(historical_data['BTCUSDT']) < 100:
            print("❌ Insufficient data for testing")
            return
        
        print(f"✅ Data loaded: {len(historical_data['BTCUSDT'])} records")
        
        # Test configurations: No leverage, 2x, 3x, 5x
        test_configs = [
            {
                'name': 'No Leverage (1x)',
                'use_leverage': False,
                'leverage': 1.0,
                'leverage_risk_factor': 1.0
            },
            {
                'name': 'Conservative Leverage (2x)',
                'use_leverage': True,
                'leverage': 2.0,
                'leverage_risk_factor': 0.7
            },
            {
                'name': 'Moderate Leverage (3x)',
                'use_leverage': True,
                'leverage': 3.0,
                'leverage_risk_factor': 0.6
            },
            {
                'name': 'Aggressive Leverage (5x)',
                'use_leverage': True,
                'leverage': 5.0,
                'leverage_risk_factor': 0.4
            }
        ]
        
        results = []
        
        for test_config in test_configs:
            print(f"\n{'='*60}")
            print(f"TESTING: {test_config['name']}")
            print(f"{'='*60}")
            
            # Create strategy configuration
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
                'position_size': 0.02,
                # Leverage settings from test config
                'use_leverage': test_config['use_leverage'],
                'leverage': test_config['leverage'],
                'max_leverage': 10.0,
                'leverage_risk_factor': test_config['leverage_risk_factor']
            }
            
            # Create strategy
            strategy = create_strategy('day_trading_strategy', strategy_config)
            
            # Display strategy info
            info = strategy.get_strategy_info()
            risk_params = strategy.get_risk_parameters()
            
            print(f"\n📋 STRATEGY CONFIGURATION:")
            print(f"   Leverage Enabled: {info['leverage_enabled']}")
            print(f"   Leverage Ratio: {info['leverage_ratio']:.1f}x")
            print(f"   Base Position Size: {risk_params['position_size']:.3f}")
            print(f"   Leverage Risk Factor: {risk_params['leverage_risk_factor']:.2f}")
            
            # Test signal generation on sample data
            print(f"\n🧪 Testing signal generation...")
            test_data = historical_data['BTCUSDT'].tail(100)
            signal = await strategy.analyze_market_data('BTCUSDT', test_data)
            
            if signal:
                print(f"✅ Signal Generated:")
                print(f"   Action: {signal.action}")
                print(f"   Price: ${signal.price:.2f}")
                print(f"   Position Size: {signal.quantity:.4f}")
                print(f"   Confidence: {signal.confidence:.2%}")
                print(f"   Stop Loss: ${signal.stop_loss:.2f}")
                print(f"   Take Profit: ${signal.take_profit:.2f}")
                
                # Show leverage-specific metadata
                if 'leverage' in signal.metadata:
                    metadata = signal.metadata
                    print(f"   Leverage Used: {metadata['leverage']:.1f}x")
                    print(f"   Base Position: {metadata['base_position_size']:.4f}")
                    print(f"   Actual Position: {metadata['actual_position_size']:.4f}")
                    
                    # Calculate potential returns
                    leverage_multiplier = metadata['leverage'] if metadata['leveraged_position'] else 1.0
                    potential_profit = (signal.take_profit - signal.price) / signal.price * 100 * leverage_multiplier
                    potential_loss = (signal.price - signal.stop_loss) / signal.price * 100 * leverage_multiplier
                    
                    print(f"\n💰 POTENTIAL OUTCOMES (with {leverage_multiplier:.1f}x leverage):")
                    print(f"   Max Profit: {potential_profit:.2f}%")
                    print(f"   Max Loss: -{potential_loss:.2f}%")
                    print(f"   Risk/Reward Ratio: 1:{potential_profit/potential_loss:.2f}")
            else:
                print("⚪ No signal generated (normal)")
            
            # Quick backtest
            print(f"\n🔄 Running quick backtest...")
            
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=strategy_config['position_size']
            )
            
            engine = BacktestEngine(backtest_config)
            backtest_results = await engine.run_backtest(
                strategy=strategy,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Analyze results
            capital = backtest_results['capital']
            trades = backtest_results['trades']
            risk_metrics = backtest_results['risk_metrics']
            
            print(f"\n📈 BACKTEST RESULTS:")
            print(f"   Total Return: {capital['total_return_pct']:.2f}%")
            print(f"   Total Trades: {trades['total']}")
            if trades['total'] > 0:
                print(f"   Win Rate: {trades.get('win_rate_pct', 0):.1f}%")
                print(f"   Profit Factor: {trades.get('profit_factor', 0):.2f}")
            print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
            
            # Store results
            results.append({
                'name': test_config['name'],
                'leverage': test_config['leverage'],
                'return_pct': capital['total_return_pct'],
                'trades': trades['total'],
                'win_rate': trades.get('win_rate_pct', 0),
                'sharpe': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown_pct'],
                'risk_adjusted_return': capital['total_return_pct'] / max(risk_metrics['max_drawdown_pct'], 1)
            })
        
        # Compare results
        print(f"\n{'='*80}")
        print("LEVERAGE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Strategy':<25} {'Leverage':<10} {'Return%':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7} {'MaxDD%':<7} {'R/DD':<6}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['name']:<25} "
                  f"{result['leverage']:<10.1f} "
                  f"{result['return_pct']:<8.2f} "
                  f"{result['trades']:<7} "
                  f"{result['win_rate']:<6.1f} "
                  f"{result['sharpe']:<7.2f} "
                  f"{result['max_drawdown']:<7.2f} "
                  f"{result['risk_adjusted_return']:<6.2f}")
        
        # Recommendations
        print(f"\n🎯 LEVERAGE RECOMMENDATIONS:")
        print("-" * 50)
        
        best_return = max(results, key=lambda x: x['return_pct'])
        best_risk_adjusted = max(results, key=lambda x: x['risk_adjusted_return'])
        best_sharpe = max(results, key=lambda x: x['sharpe'])
        
        print(f"Best Raw Return: {best_return['name']} ({best_return['return_pct']:.2f}%)")
        print(f"Best Risk-Adjusted: {best_risk_adjusted['name']} (R/DD: {best_risk_adjusted['risk_adjusted_return']:.2f})")
        print(f"Best Sharpe Ratio: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        
        # Overall assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        if best_risk_adjusted['leverage'] <= 2.0:
            print("✅ CONSERVATIVE APPROACH RECOMMENDED")
            print("   Lower leverage shows better risk-adjusted returns")
        elif best_risk_adjusted['leverage'] <= 3.0:
            print("⚠️ MODERATE LEVERAGE ACCEPTABLE")
            print("   Moderate leverage provides good balance")
        else:
            print("⚠️ HIGH LEVERAGE RISKY")
            print("   High leverage increases returns but significantly increases risk")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        logger.error("Error in leveraged strategy test", error=str(e))
        raise
    
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    asyncio.run(test_leveraged_strategies())