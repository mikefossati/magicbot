#!/usr/bin/env python3
"""
Scalping Strategy Parameter Optimization Script
Systematically tests different parameter combinations to find optimal settings
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import itertools
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

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

class ScalpingParameterOptimizer:
    """Optimize EMA Scalping Strategy parameters"""
    
    def __init__(self):
        self.results = []
        self.exchange = None
        self.data_manager = None
        self.historical_data = None
        
    async def initialize(self, symbols: List[str], interval: str = '5m', days_back: int = 7):
        """Initialize exchange connection and fetch data"""
        logger.info("Initializing optimizer", symbols=symbols, interval=interval, days_back=days_back)
        
        # Initialize exchange
        self.exchange = BinanceExchange()
        await self.exchange.connect()
        
        # Set up data manager
        self.data_manager = HistoricalDataManager(self.exchange)
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info("Fetching historical data...")
        self.historical_data = await self.data_manager.get_multiple_symbols_data(
            symbols=symbols,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        total_records = sum(len(df) for df in self.historical_data.values())
        logger.info("Historical data loaded", total_records=total_records)
    
    def define_parameter_space(self) -> Dict[str, List]:
        """Define the parameter space for optimization"""
        return {
            'fast_ema': [3, 5, 8, 12],
            'slow_ema': [8, 13, 21, 34],
            'signal_ema': [13, 21, 34, 55],
            'volume_multiplier': [1.2, 1.5, 2.0, 2.5],
            'rsi_overbought': [70, 75, 80],
            'rsi_oversold': [20, 25, 30],
            'stop_loss_pct': [0.3, 0.5, 0.7, 1.0],
            'take_profit_pct': [0.6, 1.0, 1.5, 2.0],
            'position_size': [0.01, 0.02, 0.03]
        }
    
    def generate_parameter_combinations(self, max_combinations: int = 500) -> List[Dict]:
        """Generate parameter combinations with constraints"""
        param_space = self.define_parameter_space()
        
        # Generate all possible combinations
        keys, values = zip(*param_space.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Filter valid combinations
        valid_combinations = []
        for combo in all_combinations:
            # Constraint: fast_ema < slow_ema < signal_ema
            if combo['fast_ema'] >= combo['slow_ema']:
                continue
            if combo['slow_ema'] >= combo['signal_ema']:
                continue
            
            # Constraint: reasonable risk/reward ratio
            risk_reward = combo['take_profit_pct'] / combo['stop_loss_pct']
            if risk_reward < 1.0 or risk_reward > 5.0:
                continue
            
            # Constraint: RSI thresholds
            if combo['rsi_oversold'] >= combo['rsi_overbought']:
                continue
            
            valid_combinations.append(combo)
        
        # Limit combinations if too many
        if len(valid_combinations) > max_combinations:
            # Use systematic sampling to get diverse parameter sets
            step = len(valid_combinations) // max_combinations
            valid_combinations = valid_combinations[::step][:max_combinations]
        
        logger.info("Parameter combinations generated", 
                   total=len(all_combinations),
                   valid=len(valid_combinations))
        
        return valid_combinations
    
    async def test_parameter_combination(self, params: Dict, symbols: List[str]) -> Dict:
        """Test a single parameter combination"""
        try:
            # Create strategy config
            strategy_config = {
                'symbols': symbols,
                **params
            }
            
            # Create strategy
            strategy = create_strategy('ema_scalping_strategy', strategy_config)
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_sizing='percentage',
                position_size=params['position_size']
            )
            
            # Run backtest
            engine = BacktestEngine(backtest_config)
            results = await engine.run_backtest(
                strategy=strategy,
                historical_data=self.historical_data,
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )
            
            # Calculate additional metrics
            total_return = results['capital']['total_return_pct']
            sharpe_ratio = results['risk_metrics']['sharpe_ratio']
            max_drawdown = results['risk_metrics']['max_drawdown_pct']
            total_trades = results['trades']['total']
            win_rate = results['trades']['win_rate_pct']
            profit_factor = results['trades']['profit_factor']
            
            # Calculate risk-adjusted return
            risk_adjusted_return = total_return / max(max_drawdown, 1.0) if max_drawdown > 0 else total_return
            
            # Calculate score (composite metric)
            score = self._calculate_composite_score(
                total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, profit_factor
            )
            
            return {
                'parameters': params,
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_trades': total_trades,
                'win_rate_pct': win_rate,
                'profit_factor': profit_factor,
                'risk_adjusted_return': risk_adjusted_return,
                'composite_score': score,
                'success': True
            }
            
        except Exception as e:
            logger.error("Parameter test failed", params=params, error=str(e))
            return {
                'parameters': params,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_composite_score(self, total_return: float, sharpe_ratio: float, 
                                 max_drawdown: float, win_rate: float, 
                                 total_trades: int, profit_factor: float) -> float:
        """Calculate composite score for parameter ranking"""
        # Normalize metrics (0-1 scale)
        return_score = max(0, min(1, (total_return + 50) / 100))  # -50% to +50% -> 0 to 1
        sharpe_score = max(0, min(1, (sharpe_ratio + 2) / 4))     # -2 to +2 -> 0 to 1
        drawdown_score = max(0, 1 - max_drawdown / 50)            # 0% to 50% -> 1 to 0
        winrate_score = win_rate / 100                            # 0% to 100% -> 0 to 1
        trades_score = min(1, total_trades / 50)                  # 0 to 50+ trades -> 0 to 1
        pf_score = max(0, min(1, (profit_factor - 0.5) / 2))     # 0.5 to 2.5 -> 0 to 1
        
        # Weighted composite score
        composite = (
            return_score * 0.25 +      # 25% weight on returns
            sharpe_score * 0.20 +      # 20% weight on risk-adjusted returns
            drawdown_score * 0.20 +    # 20% weight on drawdown control
            winrate_score * 0.15 +     # 15% weight on win rate
            trades_score * 0.10 +      # 10% weight on trade frequency
            pf_score * 0.10            # 10% weight on profit factor
        )
        
        return composite
    
    async def run_optimization(self, symbols: List[str] = None, max_combinations: int = 200):
        """Run the complete optimization process"""
        if symbols is None:
            symbols = ['BTCUSDT']
        
        logger.info("Starting scalping parameter optimization", 
                   symbols=symbols, max_combinations=max_combinations)
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(max_combinations)
        
        # Test each combination
        successful_tests = 0
        for i, params in enumerate(combinations):
            logger.info("Testing combination", 
                       progress=f"{i+1}/{len(combinations)}", 
                       params=params)
            
            result = await self.test_parameter_combination(params, symbols)
            self.results.append(result)
            
            if result['success']:
                successful_tests += 1
                logger.info("Test completed", 
                           return_pct=result['total_return_pct'],
                           sharpe=result['sharpe_ratio'],
                           trades=result['total_trades'],
                           score=result['composite_score'])
        
        logger.info("Optimization completed", 
                   total_tests=len(combinations),
                   successful=successful_tests)
    
    def analyze_results(self, top_n: int = 10) -> List[Dict]:
        """Analyze optimization results and return top performers"""
        # Filter successful results
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            logger.error("No successful optimization results")
            return []
        
        # Sort by composite score
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Get top N results
        top_results = successful_results[:top_n]
        
        logger.info("Top performing parameter sets identified", count=len(top_results))
        
        return top_results
    
    def generate_optimization_report(self, top_results: List[Dict]) -> str:
        """Generate a comprehensive optimization report"""
        if not top_results:
            return "No optimization results available."
        
        report = []
        report.append("=" * 100)
        report.append("SCALPING STRATEGY PARAMETER OPTIMIZATION REPORT")
        report.append("=" * 100)
        report.append("")
        
        # Summary statistics
        all_successful = [r for r in self.results if r['success']]
        if all_successful:
            avg_return = np.mean([r['total_return_pct'] for r in all_successful])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_successful])
            avg_drawdown = np.mean([r['max_drawdown_pct'] for r in all_successful])
            
            report.append("📊 OPTIMIZATION SUMMARY")
            report.append("-" * 50)
            report.append(f"Total parameter combinations tested: {len(self.results)}")
            report.append(f"Successful tests: {len(all_successful)}")
            report.append(f"Average return: {avg_return:.2f}%")
            report.append(f"Average Sharpe ratio: {avg_sharpe:.2f}")
            report.append(f"Average max drawdown: {avg_drawdown:.2f}%")
            report.append("")
        
        # Top performers
        report.append("🏆 TOP PERFORMING PARAMETER SETS")
        report.append("-" * 80)
        report.append(f"{'Rank':<5} {'Return%':<8} {'Sharpe':<7} {'MaxDD%':<7} {'Trades':<7} {'WinRate%':<9} {'Score':<6}")
        report.append("-" * 80)
        
        for i, result in enumerate(top_results, 1):
            report.append(
                f"{i:<5} "
                f"{result['total_return_pct']:<8.2f} "
                f"{result['sharpe_ratio']:<7.2f} "
                f"{result['max_drawdown_pct']:<7.2f} "
                f"{result['total_trades']:<7} "
                f"{result['win_rate_pct']:<9.2f} "
                f"{result['composite_score']:<6.3f}"
            )
        
        report.append("")
        
        # Best parameter set details
        if top_results:
            best = top_results[0]
            report.append("🎯 OPTIMAL PARAMETERS (Best Overall)")
            report.append("-" * 50)
            params = best['parameters']
            report.append(f"Fast EMA: {params['fast_ema']}")
            report.append(f"Slow EMA: {params['slow_ema']}")
            report.append(f"Signal EMA: {params['signal_ema']}")
            report.append(f"Volume Multiplier: {params['volume_multiplier']}")
            report.append(f"RSI Overbought: {params['rsi_overbought']}")
            report.append(f"RSI Oversold: {params['rsi_oversold']}")
            report.append(f"Stop Loss %: {params['stop_loss_pct']}")
            report.append(f"Take Profit %: {params['take_profit_pct']}")
            report.append(f"Position Size: {params['position_size']}")
            report.append(f"Risk/Reward Ratio: {params['take_profit_pct']/params['stop_loss_pct']:.2f}")
            report.append("")
        
        # Parameter sensitivity analysis
        report.append("📈 PARAMETER SENSITIVITY ANALYSIS")
        report.append("-" * 50)
        
        # Analyze which parameters have the most impact
        param_impacts = self._analyze_parameter_impact(all_successful)
        for param, impact in param_impacts.items():
            report.append(f"{param}: {impact:.3f} correlation with performance")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def _analyze_parameter_impact(self, results: List[Dict]) -> Dict[str, float]:
        """Analyze correlation between parameters and performance"""
        if len(results) < 10:
            return {}
        
        # Create DataFrame for analysis
        data = []
        for result in results:
            row = result['parameters'].copy()
            row['performance'] = result['composite_score']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        correlations = {}
        for param in df.columns:
            if param != 'performance':
                try:
                    corr = df[param].corr(df['performance'])
                    if not pd.isna(corr):
                        correlations[param] = abs(corr)  # Use absolute correlation
                except:
                    continue
        
        # Sort by impact
        return dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    
    def save_results(self, filename: str = None):
        """Save optimization results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scalping_optimization_{timestamp}.json"
        
        results_dir = Path('backtest_results')
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        # Prepare data for JSON serialization
        json_data = {
            'optimization_timestamp': datetime.now().isoformat(),
            'total_combinations': len(self.results),
            'successful_combinations': len([r for r in self.results if r['success']]),
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info("Optimization results saved", filepath=filepath)
        return filepath
    
    async def cleanup(self):
        """Clean up resources"""
        if self.exchange:
            await self.exchange.disconnect()

async def main():
    """Main optimization function"""
    optimizer = ScalpingParameterOptimizer()
    
    try:
        # Initialize with data
        await optimizer.initialize(
            symbols=['BTCUSDT'], 
            interval='5m', 
            days_back=7
        )
        
        # Run optimization
        await optimizer.run_optimization(
            symbols=['BTCUSDT'],
            max_combinations=100  # Adjust based on computational resources
        )
        
        # Analyze results
        top_results = optimizer.analyze_results(top_n=10)
        
        # Generate and display report
        if top_results:
            report = optimizer.generate_optimization_report(top_results)
            print(report)
            
            # Save results
            optimizer.save_results()
        else:
            print("No successful optimization results found.")
    
    finally:
        await optimizer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())