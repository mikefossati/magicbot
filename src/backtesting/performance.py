import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()

class PerformanceAnalyzer:
    """Advanced performance analysis for backtesting results"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.equity_curve = pd.DataFrame(results['equity_curve'], columns=['timestamp', 'equity'])
        self.trades = results['trades_detail']
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive performance report"""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("MAGICBOT BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Strategy Information
        report.append("STRATEGY INFORMATION")
        report.append("-" * 30)
        report.append(f"Strategy: {self.results.get('strategy_name', 'Unknown')}")
        
        period = self.results['backtest_period']
        if period['start'] and period['end']:
            report.append(f"Period: {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}")
            report.append(f"Duration: {period['duration_days']} days")
        report.append("")
        
        # Capital & Returns
        report.append("CAPITAL & RETURNS")
        report.append("-" * 30)
        capital = self.results['capital']
        report.append(f"Initial Capital: ${capital['initial']:,.2f}")
        report.append(f"Final Capital: ${capital['final']:,.2f}")
        report.append(f"Total Return: {capital['total_return_pct']:.2f}%")
        
        if period['duration_days'] > 0:
            annualized_return = ((capital['final'] / capital['initial']) ** (365 / period['duration_days']) - 1) * 100
            report.append(f"Annualized Return: {annualized_return:.2f}%")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 30)
        trades = self.results['trades']
        report.append(f"Total Trades: {trades['total']}")
        report.append(f"Winning Trades: {trades['winning']}")
        report.append(f"Losing Trades: {trades['losing']}")
        report.append(f"Win Rate: {trades['win_rate_pct']:.2f}%")
        report.append(f"Average Win: ${trades['avg_win']:.2f}")
        report.append(f"Average Loss: ${trades['avg_loss']:.2f}")
        report.append(f"Profit Factor: {trades['profit_factor']:.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 30)
        risk = self.results['risk_metrics']
        report.append(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {risk['max_drawdown_pct']:.2f}%")
        report.append(f"Volatility: {risk['volatility_pct']:.2f}%")
        report.append("")
        
        # Trade Analysis
        if self.trades:
            report.extend(self._analyze_trades())
        
        # Monthly Returns
        if len(self.equity_curve) > 30:
            report.extend(self._calculate_monthly_returns())
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info("Performance report saved", path=save_path)
        
        return report_text
    
    def _analyze_trades(self) -> List[str]:
        """Analyze individual trades"""
        analysis = []
        analysis.append("TRADE ANALYSIS")
        analysis.append("-" * 30)
        
        if not self.trades:
            analysis.append("No trades executed")
            return analysis
        
        # Convert trades to DataFrame
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'duration': (trade.exit_time - trade.entry_time).total_seconds() / 3600,  # hours
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price
            })
        
        trades_df = pd.DataFrame(trade_data)
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        
        analysis.append(f"Best Trade: {best_trade['symbol']} - ${best_trade['pnl']:.2f} ({best_trade['pnl_pct']:.2f}%)")
        analysis.append(f"Worst Trade: {worst_trade['symbol']} - ${worst_trade['pnl']:.2f} ({worst_trade['pnl_pct']:.2f}%)")
        
        # Average trade duration
        avg_duration = trades_df['duration'].mean()
        analysis.append(f"Average Trade Duration: {avg_duration:.1f} hours")
        
        # Symbol performance
        symbol_performance = trades_df.groupby('symbol')['pnl'].agg(['count', 'sum', 'mean'])
        analysis.append("")
        analysis.append("Performance by Symbol:")
        for symbol, stats in symbol_performance.iterrows():
            analysis.append(f"  {symbol}: {stats['count']} trades, Total P&L: ${stats['sum']:.2f}, Avg: ${stats['mean']:.2f}")
        
        analysis.append("")
        return analysis
    
    def _calculate_monthly_returns(self) -> List[str]:
        """Calculate monthly returns breakdown"""
        analysis = []
        analysis.append("MONTHLY RETURNS")
        analysis.append("-" * 30)
        
        # Create a copy to avoid modifying the original
        equity_df = self.equity_curve.copy()
        
        # Resample equity curve to monthly
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        monthly_equity = equity_df.resample('M')['equity'].last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        for date, return_pct in monthly_returns.items():
            analysis.append(f"{date.strftime('%Y-%m')}: {return_pct:.2f}%")
        
        analysis.append("")
        analysis.append(f"Best Month: {monthly_returns.max():.2f}%")
        analysis.append(f"Worst Month: {monthly_returns.min():.2f}%")
        analysis.append(f"Average Monthly Return: {monthly_returns.mean():.2f}%")
        analysis.append("")
        
        return analysis
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve['timestamp'], self.equity_curve['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Equity curve plot saved", path=save_path)
        
        plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """Plot drawdown chart"""
        peak = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - peak) / peak * 100
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(self.equity_curve['timestamp'], drawdown, 0, alpha=0.3, color='red')
        plt.plot(self.equity_curve['timestamp'], drawdown, color='red')
        plt.title('Drawdown Chart')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Drawdown plot saved", path=save_path)
        
        plt.show()
    
    def plot_trade_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of trade P&L"""
        if not self.trades:
            print("No trades to plot")
            return
        
        pnls = [trade.pnl for trade in self.trades]
        
        plt.figure(figsize=(10, 6))
        plt.hist(pnls, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Trade distribution plot saved", path=save_path)
        
        plt.show()
    
    def plot_rolling_sharpe(self, window_days: int = 30, save_path: Optional[str] = None):
        """Plot rolling Sharpe ratio"""
        if len(self.equity_curve) < window_days:
            print(f"Not enough data for {window_days}-day rolling Sharpe")
            return
        
        # Calculate returns
        equity_df = self.equity_curve.copy()
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = equity_df['returns'].rolling(window=window_days).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else 0
        )
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['timestamp'], rolling_sharpe)
        plt.axhline(y=1, color='green', linestyle='--', label='Good (1.0)')
        plt.axhline(y=2, color='blue', linestyle='--', label='Excellent (2.0)')
        plt.title(f'{window_days}-Day Rolling Sharpe Ratio')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Rolling Sharpe plot saved", path=save_path)
        
        plt.show()
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics dictionary"""
        stats = {
            'total_return_pct': self.results['capital']['total_return_pct'],
            'sharpe_ratio': self.results['risk_metrics']['sharpe_ratio'],
            'max_drawdown_pct': self.results['risk_metrics']['max_drawdown_pct'],
            'win_rate_pct': self.results['trades']['win_rate_pct'],
            'profit_factor': self.results['trades']['profit_factor'],
            'total_trades': self.results['trades']['total'],
            'volatility_pct': self.results['risk_metrics']['volatility_pct']
        }
        
        # Calculate additional metrics
        if self.trades:
            trade_pnls = [trade.pnl for trade in self.trades]
            stats['largest_win'] = max(trade_pnls)
            stats['largest_loss'] = min(trade_pnls)
            stats['avg_trade_pnl'] = np.mean(trade_pnls)
            
            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for trade in self.trades:
                if trade.pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            stats['max_consecutive_wins'] = max_consecutive_wins
            stats['max_consecutive_losses'] = max_consecutive_losses
        
        return stats
    
    def compare_to_benchmark(self, benchmark_return: float) -> Dict[str, Any]:
        """Compare strategy performance to a benchmark"""
        strategy_return = self.results['capital']['total_return_pct']
        
        comparison = {
            'strategy_return_pct': strategy_return,
            'benchmark_return_pct': benchmark_return,
            'excess_return_pct': strategy_return - benchmark_return,
            'outperformed': strategy_return > benchmark_return
        }
        
        # Calculate information ratio if we have enough data
        if len(self.equity_curve) > 1:
            equity_df = self.equity_curve.copy()
            equity_df['returns'] = equity_df['equity'].pct_change().dropna()
            
            # Assume benchmark has constant daily return
            daily_benchmark_return = (1 + benchmark_return / 100) ** (1/252) - 1
            excess_returns = equity_df['returns'] - daily_benchmark_return
            
            if excess_returns.std() != 0:
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                comparison['information_ratio'] = information_ratio
        
        return comparison