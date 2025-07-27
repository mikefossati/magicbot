import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import asyncio
import structlog

from .engine import BacktestEngine, BacktestConfig, Trade
from ..strategies.base import BaseStrategy
from ..data.historical_manager import HistoricalDataManager
from ..exchanges.binance_exchange import BinanceBacktestingExchange

logger = structlog.get_logger()

class DashboardBacktestEngine(BacktestEngine):
    """
    Enhanced backtesting engine specifically designed for dashboard usage
    with progress tracking and real-time updates
    """
    
    def __init__(self, config: BacktestConfig, progress_callback: Optional[Callable] = None):
        super().__init__(config)
        self.progress_callback = progress_callback
        self.total_steps = 0
        self.current_step = 0
        
    async def run_backtest_with_progress(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with progress tracking and optimized data loading
        """
        if progress_callback:
            self.progress_callback = progress_callback
            
        exchange = None
        try:
            # Step 1: Load historical data
            await self._update_progress(5, "Loading historical data...")
            exchange = BinanceBacktestingExchange()
            await exchange.connect()
            data_manager = HistoricalDataManager(exchange)
            
            historical_data = await data_manager.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if historical_data is None or historical_data.empty:
                raise ValueError(f"No historical data available for {symbol} from {start_date} to {end_date}")
            
            await self._update_progress(20, "Preparing data...")
            
            # Format data for backtesting engine
            formatted_data = {symbol: historical_data}
            
            # Step 2: Initialize strategy
            await self._update_progress(30, "Initializing strategy...")
            
            # Step 3: Run backtest with progress tracking
            await self._update_progress(40, "Running backtest simulation...")
            
            results = await self._run_backtest_with_tracking(
                strategy=strategy,
                historical_data=formatted_data,
                start_date=start_date,
                end_date=end_date
            )
            
            await self._update_progress(80, "Processing results...")
            
            # Step 4: Enhance results
            enhanced_results = await self._enhance_results(results, historical_data, symbol, timeframe)
            
            await self._update_progress(100, "Backtest completed successfully!")
            
            return enhanced_results
            
        except Exception as e:
            logger.error("Dashboard backtest failed", error=str(e))
            await self._update_progress(0, f"Error: {str(e)}")
            raise
        finally:
            # Ensure exchange is disconnected
            if exchange:
                await exchange.disconnect()
    
    async def _run_backtest_with_tracking(
        self,
        strategy: BaseStrategy,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run backtest with progress tracking"""
        
        # Reset state
        self._reset_state()
        
        # Get all timestamps
        all_timestamps = self._get_unified_timestamps(historical_data)
        self.total_steps = len(all_timestamps)
        self.current_step = 0
        
        if len(all_timestamps) == 0:
            return self._create_empty_results(strategy.name, start_date, end_date)
        
        # Add initial equity point
        first_timestamp = all_timestamps[0]
        self.equity_curve.append((first_timestamp, self.capital))
        
        # Process each timestamp with progress updates
        for i, timestamp in enumerate(all_timestamps):
            self.current_time = timestamp
            self.current_step = i
            
            # Update progress every 100 steps
            if i % 100 == 0:
                progress = 40 + (i / len(all_timestamps)) * 35  # 40-75% range
                await self._update_progress(progress, f"Processing {i+1}/{len(all_timestamps)} data points...")
            
            # Get current market data
            current_market_data = self._get_market_data_at_time(historical_data, timestamp)
            
            # Update positions
            self._update_positions(current_market_data)
            
            # Generate and process signals
            try:
                strategy_data = self._prepare_strategy_data(historical_data, timestamp)
                signals = await strategy.generate_signals(strategy_data)
                
                for signal in signals:
                    await self._process_signal(signal, current_market_data)
                    
            except Exception as e:
                logger.warning("Error processing signals", error=str(e), timestamp=timestamp)
                continue
            
            # Record equity curve
            if i % 10 == 0 or i == len(all_timestamps) - 1:
                total_equity = self._calculate_total_equity(current_market_data)
                self.equity_curve.append((timestamp, total_equity))
        
        # Close remaining positions
        if all_timestamps:
            final_market_data = self._get_market_data_at_time(historical_data, all_timestamps[-1])
            self._close_all_positions(final_market_data)
        
        # Calculate final results
        results = self._calculate_performance_metrics(strategy.name, start_date, end_date)
        
        return results
    
    async def _enhance_results(
        self, 
        results: Dict, 
        historical_data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """Enhance results with visualization data"""
        
        enhanced = results.copy()
        
        # Add OHLCV data for charting
        ohlcv_data = []
        for _, row in historical_data.iterrows():
            ohlcv_data.append({
                "timestamp": int(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })
        
        enhanced["ohlcv_data"] = ohlcv_data
        
        # Add trade markers for visualization
        trade_markers = []
        for trade in self.trades:
            # Entry marker
            trade_markers.append({
                "timestamp": int(trade.entry_time.timestamp() * 1000),
                "price": float(trade.entry_price),
                "type": "entry",
                "side": trade.side,
                "symbol": trade.symbol
            })
            
            # Exit marker
            trade_markers.append({
                "timestamp": int(trade.exit_time.timestamp() * 1000),
                "price": float(trade.exit_price),
                "type": "exit",
                "side": trade.side,
                "symbol": trade.symbol,
                "pnl": float(trade.pnl)
            })
        
        enhanced["trade_markers"] = trade_markers
        
        # Add signal markers
        signal_markers = []
        for signal_log in self.signals_log:
            signal_markers.append({
                "timestamp": int(signal_log["timestamp"].timestamp() * 1000),
                "price": float(signal_log["price"]),
                "action": signal_log["action"],
                "confidence": float(signal_log["confidence"]),
                "symbol": signal_log["symbol"]
            })
        
        enhanced["signal_markers"] = signal_markers
        
        # Add equity curve data
        equity_data = []
        for timestamp, equity in self.equity_curve:
            equity_data.append({
                "timestamp": int(timestamp.timestamp() * 1000),
                "equity": float(equity)
            })
        
        enhanced["equity_data"] = equity_data
        
        # Calculate additional metrics
        enhanced["metrics"] = self._calculate_additional_metrics()
        
        # Add metadata
        enhanced["metadata"] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_points": len(historical_data),
            "signals_generated": len(self.signals_log),
            "execution_time": datetime.now()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        return self._convert_numpy_types(enhanced)
    
    def _calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional performance metrics for dashboard"""
        if not self.trades:
            return {}
        
        trade_pnls = [trade.pnl for trade in self.trades]
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        streak_type = None
        
        for trade in self.trades:
            if trade.pnl > 0:  # Winning trade
                if streak_type == 'win':
                    current_streak += 1
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    current_streak = 1
                    streak_type = 'win'
            else:  # Losing trade
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                    current_streak = 1
                    streak_type = 'loss'
        
        # Update final streak
        if streak_type == 'win':
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        # Calculate expectancy
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]
        
        if winning_trades and losing_trades:
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean(winning_trades)
            avg_loss = abs(np.mean(losing_trades))
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            expectancy = 0
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "largest_win": max(trade_pnls) if trade_pnls else 0,
            "largest_loss": min(trade_pnls) if trade_pnls else 0,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "expectancy": expectancy,
            "profit_factor": abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            "average_trade_duration": self._calculate_avg_trade_duration()
        }
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours"""
        if not self.trades:
            return 0
        
        durations = []
        for trade in self.trades:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Convert to hours
            durations.append(duration)
        
        return np.mean(durations)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle custom objects like Trade dataclass
            result = {}
            for key, value in obj.__dict__.items():
                result[key] = self._convert_numpy_types(value)
            return result
        else:
            return obj
    
    async def _update_progress(self, percentage: float, message: str = ""):
        """Update progress if callback is available"""
        if self.progress_callback:
            try:
                await self.progress_callback(percentage, message)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))
        
        # Also log progress
        if percentage % 20 == 0:  # Log every 20%
            logger.info("Backtest progress", percentage=percentage, message=message)

class BacktestSession:
    """Manages a backtesting session with state tracking"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.status = "pending"
        self.progress = 0.0
        self.message = ""
        self.start_time = datetime.now()
        self.end_time = None
        self.results = None
        self.error = None
    
    async def update_progress(self, percentage: float, message: str = ""):
        """Update session progress"""
        self.progress = percentage
        self.message = message
        if percentage >= 100:
            self.status = "completed"
            self.end_time = datetime.now()
        elif percentage > 0:
            self.status = "running"
    
    def set_error(self, error: str):
        """Set session error state"""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.now()
    
    def set_results(self, results: Dict):
        """Set session results"""
        self.results = results
        if self.status != "failed":
            self.status = "completed"
            self.end_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error
        }