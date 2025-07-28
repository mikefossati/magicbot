import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from decimal import Decimal
import structlog
from dataclasses import dataclass, field

from ..strategies.base import BaseStrategy
from ..strategies.signal import Signal
from ..data.historical_manager import HistoricalDataManager

logger = structlog.get_logger()

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    slippage: float = 0.0
    strategy: str = ""

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    
    # Trailing stop loss fields
    trailing_stop_enabled: bool = False
    trailing_stop_distance: Optional[float] = None  # Distance from current price (percentage or absolute)
    trailing_stop_type: str = "percentage"  # "percentage" or "absolute"
    trailing_stop_price: Optional[float] = None  # Current trailing stop price
    highest_price: Optional[float] = None  # Highest price since position opened (for LONG)
    lowest_price: Optional[float] = None   # Lowest price since position opened (for SHORT)

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005   # 0.05% slippage
    max_positions: int = 10
    position_sizing: str = "fixed"  # 'fixed', 'percentage', 'kelly'
    position_size: float = 0.1      # 10% of capital or fixed amount

class BacktestEngine:
    """
    Event-driven backtesting engine that simulates trading strategies
    on historical data with realistic trading costs and constraints
    """
    
    def __init__(self, initial_balance: float = 10000.0, fast_mode: bool = False, signal_interval: int = 10):
        """Initialize backtesting engine"""
        self.initial_balance = initial_balance
        self.initial_capital = initial_balance  # Backward compatibility
        self.balance = initial_balance
        self.capital = initial_balance  # Backward compatibility
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.current_time: Optional[datetime] = None
        self.portfolio_values: List[Dict] = []
        self.logger = structlog.get_logger()
        self.fast_mode = fast_mode  # Skip detailed tracking for optimization
        self.signal_interval = signal_interval  # Only check for signals every N periods
        self.current_time: Optional[datetime] = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        self.position_sizing = "fixed"  # 'fixed', 'percentage', 'kelly'
        self.position_size = 0.1      # 10% of capital or fixed amount
        
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        historical_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Run a complete backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            historical_data: Historical market data for all symbols
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        logger.info("Starting backtest",
                   strategy=strategy.strategy_name,
                   start_date=start_date,
                   end_date=end_date,
                   initial_capital=self.initial_capital)
        
        # Reset state
        self._reset_state()
        
        # Get all timestamps across all symbols
        all_timestamps = self._get_unified_timestamps(historical_data)
        
        if len(all_timestamps) == 0:
            logger.warning("No timestamps found in historical data")
            return self._create_empty_results(strategy.strategy_name, start_date, end_date)
        
        logger.info("Processing timestamps", total_timestamps=len(all_timestamps))
        
        # Add initial equity point
        first_timestamp = all_timestamps[0]
        self.equity_curve.append((first_timestamp, self.capital))
        
        # Process timestamps with optimization
        signals_generated = 0
        last_strategy_data = None  # Cache strategy data
        
        for i, timestamp in enumerate(all_timestamps):
            self.current_time = timestamp
            
            # Get current market data for all symbols
            current_market_data = self._get_market_data_at_time(historical_data, timestamp)
            
            # Always update position values (for stop loss/take profit)
            self._update_positions(current_market_data)
            
            # Only generate signals periodically (performance optimization)
            if i % self.signal_interval == 0 or i == len(all_timestamps) - 1:
                try:
                    # Prepare data in the format expected by strategy
                    strategy_data = self._prepare_strategy_data(historical_data, timestamp)
                    last_strategy_data = strategy_data  # Cache for next iteration
                    
                    signals = await strategy.generate_signals(strategy_data)
                    signals_generated += len(signals)
                    
                    # Process each signal
                    for signal in signals:
                        await self._process_signal(signal, current_market_data)
                    
                except Exception as e:
                    logger.error("Error processing signals", error=str(e), timestamp=timestamp)
                    continue
            
            # Record equity curve (less frequently in fast mode)
            equity_interval = 50 if self.fast_mode else 10
            if i % equity_interval == 0 or i == len(all_timestamps) - 1:
                total_equity = self._calculate_total_equity(current_market_data)
                self.equity_curve.append((timestamp, total_equity))
        
        logger.info("Signal generation completed", total_signals=signals_generated)
        
        # Close all remaining positions at the end
        if all_timestamps:
            final_market_data = self._get_market_data_at_time(historical_data, all_timestamps[-1])
            self._close_all_positions(final_market_data)
        
        # Final equity point
        final_equity = self._calculate_total_equity({})
        if all_timestamps:
            self.equity_curve.append((all_timestamps[-1], final_equity))
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(strategy.strategy_name, start_date, end_date)
        
        logger.info("Backtest completed",
                   total_trades=len(self.trades),
                   final_capital=self.capital,
                   total_return=((self.capital / self.initial_capital) - 1) * 100)
        
        return results
    
    def _create_empty_results(self, strategy_name: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Create empty results when no data is available"""
        return {
            'strategy_name': strategy_name,
            'backtest_period': {
                'start': start_date,
                'end': end_date,
                'duration_days': (end_date - start_date).days
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.initial_capital,
                'total_return_pct': 0.0
            },
            'trades': {
                'total': 0,
                'winning': 0,
                'losing': 0,
                'win_rate_pct': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            },
            'risk_metrics': {
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'volatility_pct': 0.0
            },
            'equity_curve': [(start_date, self.initial_capital), (end_date, self.initial_capital)],
            'trades_detail': [],
            'signals_log': []
        }
    
    def _reset_state(self):
        """Reset backtesting state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signals_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def _get_unified_timestamps(self, historical_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get all unique timestamps across all symbols, sorted with optional subsampling"""
        all_timestamps = set()
        
        for symbol, df in historical_data.items():
            if len(df) > 0:
                # Handle timestamp as either column or index
                if 'timestamp' in df.columns:
                    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    # Timestamp is the index
                    timestamps = df.index
                    if not isinstance(timestamps[0], pd.Timestamp):
                        timestamps = pd.to_datetime(timestamps, unit='ms')
                
                # Performance optimization: subsample if too many data points
                if self.fast_mode and len(timestamps) > 1000:
                    # Take every 5th timestamp for fast mode
                    timestamps = timestamps[::5]
                    logger.info("Subsampling data for performance", 
                               symbol=symbol, 
                               original_count=len(df), 
                               subsampled_count=len(timestamps))
                
                all_timestamps.update(timestamps)
        
        timestamps_list = sorted(all_timestamps)
        
        # Additional subsampling if still too many timestamps
        if len(timestamps_list) > 2000:
            subsample_factor = max(1, len(timestamps_list) // 2000)
            timestamps_list = timestamps_list[::subsample_factor]
            logger.info("Additional timestamp subsampling", 
                       subsample_factor=subsample_factor,
                       final_count=len(timestamps_list))
        
        return timestamps_list
    
    def _get_market_data_at_time(
        self, 
        historical_data: Dict[str, pd.DataFrame], 
        timestamp: datetime
    ) -> Dict[str, Dict[str, float]]:
        """Get market data for all symbols at a specific timestamp"""
        market_data = {}
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        for symbol, df in historical_data.items():
            if len(df) == 0:
                continue
            
            # Find the closest timestamp (latest data point before or at target time)
            if 'timestamp' in df.columns:
                # Timestamp is a column
                mask = df['timestamp'] <= timestamp_ms
                if mask.any():
                    row = df[mask].iloc[-1]
                    timestamp_val = int(row['timestamp'])
                else:
                    continue
            else:
                # Timestamp is the index
                mask = df.index <= timestamp
                if mask.any():
                    row = df[mask].iloc[-1]
                    timestamp_val = int(row.name.timestamp() * 1000)
                else:
                    continue
        
            market_data[symbol] = {
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'timestamp': timestamp_val
            }
        
        return market_data
    
    def _prepare_strategy_data(
        self, 
        historical_data: Dict[str, pd.DataFrame], 
        current_time: datetime
    ) -> Dict[str, List[Dict]]:
        """Prepare historical data for strategy analysis with optimization"""
        strategy_data = {}
        current_timestamp_ms = int(current_time.timestamp() * 1000)
        
        for symbol, df in historical_data.items():
            if len(df) == 0:
                continue
            
            # Get all data up to current time
            if 'timestamp' in df.columns:
                # Timestamp is a column
                mask = df['timestamp'] <= current_timestamp_ms
                historical_df = df[mask]
            else:
                # Timestamp is the index
                mask = df.index <= current_time
                historical_df = df[mask]
            
            if len(historical_df) > 0:
                # Performance optimization: limit historical data for strategies
                # Most strategies don't need more than 200 periods
                max_history = 500 if self.fast_mode else 1000
                if len(historical_df) > max_history:
                    historical_df = historical_df.tail(max_history)
                
                # Convert to list of dictionaries (optimized)
                if 'timestamp' in historical_df.columns:
                    timestamps = historical_df['timestamp'].astype(int)
                else:
                    timestamps = (historical_df.index.astype(int) // 1000000).astype(int)
                
                strategy_data[symbol] = [
                    {
                        'timestamp': int(timestamps.iloc[i]),
                        'open': float(historical_df['open'].iloc[i]),
                        'high': float(historical_df['high'].iloc[i]),
                        'low': float(historical_df['low'].iloc[i]),
                        'close': float(historical_df['close'].iloc[i]),
                        'volume': float(historical_df['volume'].iloc[i])
                    }
                    for i in range(len(historical_df))
                ]
        
        return strategy_data
    
    async def _process_signal(self, signal: Signal, market_data: Dict[str, Dict[str, float]]):
        """Process a trading signal"""
        if signal.symbol not in market_data:
            logger.warning("No market data for signal", symbol=signal.symbol)
            return
        
        current_price = market_data[signal.symbol]['close']
        
        # Log the signal
        self.signals_log.append({
            'timestamp': self.current_time,
            'symbol': signal.symbol,
            'action': signal.action,
            'price': current_price,
            'confidence': signal.confidence,
            'metadata': signal.metadata
        })
        
        if signal.action == 'BUY':
            await self._execute_buy(signal, current_price)
        elif signal.action == 'SELL':
            await self._execute_sell(signal, current_price)
    
    async def _execute_buy(self, signal: Signal, current_price: float):
        """Execute a buy order"""
        # Check if we already have a position
        if signal.symbol in self.positions:
            logger.debug("Already have position", symbol=signal.symbol)
            return
        
        # Calculate position size
        position_value = self._calculate_position_size(current_price)
        quantity = position_value / current_price
        
        # Check if we have enough capital
        total_cost = position_value * (1 + self.commission_rate + self.slippage_rate)
        
        if total_cost > self.capital:
            logger.debug("Insufficient capital for trade", 
                        required=total_cost, 
                        available=self.capital)
            return
        
        # Apply slippage
        execution_price = current_price * (1 + self.slippage_rate)
        
        # Extract stop loss, take profit, and trailing stop loss from signal metadata
        stop_loss = None
        take_profit = None
        trailing_stop_enabled = False
        trailing_stop_distance = None
        trailing_stop_type = "percentage"
        
        if signal.metadata:
            stop_loss = signal.metadata.get('stop_loss')
            take_profit = signal.metadata.get('take_profit')
            trailing_stop_enabled = signal.metadata.get('trailing_stop_enabled', False)
            trailing_stop_distance = signal.metadata.get('trailing_stop_distance')
            trailing_stop_type = signal.metadata.get('trailing_stop_type', 'percentage')
        
        # Initialize trailing stop price if enabled
        trailing_stop_price = None
        if trailing_stop_enabled and trailing_stop_distance:
            if trailing_stop_type == "percentage":
                # For LONG positions, trailing stop is below current price
                trailing_stop_price = execution_price * (1 - trailing_stop_distance / 100)
            else:  # absolute
                trailing_stop_price = execution_price - trailing_stop_distance
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            side='LONG',
            quantity=quantity,
            entry_price=execution_price,
            entry_time=self.current_time,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_enabled=trailing_stop_enabled,
            trailing_stop_distance=trailing_stop_distance,
            trailing_stop_type=trailing_stop_type,
            trailing_stop_price=trailing_stop_price,
            highest_price=execution_price,  # Initialize with entry price
            lowest_price=execution_price    # Initialize with entry price
        )
        
        self.positions[signal.symbol] = position
        
        # Deduct capital
        commission = position_value * self.commission_rate
        self.capital -= (position_value + commission)
        
        logger.debug("Buy executed",
                    symbol=signal.symbol,
                    quantity=quantity,
                    price=execution_price,
                    cost=position_value)
    
    async def _execute_sell(self, signal: Signal, current_price: float):
        """Execute a sell order (close position or open short)"""
        if signal.symbol in self.positions:
            # Close existing long position
            position = self.positions[signal.symbol]
            
            # Apply slippage
            execution_price = current_price * (1 - self.slippage_rate)
            
            # Calculate P&L for LONG position
            if position.side == 'LONG':
                pnl = (execution_price - position.entry_price) * position.quantity
                pnl_pct = (execution_price / position.entry_price - 1) * 100
            else:
                # Closing SHORT position
                pnl = (position.entry_price - execution_price) * position.quantity  
                pnl_pct = (position.entry_price / execution_price - 1) * 100
            
            # Calculate commission
            trade_value = execution_price * position.quantity
            commission = trade_value * self.commission_rate
            
            # Net P&L after commission
            net_pnl = pnl - commission
            
            # Add proceeds to capital (original position value + net P&L)
            original_position_value = position.entry_price * position.quantity
            self.capital += original_position_value + net_pnl
            
            # Create trade record
            trade = Trade(
                symbol=signal.symbol,
                entry_time=position.entry_time,
                exit_time=self.current_time,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=execution_price,
                quantity=position.quantity,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                slippage=(current_price - execution_price) * position.quantity,
                strategy=signal.metadata.get('strategy', 'unknown') if signal.metadata else 'unknown'
            )
            
            self.trades.append(trade)
            
            # Update counters
            self.total_trades += 1
            if net_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Remove position
            del self.positions[signal.symbol]
            
            logger.debug("Sell executed (close position)",
                        symbol=signal.symbol,
                        side=position.side,
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                        exit_price=execution_price,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct)
        else:
            # No existing position - could open a SHORT position
            # For now, just log that no position exists
            logger.debug("No position to sell", symbol=signal.symbol)
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on configuration"""
        if self.position_sizing == 'fixed':
            return self.position_size
        elif self.position_sizing == 'percentage':
            return self.capital * self.position_size
        else:
            # Default to percentage of capital
            return self.capital * self.position_size
    
    def _update_positions(self, market_data: Dict[str, Dict[str, float]]):
        """Update current position values and check for stop loss/take profit"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]['close']
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                
                # Update trailing stop loss if enabled
                self._update_trailing_stop(position)
                
                # Check stop loss and take profit conditions
                close_reason = self._check_stop_conditions(position)
                if close_reason:
                    positions_to_close.append((symbol, close_reason))
        
        # Close positions that hit stop loss or take profit
        for symbol, reason in positions_to_close:
            self._close_position_automatically(symbol, market_data[symbol]['close'], reason)
    
    def _calculate_total_equity(self, market_data: Dict[str, Dict[str, float]]) -> float:
        """Calculate total equity (capital + unrealized P&L)"""
        unrealized_pnl = 0
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]['close']
                unrealized_pnl += (position.current_price - position.entry_price) * position.quantity
        
        return self.capital + unrealized_pnl
    
    def _update_trailing_stop(self, position: Position):
        """Update trailing stop loss price based on current market price"""
        if not position.trailing_stop_enabled or not position.trailing_stop_distance:
            return
        
        current_price = position.current_price
        
        # Update highest/lowest price tracking
        if position.side == 'LONG':
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
                
                # Calculate new trailing stop price
                if position.trailing_stop_type == "percentage":
                    new_trailing_stop = position.highest_price * (1 - position.trailing_stop_distance / 100)
                else:  # absolute
                    new_trailing_stop = position.highest_price - position.trailing_stop_distance
                
                # Only update if new trailing stop is higher (more favorable)
                if position.trailing_stop_price is None or new_trailing_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing_stop
                    
                    logger.debug("Trailing stop updated for LONG position",
                               symbol=position.symbol,
                               highest_price=position.highest_price,
                               new_trailing_stop=position.trailing_stop_price)
        
        elif position.side == 'SHORT':
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
                
                # Calculate new trailing stop price
                if position.trailing_stop_type == "percentage":
                    new_trailing_stop = position.lowest_price * (1 + position.trailing_stop_distance / 100)
                else:  # absolute
                    new_trailing_stop = position.lowest_price + position.trailing_stop_distance
                
                # Only update if new trailing stop is lower (more favorable)
                if position.trailing_stop_price is None or new_trailing_stop < position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing_stop
                    
                    logger.debug("Trailing stop updated for SHORT position",
                               symbol=position.symbol,
                               lowest_price=position.lowest_price,
                               new_trailing_stop=position.trailing_stop_price)
    
    def _check_stop_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed due to stop loss, take profit, or trailing stop"""
        if position.side == 'LONG':
            # Check trailing stop loss first (it may be more restrictive than regular stop loss)
            if position.trailing_stop_enabled and position.trailing_stop_price and position.current_price <= position.trailing_stop_price:
                return 'trailing_stop'
            # Check regular stop loss
            elif position.stop_loss and position.current_price <= position.stop_loss:
                return 'stop_loss'
            # Check take profit
            elif position.take_profit and position.current_price >= position.take_profit:
                return 'take_profit'
                
        elif position.side == 'SHORT':
            # Check trailing stop loss first (it may be more restrictive than regular stop loss)
            if position.trailing_stop_enabled and position.trailing_stop_price and position.current_price >= position.trailing_stop_price:
                return 'trailing_stop'
            # Check regular stop loss
            elif position.stop_loss and position.current_price >= position.stop_loss:
                return 'stop_loss'
            # Check take profit
            elif position.take_profit and position.current_price <= position.take_profit:
                return 'take_profit'
        
        return None
    
    def _close_position_automatically(self, symbol: str, current_price: float, reason: str):
        """Close a position automatically due to stop loss or take profit"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Use stop loss, take profit, or trailing stop price for execution
        if reason == 'stop_loss' and position.stop_loss:
            execution_price = position.stop_loss
        elif reason == 'take_profit' and position.take_profit:
            execution_price = position.take_profit
        elif reason == 'trailing_stop' and position.trailing_stop_price:
            execution_price = position.trailing_stop_price
        else:
            execution_price = current_price
        
        # Apply minimal slippage for automatic executions
        if position.side == 'LONG':
            execution_price = execution_price * (1 - self.slippage_rate / 2)  # Reduced slippage
        else:
            execution_price = execution_price * (1 + self.slippage_rate / 2)
        
        # Calculate P&L
        if position.side == 'LONG':
            pnl = (execution_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - execution_price) * position.quantity
        
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100
        
        # Calculate commission
        trade_value = execution_price * position.quantity
        commission = trade_value * self.commission_rate
        
        # Net P&L after commission
        net_pnl = pnl - commission
        
        # Add proceeds to capital (original position value + net P&L)
        original_position_value = position.entry_price * position.quantity
        self.capital += original_position_value + net_pnl
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=(current_price - execution_price) * position.quantity,
            strategy=f"auto_{reason}"
        )
        
        self.trades.append(trade)
        
        # Update counters
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Position closed automatically ({reason})",
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=execution_price,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    reason=reason)
    
    def _close_all_positions(self, market_data: Dict[str, Dict[str, float]]):
        """Close all remaining positions at the end of backtest"""
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                self._close_position_automatically(symbol, market_data[symbol]['close'], 'backtest_end')
    
    def _calculate_performance_metrics(self, strategy_name: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        # Ensure we have at least basic equity curve
        if not self.equity_curve:
            self.equity_curve = [(start_date, self.initial_capital), (end_date, self.capital)]
        
        # Convert equity curve to DataFrame for analysis
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        
        # Basic metrics
        total_return = (self.capital / self.initial_capital - 1) * 100
        
        # Trade analysis
        if self.trades:
            trade_pnls = [trade.pnl for trade in self.trades]
            win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
            avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
            avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if any(pnl < 0 for pnl in trade_pnls) else 0
            profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 and self.losing_trades > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Risk metrics
        if len(equity_df) > 1:
            equity_df['returns'] = equity_df['equity'].pct_change()
            daily_returns = equity_df['returns'].dropna()
            
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            volatility = 0
        
        return {
            'strategy_name': strategy_name,
            'backtest_period': {
                'start': start_date,
                'end': end_date,
                'duration_days': (end_date - start_date).days
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.capital,
                'total_return_pct': total_return
            },
            'trades': {
                'total': self.total_trades,
                'winning': self.winning_trades,
                'losing': self.losing_trades,
                'win_rate_pct': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'volatility_pct': volatility
            },
            'equity_curve': self.equity_curve,
            'trades_detail': self.trades,
            'signals_log': self.signals_log
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return (excess_returns / returns.std()) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if len(equity_series) == 0:
            return 0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return abs(drawdown.min()) * 100