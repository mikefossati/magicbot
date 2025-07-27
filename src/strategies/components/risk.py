"""
Risk management components for trading strategies.

This module provides configurable risk management functionality
including position sizing, stop losses, and leverage management.
"""

from typing import Dict, Any, Tuple
from decimal import Decimal
import structlog

logger = structlog.get_logger()

class RiskManager:
    """Manages risk-related calculations and validations"""
    
    def __init__(self, strategy_name: str, params: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.params = params
        
        # Risk management parameters
        self.stop_loss_pct = params.get('stop_loss_pct', 2.0)
        self.take_profit_pct = params.get('take_profit_pct', 4.0)
        self.trailing_stop_pct = params.get('trailing_stop_pct', 1.0)
        self.max_daily_trades = params.get('max_daily_trades', 10)
        
        # Leverage settings
        self.use_leverage = params.get('use_leverage', False)
        self.leverage = params.get('leverage', 1.0)
        self.max_leverage = params.get('max_leverage', 10.0)
        self.leverage_risk_factor = params.get('leverage_risk_factor', 0.6)
        
        # Position sizing
        self.base_position_size = params.get('position_size', 0.01)
        self.max_position_size = params.get('max_position_size', 0.1)
        self.confidence_scaling = params.get('confidence_position_scaling', False)
        
        logger.debug("Risk manager initialized", 
                    strategy=strategy_name,
                    use_leverage=self.use_leverage,
                    leverage=self.leverage,
                    base_position_size=self.base_position_size)
    
    def calculate_position_size(self, current_price: float, confidence: float, 
                              account_balance: float = None) -> float:
        """
        Calculate position size based on risk parameters and confidence.
        
        Args:
            current_price: Current market price
            confidence: Signal confidence (0-1)
            account_balance: Optional account balance for risk-based sizing
            
        Returns:
            Position size as fraction of portfolio
        """
        base_size = float(self.base_position_size)
        
        # Apply confidence scaling if enabled
        if self.confidence_scaling:
            min_scale = self.params.get('min_confidence_scale', 0.5)
            confidence_multiplier = min_scale + (confidence * (1 - min_scale))
            base_size = base_size * confidence_multiplier
        
        # Apply leverage adjustments
        if self.use_leverage and self.leverage > 1:
            # Reduce base position size when using leverage to manage risk
            adjusted_size = base_size * self.leverage_risk_factor
            
            # Further adjust based on confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
            final_size = adjusted_size * confidence_multiplier
            
            logger.debug("Leveraged position sizing",
                        base_size=base_size,
                        leverage=self.leverage,
                        risk_factor=self.leverage_risk_factor,
                        confidence=confidence,
                        final_size=final_size)
        else:
            final_size = base_size
        
        # Ensure position size doesn't exceed maximum
        final_size = min(final_size, self.max_position_size)
        
        return final_size
    
    def calculate_stop_levels(self, current_price: float, action: str, 
                            atr: float = None) -> Dict[str, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            current_price: Current market price
            action: Signal action (BUY/SELL)
            atr: Average True Range for dynamic stops
            
        Returns:
            Dictionary with stop_loss and take_profit levels
        """
        if self.use_leverage and self.leverage > 1:
            return self._calculate_leveraged_stops(current_price, action, atr)
        else:
            return self._calculate_standard_stops(current_price, action, atr)
    
    def _calculate_standard_stops(self, current_price: float, action: str, 
                                atr: float = None) -> Dict[str, float]:
        """Calculate standard stop loss and take profit levels"""
        if action == 'BUY':
            if atr is not None:
                # Use ATR-based stops (more dynamic)
                stop_loss = current_price - (atr * 2.0)
            else:
                # Use percentage-based stops
                stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            
            take_profit = current_price * (1 + self.take_profit_pct / 100)
            
        else:  # SELL
            if atr is not None:
                stop_loss = current_price + (atr * 2.0)
            else:
                stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            
            take_profit = current_price * (1 - self.take_profit_pct / 100)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def _calculate_leveraged_stops(self, current_price: float, action: str, 
                                 atr: float = None) -> Dict[str, float]:
        """Calculate stop loss and take profit levels adjusted for leverage"""
        # Tighter stops with leverage to manage risk
        leverage_adjustment = 1.0 / (1.0 + (self.leverage - 1) * 0.3)  # Reduce stops by 30% per leverage unit
        
        adjusted_stop_pct = self.stop_loss_pct * leverage_adjustment
        adjusted_profit_pct = self.take_profit_pct * leverage_adjustment
        
        if action == 'BUY':
            if atr is not None:
                # Use more conservative ATR-based stops for leverage
                atr_stop_multiplier = 1.5 if self.leverage <= 3 else 1.0
                atr_based_stop = atr * atr_stop_multiplier
                percentage_stop = current_price * (adjusted_stop_pct / 100)
                stop_loss = current_price - min(percentage_stop, atr_based_stop)
            else:
                stop_loss = current_price * (1 - adjusted_stop_pct / 100)
            
            take_profit = current_price * (1 + adjusted_profit_pct / 100)
            
        else:  # SELL
            if atr is not None:
                atr_stop_multiplier = 1.5 if self.leverage <= 3 else 1.0
                atr_based_stop = atr * atr_stop_multiplier
                percentage_stop = current_price * (adjusted_stop_pct / 100)
                stop_loss = current_price + min(percentage_stop, atr_based_stop)
            else:
                stop_loss = current_price * (1 + adjusted_stop_pct / 100)
            
            take_profit = current_price * (1 - adjusted_profit_pct / 100)
        
        logger.debug("Leveraged stops calculated",
                    leverage=self.leverage,
                    original_stop_pct=self.stop_loss_pct,
                    adjusted_stop_pct=adjusted_stop_pct,
                    stop_loss=stop_loss,
                    take_profit=take_profit)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                              action: str, highest_profit: float = 0) -> float:
        """
        Calculate trailing stop level.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            action: Position action (BUY/SELL)
            highest_profit: Highest profit achieved (for trailing logic)
            
        Returns:
            Trailing stop price
        """
        trailing_distance_pct = self.trailing_stop_pct / 100
        
        if action == 'BUY':
            # For long positions, trail below current price
            trailing_stop = current_price * (1 - trailing_distance_pct)
        else:
            # For short positions, trail above current price
            trailing_stop = current_price * (1 + trailing_distance_pct)
        
        return trailing_stop
    
    def validate_risk_parameters(self) -> Tuple[bool, list]:
        """
        Validate risk management parameters.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic parameter validation
        if self.stop_loss_pct <= 0:
            errors.append("stop_loss_pct must be positive")
        
        if self.take_profit_pct <= 0:
            errors.append("take_profit_pct must be positive")
        
        if self.base_position_size <= 0 or self.base_position_size > 1:
            errors.append("position_size must be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            errors.append("max_position_size must be between 0 and 1")
        
        if self.base_position_size > self.max_position_size:
            errors.append("position_size cannot exceed max_position_size")
        
        # Leverage validation
        if self.use_leverage:
            if self.leverage <= 0 or self.leverage > self.max_leverage:
                errors.append(f"leverage must be between 0 and {self.max_leverage}")
            
            if not (0 < self.leverage_risk_factor <= 1):
                errors.append("leverage_risk_factor must be between 0 and 1")
        
        # Risk/reward validation
        if self.stop_loss_pct >= self.take_profit_pct:
            errors.append("take_profit_pct should be greater than stop_loss_pct for positive risk/reward")
        
        return len(errors) == 0, errors
    
    def get_risk_metrics(self, entry_price: float, current_price: float, 
                        position_size: float, action: str) -> Dict[str, float]:
        """
        Calculate current risk metrics for a position.
        
        Args:
            entry_price: Position entry price
            current_price: Current market price
            position_size: Position size
            action: Position action (BUY/SELL)
            
        Returns:
            Dictionary with risk metrics
        """
        if action == 'BUY':
            unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Apply leverage to PnL
        if self.use_leverage:
            unrealized_pnl_pct *= self.leverage
        
        # Calculate risk metrics
        portfolio_risk_pct = position_size * abs(unrealized_pnl_pct)
        max_loss_pct = position_size * self.stop_loss_pct
        max_gain_pct = position_size * self.take_profit_pct
        
        return {
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'portfolio_risk_pct': portfolio_risk_pct,
            'max_loss_pct': max_loss_pct,
            'max_gain_pct': max_gain_pct,
            'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct,
            'position_size': position_size,
            'leverage': self.leverage if self.use_leverage else 1.0
        }
    
    def get_daily_trade_status(self, current_trades: int) -> Dict[str, Any]:
        """
        Get current daily trading status.
        
        Args:
            current_trades: Number of trades executed today
            
        Returns:
            Dictionary with trading status
        """
        return {
            'current_trades': current_trades,
            'max_daily_trades': self.max_daily_trades,
            'trades_remaining': max(0, self.max_daily_trades - current_trades),
            'can_trade': current_trades < self.max_daily_trades,
            'utilization_pct': (current_trades / self.max_daily_trades) * 100
        }