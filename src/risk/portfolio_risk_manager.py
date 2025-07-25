from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from ..database.connection import db

logger = structlog.get_logger()

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: Decimal
    max_position_size: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal
    var_1d: Decimal  # 1-day Value at Risk
    concentration_risk: Dict[str, Decimal]
    correlation_risk: Decimal

@dataclass
class RiskViolation:
    """Risk rule violation"""
    rule_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    current_value: Decimal
    threshold: Decimal
    symbol: Optional[str] = None
    action_required: str = ""

class PortfolioRiskManager:
    """Advanced portfolio risk management"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Risk limits from config
        self.max_total_exposure = Decimal(str(config.get('max_total_exposure', 0.95)))
        self.max_position_size = Decimal(str(config.get('max_position_size', 0.1)))
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', 0.05)))
        self.max_drawdown = Decimal(str(config.get('max_drawdown', 0.2)))
        self.max_correlation = Decimal(str(config.get('max_correlation', 0.8)))
        self.concentration_limit = Decimal(str(config.get('concentration_limit', 0.3)))
        
        # VaR parameters
        self.var_confidence = 0.95
        self.var_holding_period = 1  # days
        
    async def evaluate_portfolio_risk(
        self, 
        portfolio_value: Decimal,
        positions: Dict[str, Dict],
        market_data: Dict[str, Dict]
    ) -> Tuple[RiskMetrics, List[RiskViolation]]:
        """Comprehensive portfolio risk evaluation"""
        
        violations = []
        
        # Calculate current exposures
        total_exposure = self._calculate_total_exposure(positions, portfolio_value)
        
        # Position size checks
        position_violations = await self._check_position_sizes(positions, portfolio_value)
        violations.extend(position_violations)
        
        # Drawdown checks
        drawdown_violations = await self._check_drawdown_limits()
        violations.extend(drawdown_violations)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(positions)
        concentration_violations = self._check_concentration_limits(concentration_risk)
        violations.extend(concentration_violations)
        
        # Calculate VaR
        var_1d = await self._calculate_var(positions, market_data)
        
        # Correlation risk (simplified)
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Current drawdown
        current_dd, max_dd = await self._get_drawdown_metrics()
        
        metrics = RiskMetrics(
            total_exposure=total_exposure,
            max_position_size=max([
                abs(pos['quantity'] * pos['current_price']) / portfolio_value 
                for pos in positions.values()
            ] or [Decimal('0')]),
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            var_1d=var_1d,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk
        )
        
        # Log violations
        for violation in violations:
            await self._log_risk_violation(violation)
        
        return metrics, violations
    
    def _calculate_total_exposure(
        self, 
        positions: Dict[str, Dict], 
        portfolio_value: Decimal
    ) -> Decimal:
        """Calculate total portfolio exposure"""
        total_exposure = Decimal('0')
        
        for position in positions.values():
            position_value = abs(
                Decimal(str(position['quantity'])) * 
                Decimal(str(position['current_price']))
            )
            total_exposure += position_value
        
        return total_exposure / portfolio_value if portfolio_value > 0 else Decimal('0')
    
    async def _check_position_sizes(
        self, 
        positions: Dict[str, Dict], 
        portfolio_value: Decimal
    ) -> List[RiskViolation]:
        """Check individual position size limits"""
        violations = []
        
        for symbol, position in positions.items():
            position_value = abs(
                Decimal(str(position['quantity'])) * 
                Decimal(str(position['current_price']))
            )
            position_percentage = position_value / portfolio_value
            
            if position_percentage > self.max_position_size:
                violations.append(RiskViolation(
                    rule_type="POSITION_SIZE_LIMIT",
                    severity="HIGH",
                    current_value=position_percentage,
                    threshold=self.max_position_size,
                    symbol=symbol,
                    action_required="REDUCE_POSITION"
                ))
        
        return violations
    
    async def _check_drawdown_limits(self) -> List[RiskViolation]:
        """Check portfolio drawdown limits"""
        violations = []
        
        # Get current drawdown
        current_dd, max_dd = await self._get_drawdown_metrics()
        
        if current_dd > self.max_daily_loss:
            violations.append(RiskViolation(
                rule_type="DAILY_LOSS_LIMIT",
                severity="CRITICAL",
                current_value=current_dd,
                threshold=self.max_daily_loss,
                action_required="STOP_TRADING"
            ))
        
        if max_dd > self.max_drawdown:
            violations.append(RiskViolation(
                rule_type="MAX_DRAWDOWN_LIMIT",
                severity="HIGH",
                current_value=max_dd,
                threshold=self.max_drawdown,
                action_required="REDUCE_EXPOSURE"
            ))
        
        return violations
    
    def _calculate_concentration_risk(
        self, 
        positions: Dict[str, Dict]
    ) -> Dict[str, Decimal]:
        """Calculate concentration by asset, sector, etc."""
        concentration = {}
        
        total_value = sum(
            abs(Decimal(str(pos['quantity'])) * Decimal(str(pos['current_price'])))
            for pos in positions.values()
        )
        
        if total_value == 0:
            return concentration
        
        # By symbol (this is basic - could extend to sectors, asset classes, etc.)
        for symbol, position in positions.items():
            position_value = abs(
                Decimal(str(position['quantity'])) * 
                Decimal(str(position['current_price']))
            )
            concentration[symbol] = position_value / total_value
        
        return concentration
    
    def _check_concentration_limits(
        self, 
        concentration_risk: Dict[str, Decimal]
    ) -> List[RiskViolation]:
        """Check concentration limits"""
        violations = []
        
        for symbol, concentration in concentration_risk.items():
            if concentration > self.concentration_limit:
                violations.append(RiskViolation(
                    rule_type="CONCENTRATION_LIMIT",
                    severity="MEDIUM",
                    current_value=concentration,
                    threshold=self.concentration_limit,
                    symbol=symbol,
                    action_required="DIVERSIFY"
                ))
        
        return violations
    
    async def _calculate_var(
        self, 
        positions: Dict[str, Dict], 
        market_data: Dict[str, Dict]
    ) -> Decimal:
        """Calculate 1-day Value at Risk (simplified)"""
        # This is a basic VaR calculation
        # In production, you'd use more sophisticated methods
        
        total_var = Decimal('0')
        
        for symbol, position in positions.items():
            if symbol in market_data:
                # Get recent price volatility (simplified)
                price_volatility = await self._get_price_volatility(symbol, days=30)
                
                position_value = abs(
                    Decimal(str(position['quantity'])) * 
                    Decimal(str(position['current_price']))
                )
                
                # VaR = position_value * volatility * confidence_multiplier
                # Using 1.65 for 95% confidence (normal distribution)
                position_var = position_value * price_volatility * Decimal('1.65')
                total_var += position_var
        
        return total_var
    
    async def _get_price_volatility(self, symbol: str, days: int = 30) -> Decimal:
        """Calculate historical price volatility"""
        query = """
        SELECT close_price
        FROM market_data
        WHERE symbol = $1 
        AND interval_type = '1d'
        AND timestamp >= $2
        ORDER BY timestamp ASC
        """
        
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        rows = await db.fetch_all(query, symbol, start_time)
        
        if len(rows) < 2:
            return Decimal('0.02')  # Default 2% volatility
        
        prices = [Decimal(str(row['close_price'])) for row in rows]
        returns = []
        
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(float(daily_return))
        
        if not returns:
            return Decimal('0.02')
        
        # Calculate standard deviation
        import statistics
        volatility = Decimal(str(statistics.stdev(returns)))
        
        return volatility
    
    def _calculate_correlation_risk(self, positions: Dict[str, Dict]) -> Decimal:
        """Calculate portfolio correlation risk (simplified)"""
        # This is simplified - in production, you'd calculate actual correlations
        # between assets based on historical data
        
        if len(positions) <= 1:
            return Decimal('0')
        
        # Simple proxy: if holding correlated assets (e.g., BTC and ETH)
        crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        crypto_positions = [s for s in positions.keys() if s in crypto_symbols]
        
        if len(crypto_positions) > 1:
            return Decimal('0.7')  # Assume 70% correlation between crypto assets
        
        return Decimal('0.3')  # Lower correlation for diverse assets
    
    async def _get_drawdown_metrics(self) -> Tuple[Decimal, Decimal]:
        """Get current and maximum drawdown"""
        # This would typically look at portfolio equity curve
        # For now, return simplified metrics
        
        query = """
        SELECT total_pnl
        FROM strategy_performance
        WHERE date >= $1
        ORDER BY date DESC
        LIMIT 30
        """
        
        start_date = datetime.now().date() - timedelta(days=30)
        rows = await db.fetch_all(query, start_date)
        
        if not rows:
            return Decimal('0'), Decimal('0')
        
        pnls = [Decimal(str(row['total_pnl'])) for row in rows]
        
        # Calculate drawdown from peak
        peak = max(pnls) if pnls else Decimal('0')
        current = pnls[0] if pnls else Decimal('0')
        
        current_dd = (peak - current) / peak if peak > 0 else Decimal('0')
        
        # Max drawdown calculation (simplified)
        max_dd = Decimal('0')
        running_peak = Decimal('0')
        
        for pnl in reversed(pnls):
            if pnl > running_peak:
                running_peak = pnl
            
            if running_peak > 0:
                dd = (running_peak - pnl) / running_peak
                if dd > max_dd:
                    max_dd = dd
        
        return current_dd, max_dd
    
    async def _log_risk_violation(self, violation: RiskViolation):
        """Log risk violation to database"""
        query = """
        INSERT INTO risk_events 
        (event_type, severity, symbol, description, current_value, threshold_value, action_taken)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        description = f"{violation.rule_type}: {violation.current_value:.4f} exceeds {violation.threshold:.4f}"
        
        await db.execute(
            query,
            violation.rule_type,
            violation.severity,
            violation.symbol,
            description,
            float(violation.current_value),
            float(violation.threshold),
            violation.action_required
        )
        
        logger.warning("Risk violation detected",
                      rule_type=violation.rule_type,
                      severity=violation.severity,
                      symbol=violation.symbol,
                      current_value=violation.current_value,
                      threshold=violation.threshold)
    
    async def should_allow_trade(
        self, 
        symbol: str, 
        side: str, 
        quantity: Decimal,
        price: Decimal,
        portfolio_value: Decimal,
        current_positions: Dict[str, Dict]
    ) -> Tuple[bool, str]:
        """Determine if a trade should be allowed based on risk rules"""
        
        trade_value = quantity * price
        trade_percentage = trade_value / portfolio_value
        
        # Position size check
        if trade_percentage > self.max_position_size:
            return False, f"Trade size {trade_percentage:.2%} exceeds limit {self.max_position_size:.2%}"
        
        # Check if adding to existing position violates limits
        if symbol in current_positions:
            current_pos = current_positions[symbol]
            current_value = abs(
                Decimal(str(current_pos['quantity'])) * 
                Decimal(str(current_pos['current_price']))
            )
            
            if side == 'BUY' and current_pos['quantity'] > 0:
                # Adding to long position
                new_total_value = current_value + trade_value
                new_percentage = new_total_value / portfolio_value
                
                if new_percentage > self.max_position_size:
                    return False, f"Combined position {new_percentage:.2%} would exceed limit"
        
        # Check daily loss limits
        current_dd, _ = await self._get_drawdown_metrics()
        if current_dd > self.max_daily_loss:
            return False, f"Daily loss limit {self.max_daily_loss:.2%} already exceeded"
        
        return True, "Trade approved"
