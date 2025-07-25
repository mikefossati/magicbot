import structlog
import logging
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

class DatabaseLogHandler(logging.Handler):
    """Custom log handler that stores logs in database"""
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
    
    def emit(self, record):
        """Emit log record to database"""
        try:
            # Only log WARNING and above to database to avoid spam
            if record.levelno >= logging.WARNING:
                # Create task for async operation but don't wait
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._store_log(record))
                except RuntimeError:
                    # No event loop running, skip database logging
                    pass
        except Exception:
            pass  # Don't let logging errors crash the app
    
    async def _store_log(self, record):
        """Store log record in database"""
        try:
            # Import here to avoid circular imports
            from ..database.connection import db
            
            query = """
            INSERT INTO system_logs 
            (level, logger, message, module, function, line_number, exception, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            metadata = {
                'thread': getattr(record, 'thread', None),
                'thread_name': getattr(record, 'threadName', None),
                'process': getattr(record, 'process', None),
                'process_name': getattr(record, 'processName', None),
            }
            
            # Add extra fields if available
            if hasattr(record, 'symbol'):
                metadata['symbol'] = record.symbol
            if hasattr(record, 'strategy'):
                metadata['strategy'] = record.strategy
            
            await db.execute(
                query,
                record.levelname,
                record.name,
                record.getMessage(),
                getattr(record, 'module', record.filename),
                getattr(record, 'funcName', ''),
                record.lineno,
                record.exc_text if record.exc_info else None,
                json.dumps(metadata)
            )
        except Exception as e:
            # Fall back to console logging
            print(f"Failed to store log in database: {e}")

def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog with compatible processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add console processor for development
    try:
        if sys.stdout.isatty():
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(structlog.processors.JSONRenderer())
    except:
        # Fallback to JSON renderer if sys.stdout check fails
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        today = datetime.now().strftime('%Y%m%d')
        file_handler = logging.FileHandler(
            log_dir / f"magicbot_{today}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Database handler (only try if we're not in setup mode)
    try:
        if 'setup_week2.py' not in ' '.join(sys.argv):
            db_handler = DatabaseLogHandler()
            root_logger.addHandler(db_handler)
    except Exception as e:
        logging.warning(f"Could not setup database logging: {e}")
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logging.info("Logging system initialized", extra={'level': log_level})

class TradingLogger:
    """Specialized logger for trading events"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def signal_generated(self, symbol: str, action: str, price: float, 
                        confidence: float, strategy: str):
        """Log signal generation"""
        self.logger.info("Signal generated",
                        symbol=symbol,
                        action=action,
                        price=price,
                        confidence=confidence,
                        strategy=strategy)
    
    def trade_executed(self, symbol: str, side: str, quantity: float,
                      price: float, order_id: str, strategy: str):
        """Log trade execution"""
        self.logger.info("Trade executed",
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        order_id=order_id,
                        strategy=strategy)
    
    def trade_closed(self, symbol: str, pnl: float, pnl_pct: float,
                    duration: str, strategy: str):
        """Log trade closure"""
        self.logger.info("Trade closed",
                        symbol=symbol,
                        pnl=pnl,
                        pnl_percentage=pnl_pct,
                        duration=duration,
                        strategy=strategy)
    
    def risk_violation(self, rule_type: str, severity: str, symbol: str,
                      current_value: float, threshold: float):
        """Log risk violation"""
        self.logger.warning("Risk violation",
                           rule_type=rule_type,
                           severity=severity,
                           symbol=symbol,
                           current_value=current_value,
                           threshold=threshold)
    
    def portfolio_update(self, total_value: float, unrealized_pnl: float,
                        position_count: int, exposure: float):
        """Log portfolio updates"""
        self.logger.info("Portfolio update",
                        total_value=total_value,
                        unrealized_pnl=unrealized_pnl,
                        position_count=position_count,
                        exposure=exposure)
    
    def strategy_performance(self, strategy: str, trades: int, win_rate: float,
                           total_pnl: float, sharpe_ratio: float):
        """Log strategy performance"""
        self.logger.info("Strategy performance",
                        strategy=strategy,
                        total_trades=trades,
                        win_rate=win_rate,
                        total_pnl=total_pnl,
                        sharpe_ratio=sharpe_ratio)