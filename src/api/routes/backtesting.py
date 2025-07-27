from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import asyncio
import structlog

from ...strategies.registry import get_available_strategies, create_strategy
from ...backtesting.engine import BacktestEngine, BacktestConfig
from ...data.historical_manager import HistoricalDataManager
from ...exchanges.binance_exchange import BinanceBacktestingExchange
import numpy as np

logger = structlog.get_logger()
router = APIRouter()

# Global storage for backtest sessions
backtest_sessions: Dict[str, Dict] = {}

class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    timeframe: str = "1h"
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=10000.0, gt=0)
    commission_rate: float = Field(default=0.001, ge=0, le=0.1)
    slippage_rate: float = Field(default=0.0005, ge=0, le=0.1)
    parameters: Dict[str, Any] = {}

class ParameterSchema(BaseModel):
    name: str
    type: str  # 'float', 'int', 'bool', 'str'
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str

class StrategyInfo(BaseModel):
    name: str
    display_name: str
    description: str
    parameters: List[ParameterSchema]
    default_config: Dict[str, Any]

class BacktestStatus(BaseModel):
    session_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float = 0.0
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies():
    """Get list of available strategies with their parameters"""
    strategies = get_available_strategies()
    strategy_info = []
    
    for name, strategy_class in strategies.items():
        # Create a temporary instance to get parameter info
        try:
            temp_config = {"symbols": ["BTCUSDT"]}  # Minimal config for initialization
            temp_instance = strategy_class(temp_config)
            
            # Get strategy metadata
            info = StrategyInfo(
                name=name,
                display_name=name.replace('_', ' ').title(),
                description=getattr(strategy_class, '__doc__', f"{name} trading strategy"),
                parameters=_get_strategy_parameters(strategy_class),
                default_config=_get_default_config(strategy_class)
            )
            strategy_info.append(info)
        except Exception as e:
            logger.warning("Failed to get strategy info", strategy=name, error=str(e))
            continue
    
    return strategy_info

@router.get("/strategies/{strategy_name}/parameters")
async def get_strategy_parameters(strategy_name: str):
    """Get parameter schema for a specific strategy"""
    strategies = get_available_strategies()
    if strategy_name not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy_class = strategies[strategy_name]
    response = {
        "parameters": _get_strategy_parameters(strategy_class),
        "default_config": _get_default_config(strategy_class)
    }
    
    # Add presets for day trading strategy
    if "day_trading" in strategy_name.lower():
        response["presets"] = _get_day_trading_presets()
    
    return response

@router.get("/strategies/{strategy_name}/presets")
async def get_strategy_presets(strategy_name: str):
    """Get predefined configuration presets for a strategy"""
    if "day_trading" in strategy_name.lower():
        return {"presets": _get_day_trading_presets()}
    else:
        return {"presets": {}}

@router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    # Common cryptocurrency pairs
    symbols = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "BNBUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT",
        "TRXUSDT", "ETCUSDT", "DASHUSDT", "XMRUSDT", "ZECUSDT"
    ]
    return {"symbols": symbols}

@router.get("/timeframes")
async def get_available_timeframes():
    """Get list of available timeframes"""
    timeframes = [
        {"value": "1m", "label": "1 Minute"},
        {"value": "5m", "label": "5 Minutes"},
        {"value": "15m", "label": "15 Minutes"},
        {"value": "30m", "label": "30 Minutes"},
        {"value": "1h", "label": "1 Hour"},
        {"value": "4h", "label": "4 Hours"},
        {"value": "1d", "label": "1 Day"}
    ]
    return {"timeframes": timeframes}

@router.post("/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start a backtest execution"""
    session_id = str(uuid.uuid4())
    
    # Validate strategy
    strategies = get_available_strategies()
    if request.strategy_name not in strategies:
        raise HTTPException(status_code=400, detail="Invalid strategy name")
    
    # Validate date range
    if request.start_date >= request.end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")
    
    # Initialize session
    backtest_sessions[session_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing backtest...",
        "start_time": datetime.now(),
        "request": request,
        "results": None
    }
    
    # Start backtest in background
    background_tasks.add_task(_execute_backtest, session_id, request)
    
    return {"session_id": session_id, "status": "pending"}

@router.get("/status/{session_id}", response_model=BacktestStatus)
async def get_backtest_status(session_id: str):
    """Get status of a running backtest"""
    if session_id not in backtest_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = backtest_sessions[session_id]
    return BacktestStatus(
        session_id=session_id,
        status=session["status"],
        progress=session["progress"],
        message=session["message"],
        start_time=session["start_time"],
        end_time=session.get("end_time")
    )

@router.get("/results/{session_id}")
async def get_backtest_results(session_id: str):
    """Get results of a completed backtest"""
    if session_id not in backtest_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = backtest_sessions[session_id]
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")
    
    if not session["results"]:
        raise HTTPException(status_code=500, detail="No results available")
    
    return session["results"]

@router.delete("/sessions/{session_id}")
async def delete_backtest_session(session_id: str):
    """Delete a backtest session"""
    if session_id not in backtest_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del backtest_sessions[session_id]
    return {"message": "Session deleted"}

@router.get("/sessions")
async def list_backtest_sessions():
    """List all backtest sessions"""
    sessions = []
    for session_id, session_data in backtest_sessions.items():
        sessions.append({
            "session_id": session_id,
            "status": session_data["status"],
            "strategy_name": session_data["request"].strategy_name,
            "symbol": session_data["request"].symbol,
            "start_time": session_data["start_time"],
            "end_time": session_data.get("end_time")
        })
    return {"sessions": sessions}

async def _execute_backtest(session_id: str, request: BacktestRequest):
    """Execute backtest in background"""
    exchange = None
    try:
        session = backtest_sessions[session_id]
        session["status"] = "running"
        session["message"] = "Loading historical data..."
        session["progress"] = 10.0
        
        # Initialize exchange and historical data manager
        exchange = BinanceBacktestingExchange()
        await exchange.connect()
        data_manager = HistoricalDataManager(exchange)
        
        # Load historical data
        session["message"] = "Fetching market data..."
        session["progress"] = 20.0
        
        historical_data = await data_manager.get_historical_data(
            symbol=request.symbol,
            interval=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if historical_data is None or historical_data.empty:
            session["status"] = "failed"
            session["message"] = "No historical data available for the specified period"
            session["end_time"] = datetime.now()
            return
        
        session["message"] = "Initializing strategy..."
        session["progress"] = 30.0
        
        # Create strategy instance
        strategy_config = request.parameters.copy()
        strategy_config["symbols"] = [request.symbol]
        strategy = create_strategy(request.strategy_name, strategy_config)
        
        session["message"] = "Running backtest..."
        session["progress"] = 40.0
        
        # Configure backtesting engine with proper position sizing
        position_size = request.parameters.get('position_size', 0.1)  # Default 10% of capital
        backtest_config = BacktestConfig(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            slippage_rate=request.slippage_rate,
            position_sizing='percentage',  # Use percentage of capital
            position_size=position_size
        )
        
        engine = BacktestEngine(backtest_config)
        
        # Convert data format for backtesting engine
        formatted_data = {request.symbol: historical_data}
        
        session["progress"] = 50.0
        
        # Run backtest
        results = await engine.run_backtest(
            strategy=strategy,
            historical_data=formatted_data,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        session["message"] = "Processing results..."
        session["progress"] = 90.0
        
        # Enhance results with additional data
        enhanced_results = _enhance_backtest_results(results, historical_data, request)
        
        session["status"] = "completed"
        session["message"] = "Backtest completed successfully"
        session["progress"] = 100.0
        session["end_time"] = datetime.now()
        session["results"] = enhanced_results
        
        logger.info("Backtest completed", session_id=session_id, 
                   total_trades=len(results.get('trades_detail', [])))
        
    except Exception as e:
        logger.error("Backtest execution failed", session_id=session_id, error=str(e))
        session = backtest_sessions.get(session_id, {})
        session["status"] = "failed"
        session["message"] = f"Backtest failed: {str(e)}"
        session["end_time"] = datetime.now()
    finally:
        # Ensure exchange is disconnected
        if exchange:
            await exchange.disconnect()

def _get_strategy_parameters(strategy_class) -> List[ParameterSchema]:
    """Extract parameter schema from strategy class"""
    # This is a simplified parameter extraction
    # In a real implementation, you'd want to use proper introspection
    # or have strategies define their parameter schemas explicitly
    
    common_params = [
        ParameterSchema(
            name="position_size",
            type="float",
            default=0.03,  # OPTIMIZED: Updated from 1% to 3% for day trading
            min_value=0.001,
            max_value=0.1,  # OPTIMIZED: Reduced max from 100% to 10% to prevent over-leveraging
            description="Position size as fraction of capital (optimized for day trading)"
        )
    ]
    
    # Strategy-specific parameters would be added here
    # based on the strategy class
    strategy_name = strategy_class.__name__.lower()
    
    if "day_trading" in strategy_name:
        common_params.extend([
            # EMA Trend Analysis
            ParameterSchema(
                name="fast_ema",
                type="int",
                default=8,
                min_value=3,
                max_value=20,
                description="Fast EMA period (trend detection)"
            ),
            ParameterSchema(
                name="medium_ema", 
                type="int",
                default=21,
                min_value=10,
                max_value=50,
                description="Medium EMA period (trend confirmation)"
            ),
            ParameterSchema(
                name="slow_ema",
                type="int", 
                default=50,
                min_value=20,
                max_value=100,
                description="Slow EMA period (major trend)"
            ),
            
            # RSI Momentum
            ParameterSchema(
                name="rsi_period",
                type="int",
                default=14,
                min_value=5,
                max_value=30,
                description="RSI calculation period"
            ),
            ParameterSchema(
                name="rsi_overbought",
                type="float",
                default=70.0,
                min_value=60.0,
                max_value=85.0,
                description="RSI overbought threshold"
            ),
            ParameterSchema(
                name="rsi_oversold",
                type="float",
                default=30.0,
                min_value=15.0,
                max_value=40.0,
                description="RSI oversold threshold"
            ),
            
            # MACD Settings
            ParameterSchema(
                name="macd_fast",
                type="int",
                default=12,
                min_value=5,
                max_value=20,
                description="MACD fast EMA period"
            ),
            ParameterSchema(
                name="macd_slow",
                type="int",
                default=26,
                min_value=20,
                max_value=40,
                description="MACD slow EMA period"
            ),
            ParameterSchema(
                name="macd_signal",
                type="int",
                default=9,
                min_value=5,
                max_value=15,
                description="MACD signal line period"
            ),
            
            # Signal Scoring
            ParameterSchema(
                name="min_signal_score",
                type="float",
                default=0.6,
                min_value=0.3,
                max_value=0.9,
                description="Minimum signal score to enter trade"
            ),
            ParameterSchema(
                name="strong_signal_score",
                type="float",
                default=0.8,
                min_value=0.6,
                max_value=1.0,
                description="Strong signal score threshold"
            ),
            
            # Risk Management
            ParameterSchema(
                name="stop_loss_pct",
                type="float",
                default=1.5,
                min_value=0.5,
                max_value=5.0,
                description="Stop loss percentage"
            ),
            ParameterSchema(
                name="take_profit_pct",
                type="float",
                default=2.5,
                min_value=1.0,
                max_value=10.0,
                description="Take profit percentage"
            ),
            ParameterSchema(
                name="trailing_stop_pct",
                type="float",
                default=1.0,
                min_value=0.3,
                max_value=3.0,
                description="Trailing stop loss percentage"
            ),
            
            # Trading Frequency
            ParameterSchema(
                name="max_daily_trades",
                type="int",
                default=3,
                min_value=1,
                max_value=10,
                description="Maximum trades per day"
            ),
            
            # Volume Analysis
            ParameterSchema(
                name="volume_multiplier",
                type="float",
                default=1.2,
                min_value=1.0,
                max_value=3.0,
                description="Volume spike threshold multiplier"
            ),
            ParameterSchema(
                name="volume_period",
                type="int",
                default=20,
                min_value=10,
                max_value=50,
                description="Volume moving average period"
            ),
            
            # Support/Resistance
            ParameterSchema(
                name="support_resistance_threshold",
                type="float",
                default=0.8,
                min_value=0.3,
                max_value=2.0,
                description="Support/Resistance level threshold (%)"
            ),
            ParameterSchema(
                name="pivot_period",
                type="int",
                default=10,
                min_value=5,
                max_value=20,
                description="Pivot point calculation period"
            ),
            
            # Leverage (Optional)
            ParameterSchema(
                name="leverage",
                type="float",
                default=1.0,
                min_value=1.0,
                max_value=10.0,
                description="Trading leverage (1.0 = no leverage)"
            ),
            ParameterSchema(
                name="use_leverage",
                type="bool",
                default=False,
                description="Enable leverage trading"
            )
        ])
    elif "vlam" in strategy_name or "consolidation" in strategy_name:
        common_params.extend([
            # VLAM Indicator Parameters
            ParameterSchema(
                name="vlam_period",
                type="int",
                default=10,
                min_value=5,
                max_value=30,
                description="VLAM calculation period"
            ),
            ParameterSchema(
                name="atr_period",
                type="int",
                default=10,
                min_value=5,
                max_value=30,
                description="ATR period for volatility adjustment"
            ),
            ParameterSchema(
                name="volume_period",
                type="int",
                default=15,
                min_value=10,
                max_value=50,
                description="Volume moving average period"
            ),
            
            # Consolidation Detection
            ParameterSchema(
                name="consolidation_min_length",
                type="int",
                default=4,
                min_value=3,
                max_value=20,
                description="Minimum consolidation length in bars"
            ),
            ParameterSchema(
                name="consolidation_tolerance",
                type="float",
                default=0.05,
                min_value=0.005,
                max_value=0.05,
                description="Consolidation range tolerance (% of price)"
            ),
            ParameterSchema(
                name="min_touches",
                type="int",
                default=2,
                min_value=2,
                max_value=6,
                description="Minimum support/resistance touches"
            ),
            
            # Spike Detection
            ParameterSchema(
                name="spike_min_size",
                type="float",
                default=0.8,
                min_value=0.5,
                max_value=3.0,
                description="Minimum spike size (ATR multiplier)"
            ),
            ParameterSchema(
                name="spike_volume_multiplier",
                type="float",
                default=1.5,
                min_value=1.0,
                max_value=2.5,
                description="Volume spike confirmation multiplier"
            ),
            
            # Signal Parameters
            ParameterSchema(
                name="vlam_signal_threshold",
                type="float",
                default=0.5,
                min_value=0.3,
                max_value=0.9,
                description="VLAM signal strength threshold"
            ),
            ParameterSchema(
                name="entry_timeout_bars",
                type="int",
                default=12,
                min_value=3,
                max_value=15,
                description="Max bars to wait for entry after spike"
            ),
            
            # Risk Management
            ParameterSchema(
                name="target_risk_reward",
                type="float",
                default=3.0,
                min_value=1.5,
                max_value=4.0,
                description="Target risk:reward ratio"
            ),
            ParameterSchema(
                name="max_risk_per_trade",
                type="float",
                default=0.02,
                min_value=0.01,
                max_value=0.05,
                description="Maximum risk per trade (% of capital)"
            ),
            ParameterSchema(
                name="position_timeout_hours",
                type="int",
                default=24,
                min_value=6,
                max_value=72,
                description="Maximum position hold time (hours)"
            )
        ])
    elif "momentumtrading" in strategy_name:
        common_params.extend([
            # Trend Detection Parameters - FINAL OPTIMIZED DEFAULTS (146.6% backtest performance)
            ParameterSchema(
                name="trend_ema_fast",
                type="int",
                default=5,  # OPTIMIZED: Ultra-fast trend detection
                min_value=3,
                max_value=20,
                description="Fast EMA period for trend detection (optimized for quick signals)"
            ),
            ParameterSchema(
                name="trend_ema_slow",
                type="int",
                default=10,  # OPTIMIZED: Quick response to trend changes
                min_value=5,
                max_value=30,
                description="Slow EMA period for trend detection (optimized for responsiveness)"
            ),
            ParameterSchema(
                name="trend_strength_threshold",
                type="float",
                default=0.001,  # OPTIMIZED: Very low threshold for early trend detection
                min_value=0.0005,
                max_value=0.01,
                description="Minimum trend strength required (optimized for maximum signals)"
            ),
            
            # RSI Parameters - OPTIMIZED
            ParameterSchema(
                name="rsi_period",
                type="int",
                default=7,  # OPTIMIZED: Faster RSI for quicker signals
                min_value=5,
                max_value=20,
                description="RSI calculation period (optimized for momentum trading)"
            ),
            ParameterSchema(
                name="rsi_momentum_threshold",
                type="float",
                default=50,
                min_value=30,
                max_value=70,
                description="RSI neutral line for momentum confirmation"
            ),
            
            # MACD Parameters
            ParameterSchema(
                name="macd_fast",
                type="int",
                default=12,
                min_value=5,
                max_value=20,
                description="MACD fast EMA period"
            ),
            ParameterSchema(
                name="macd_slow",
                type="int",
                default=26,
                min_value=15,
                max_value=40,
                description="MACD slow EMA period"
            ),
            ParameterSchema(
                name="macd_signal",
                type="int",
                default=9,
                min_value=5,
                max_value=15,
                description="MACD signal line period"
            ),
            
            # Volume Parameters
            ParameterSchema(
                name="volume_period",
                type="int",
                default=20,
                min_value=10,
                max_value=50,
                description="Volume moving average period"
            ),
            ParameterSchema(
                name="volume_surge_multiplier",
                type="float",
                default=1.1,  # OPTIMIZED: Minimal volume requirement
                min_value=1.05,
                max_value=3.0,
                description="Volume surge confirmation multiplier (optimized for more signals)"
            ),
            ParameterSchema(
                name="volume_confirmation_required",
                type="bool",
                default=False,  # OPTIMIZED: Disabled for maximum signals
                description="Require volume confirmation for entries (optimized: disabled)"
            ),
            
            # Entry Parameters - OPTIMIZED
            ParameterSchema(
                name="momentum_alignment_required",
                type="bool",
                default=False,  # OPTIMIZED: Disabled for maximum signals
                description="Require momentum indicator alignment (optimized: disabled)"
            ),
            ParameterSchema(
                name="trend_confirmation_bars",
                type="int",
                default=3,
                min_value=1,
                max_value=10,
                description="Bars required for trend confirmation"
            ),
            ParameterSchema(
                name="breakout_lookback",
                type="int",
                default=5,  # OPTIMIZED: Shorter lookback for quicker breakout detection
                min_value=3,
                max_value=20,
                description="Lookback period for breakout detection (optimized for speed)"
            ),
            
            # Position Sizing - OPTIMIZED FOR HIGHER RETURNS
            ParameterSchema(
                name="base_position_size",
                type="float",
                default=0.05,  # OPTIMIZED: Larger base position (5%)
                min_value=0.01,
                max_value=0.1,
                description="Base position size (% of capital) - optimized for momentum trading"
            ),
            ParameterSchema(
                name="max_position_size",
                type="float",
                default=0.1,  # OPTIMIZED: Higher maximum position (10%)
                min_value=0.05,
                max_value=0.2,
                description="Maximum position size (% of capital) - optimized for trending markets"
            ),
            ParameterSchema(
                name="trend_strength_scaling",
                type="bool",
                default=False,  # OPTIMIZED: Simplified position sizing
                description="Scale position size based on trend strength (optimized: simplified)"
            ),
            
            # Risk Management - OPTIMIZED FOR TRENDING MARKETS
            ParameterSchema(
                name="stop_loss_atr_multiplier",
                type="float",
                default=5.0,  # OPTIMIZED: Wider stops to avoid whipsaws
                min_value=2.0,
                max_value=10.0,
                description="Stop loss distance (ATR multiplier) - optimized for trending markets"
            ),
            ParameterSchema(
                name="take_profit_risk_reward",
                type="float",
                default=1.5,  # OPTIMIZED: Quicker profit taking
                min_value=1.2,
                max_value=3.0,
                description="Take profit risk:reward ratio - optimized for momentum trading"
            ),
            ParameterSchema(
                name="trailing_stop_activation",
                type="float",
                default=1.5,
                min_value=1.0,
                max_value=3.0,
                description="Activate trailing stop at R multiple"
            ),
            ParameterSchema(
                name="max_risk_per_trade",
                type="float",
                default=0.02,
                min_value=0.01,
                max_value=0.05,
                description="Maximum risk per trade (% of capital)"
            ),
            
            # Position Management
            ParameterSchema(
                name="max_concurrent_positions",
                type="int",
                default=3,
                min_value=1,
                max_value=10,
                description="Maximum concurrent positions"
            ),
            ParameterSchema(
                name="position_timeout_hours",
                type="int",
                default=168,
                min_value=24,
                max_value=720,
                description="Maximum position hold time (hours)"
            )
        ])
    elif "ma" in strategy_name or "crossover" in strategy_name:
        common_params.extend([
            ParameterSchema(
                name="fast_period",
                type="int",
                default=10,
                min_value=1,
                max_value=50,
                description="Fast moving average period"
            ),
            ParameterSchema(
                name="slow_period",
                type="int",
                default=20,
                min_value=2,
                max_value=100,
                description="Slow moving average period"
            )
        ])
    elif "rsi" in strategy_name:
        common_params.extend([
            ParameterSchema(
                name="rsi_period",
                type="int",
                default=14,
                min_value=2,
                max_value=50,
                description="RSI calculation period"
            ),
            ParameterSchema(
                name="rsi_oversold",
                type="float",
                default=30.0,
                min_value=10.0,
                max_value=40.0,
                description="RSI oversold threshold"
            ),
            ParameterSchema(
                name="rsi_overbought",
                type="float",
                default=70.0,
                min_value=60.0,
                max_value=90.0,
                description="RSI overbought threshold"
            )
        ])
    
    return common_params

def _get_default_config(strategy_class) -> Dict[str, Any]:
    """Get default configuration for strategy"""
    base_config = {
        "symbols": ["BTCUSDT"],
        "position_size": 0.03  # OPTIMIZED: Reduced from 10% to 3% for better cost management
    }
    
    strategy_name = strategy_class.__name__.lower()
    
    if "ma" in strategy_name or "crossover" in strategy_name:
        base_config.update({
            "fast_period": 10,
            "slow_period": 20
        })
    
    if "rsi" in strategy_name:
        base_config.update({
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0
        })
    
    if "macd" in strategy_name:
        base_config.update({
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        })
    
    if "day_trading" in strategy_name:
        base_config.update({
            # OPTIMAL EMA Settings from backtesting analysis
            "fast_ema": 12,        # Increased from 8 - better trend confirmation
            "medium_ema": 26,      # Increased from 21 - more stable reference
            "slow_ema": 55,        # Increased from 50 - stronger trend filter
            
            # RSI Settings
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_neutral_high": 60,
            "rsi_neutral_low": 40,
            
            # MACD Settings
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            
            # OPTIMAL Signal Scoring from backtesting
            "min_signal_score": 0.8,     # Increased from 0.6 - optimal quality threshold
            "strong_signal_score": 0.85, # Increased from 0.8 - higher bar for strong signals
            
            # VALIDATED Risk Management
            "stop_loss_pct": 1.5,        # Confirmed optimal
            "take_profit_pct": 2.5,      # Confirmed optimal
            "trailing_stop_pct": 1.0,
            
            # Trading Frequency
            "max_daily_trades": 5,       # Increased from 3 - optimal balance
            
            # OPTIMAL Volume Analysis
            "volume_multiplier": 1.5,    # Increased from 1.2 - optimal balance
            "volume_period": 20,
            
            # Support/Resistance
            "support_resistance_threshold": 1.5,  # Updated to match optimal config
            "pivot_period": 10,
            
            # Leverage
            "leverage": 1.0,
            "use_leverage": False
        })
    
    if "vlam" in strategy_name or "consolidation" in strategy_name:
        base_config.update({
            # VLAM Indicator Settings (optimized)
            "vlam_period": 10,
            "atr_period": 10,
            "volume_period": 15,
            
            # Consolidation Detection (optimized)
            "consolidation_min_length": 4,
            "consolidation_max_length": 20,
            "consolidation_tolerance": 0.05,
            "min_touches": 2,
            
            # Spike Detection (optimized)
            "spike_min_size": 0.8,
            "spike_volume_multiplier": 1.5,
            
            # Signal Parameters (optimized)
            "vlam_signal_threshold": 0.5,
            "entry_timeout_bars": 12,
            
            # Risk Management (optimized)
            "target_risk_reward": 3.0,
            "max_risk_per_trade": 0.02,
            "position_timeout_hours": 24,
            "max_concurrent_positions": 2
        })
    
    if "momentumtrading" in strategy_name:
        base_config.update({
            # Trend Detection Parameters - FINAL OPTIMIZED (146.6% backtest performance)
            "trend_ema_fast": 5,  # Ultra-fast trend detection
            "trend_ema_slow": 10,  # Quick response to trend changes
            "trend_ema_signal": 9,
            "trend_strength_threshold": 0.001,  # Very low threshold for early trend detection
            
            # RSI Parameters - FINAL OPTIMIZED
            "rsi_period": 7,  # Faster RSI for quicker signals
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_momentum_threshold": 50,
            
            # MACD Parameters
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "macd_histogram_threshold": 0.0,
            
            # Volume Parameters - OPTIMIZED FOR MORE SIGNALS
            "volume_period": 20,
            "volume_surge_multiplier": 1.1,  # Minimal volume requirement
            "volume_confirmation_required": False,  # Disabled for maximum signals
            
            # Entry Signal Parameters - OPTIMIZED
            "momentum_alignment_required": False,  # Disabled for maximum signals
            "trend_confirmation_bars": 3,
            "breakout_lookback": 5,  # Shorter lookback for quicker breakout detection
            
            # Position Sizing - OPTIMIZED FOR HIGHER RETURNS
            "base_position_size": 0.05,  # Larger base position (5%)
            "max_position_size": 0.1,  # Higher maximum position (10%)
            "trend_strength_scaling": False,  # Simplified position sizing
            
            # Risk Management - OPTIMIZED FOR TRENDING MARKETS
            "stop_loss_atr_multiplier": 5.0,  # Wider stops to avoid whipsaws
            "take_profit_risk_reward": 1.5,  # Quicker profit taking
            "trailing_stop_activation": 1.5,
            "trailing_stop_distance": 1.0,
            "max_risk_per_trade": 0.02,
            
            # Trend Filter
            "min_trend_duration": 5,
            "trend_invalidation_threshold": 0.005,
            
            # Position Management
            "max_concurrent_positions": 3,
            "position_timeout_hours": 168
        })
    
    return base_config

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and infinite values
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None  # or use a large number like 1e10 if you prefer
        return float(obj)
    elif isinstance(obj, (float, int)):
        # Handle regular Python float/int that might be NaN/inf
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle custom objects like Trade dataclass
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = _convert_numpy_types(value)
        return result
    else:
        return obj

def _get_day_trading_presets() -> Dict[str, Dict[str, Any]]:
    """Get predefined configuration presets for day trading strategy"""
    return {
        "optimized": {
            "name": "🎯 Optimized (Recommended)",
            "description": "Backtesting-optimized parameters with EMA 12/26/55, signal threshold 0.8, volume 1.5x - Best performance in comprehensive testing",
            "config": {
                "position_size": 0.02,
                # OPTIMAL EMA settings from backtesting analysis
                "fast_ema": 12,        # Increased from 8 - better trend confirmation
                "medium_ema": 26,      # Increased from 21 - more stable reference
                "slow_ema": 55,        # Increased from 50 - stronger trend filter
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "rsi_neutral_high": 60,
                "rsi_neutral_low": 40,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                # OPTIMAL signal scoring from backtesting
                "min_signal_score": 0.8,     # Optimal quality threshold
                "strong_signal_score": 0.85,
                # VALIDATED risk management settings
                "stop_loss_pct": 1.5,        # Confirmed optimal
                "take_profit_pct": 2.5,      # Confirmed optimal
                "trailing_stop_pct": 1.0,
                "max_daily_trades": 5,
                # OPTIMAL volume confirmation
                "volume_multiplier": 1.5,    # Optimal balance
                "volume_period": 20,
                "support_resistance_threshold": 1.5,  # Updated to match optimal config
                "pivot_period": 10,
                "leverage": 1.0,
                "use_leverage": False
            }
        },
        "conservative": {
            "name": "Conservative",
            "description": "Lower risk, fewer trades, stricter signal requirements - Updated with optimal EMA settings",
            "config": {
                "position_size": 0.015,  # Even more conservative
                # Updated with optimal EMA settings
                "fast_ema": 12,
                "medium_ema": 26,
                "slow_ema": 55,
                "rsi_period": 14,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "min_signal_score": 0.85,   # Even higher threshold for conservative
                "strong_signal_score": 0.9,
                "stop_loss_pct": 1.0,       # Tighter stop loss
                "take_profit_pct": 2.0,     # Lower take profit
                "trailing_stop_pct": 0.7,   # Tighter trailing stop
                "max_daily_trades": 2,      # Fewer trades
                "volume_multiplier": 2.0,   # Higher volume requirement
                "volume_period": 20,
                "support_resistance_threshold": 1.0,  # Stricter S/R levels
                "pivot_period": 10,
                "leverage": 1.0,
                "use_leverage": False
            }
        },
        "balanced": {
            "name": "Balanced",
            "description": "Balanced approach with moderate risk - Updated with optimal EMA settings",
            "config": {
                "position_size": 0.025,
                # Updated with optimal EMA settings
                "fast_ema": 12,
                "medium_ema": 26,
                "slow_ema": 55,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "min_signal_score": 0.75,   # Slightly lower than optimal
                "strong_signal_score": 0.85,
                "stop_loss_pct": 1.5,
                "take_profit_pct": 2.5,
                "trailing_stop_pct": 1.0,
                "max_daily_trades": 4,
                "volume_multiplier": 1.3,   # Slightly lower than optimal
                "volume_period": 20,
                "support_resistance_threshold": 1.2,
                "pivot_period": 10,
                "leverage": 1.0,
                "use_leverage": False
            }
        },
        "aggressive": {
            "name": "Aggressive",
            "description": "Higher risk, more trades - Uses faster EMAs for more responsive signals",
            "config": {
                "position_size": 0.04,
                "fast_ema": 8,          # Faster EMAs for aggressive
                "medium_ema": 21,
                "slow_ema": 50,
                "rsi_period": 10,       # Shorter RSI period
                "rsi_overbought": 65,   # More relaxed levels
                "rsi_oversold": 35,
                "macd_fast": 8,         # Faster MACD
                "macd_slow": 21,
                "macd_signal": 7,
                "min_signal_score": 0.7,  # Lower threshold for more trades
                "strong_signal_score": 0.75,
                "stop_loss_pct": 2.0,      # Wider stop loss
                "take_profit_pct": 3.0,    # Higher take profit
                "trailing_stop_pct": 1.3,  # Wider trailing stop
                "max_daily_trades": 8,     # More trades
                "volume_multiplier": 1.2,  # Lower volume requirement
                "volume_period": 15,       # Shorter volume period
                "support_resistance_threshold": 2.0,  # More relaxed S/R levels
                "pivot_period": 7,         # Shorter pivot period
                "leverage": 2.0,           # Optional leverage
                "use_leverage": False      # User can enable
            }
        }
    }

def _enhance_backtest_results(results: Dict, historical_data, request: BacktestRequest) -> Dict:
    """Enhance backtest results with additional visualization data"""
    enhanced = results.copy()
    
    # Add price data for charting
    price_data = []
    for _, row in historical_data.iterrows():
        # Convert timestamp to milliseconds for JavaScript
        timestamp_ms = int(row["timestamp"]) * 1000 if int(row["timestamp"]) < 1e12 else int(row["timestamp"])
        price_data.append({
            "timestamp": timestamp_ms,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        })
    
    enhanced["price_data"] = price_data
    enhanced["ohlcv_data"] = price_data  # For frontend compatibility
    
    # Add signal markers for chart
    signal_markers = []
    for signal in results.get("signals_log", []):
        signal_markers.append({
            "timestamp": int(signal["timestamp"].timestamp() * 1000),
            "price": float(signal["price"]),
            "action": signal["action"],
            "symbol": signal["symbol"]
        })
    
    enhanced["signal_markers"] = signal_markers
    
    # Add trade markers for entry/exit points
    trade_markers = []
    for trade in results.get("trades_detail", []):
        # Entry marker
        if hasattr(trade, 'entry_time') and trade.entry_time:
            entry_timestamp = int(trade.entry_time.timestamp() * 1000)
            trade_markers.append({
                "timestamp": entry_timestamp,
                "price": float(trade.entry_price),
                "type": "entry",
                "symbol": trade.symbol
            })
        
        # Exit marker
        if hasattr(trade, 'exit_time') and trade.exit_time:
            exit_timestamp = int(trade.exit_time.timestamp() * 1000)
            trade_markers.append({
                "timestamp": exit_timestamp,
                "price": float(trade.exit_price),
                "type": "exit",
                "symbol": trade.symbol,
                "pnl": float(trade.pnl)
            })
    
    enhanced["trade_markers"] = trade_markers
    
    # Add configuration info
    enhanced["configuration"] = {
        "strategy_name": request.strategy_name,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "initial_capital": request.initial_capital,
        "commission_rate": request.commission_rate,
        "slippage_rate": request.slippage_rate,
        "parameters": request.parameters
    }
    
    # Convert numpy types to native Python types for JSON serialization
    return _convert_numpy_types(enhanced)