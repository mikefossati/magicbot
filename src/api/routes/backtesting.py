from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import asyncio
import structlog

from ...strategies.registry import get_available_strategies, create_strategy, get_strategy_info
from ...backtesting.engine import BacktestEngine
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
        try:
            # Use new architecture's parameter info method
            param_info = get_strategy_info(name)
            
            # Convert to API format
            api_parameters = []
            for param_name, param_def in param_info['parameters'].items():
                api_param = ParameterSchema(
                    name=param_name,
                    type=param_def['type'],
                    default=param_def['default'],
                    min_value=param_def.get('min_value'),
                    max_value=param_def.get('max_value'),
                    description=param_def['description']
                )
                api_parameters.append(api_param)
            
            # Create default config from required parameters
            default_config = {
                'symbols': ['BTCUSDT'],
                'position_size': 0.01
            }
            
            # Add defaults for required parameters
            for param_name in param_info['required_params']:
                if param_name not in default_config:
                    param_def = param_info['parameters'].get(param_name, {})
                    if param_def.get('default') is not None:
                        default_config[param_name] = param_def['default']
            
            info = StrategyInfo(
                name=name,
                display_name=name.replace('_', ' ').title(),
                description=getattr(strategy_class, '__doc__', f"{name} trading strategy"),
                parameters=api_parameters,
                default_config=default_config
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
    
    try:
        # Use new architecture's parameter info method
        param_info = get_strategy_info(strategy_name)
        
        # Convert to API format
        api_parameters = []
        for param_name, param_def in param_info['parameters'].items():
            api_param = {
                "name": param_name,
                "type": param_def['type'],
                "default": param_def['default'],
                "min_value": param_def.get('min_value'),
                "max_value": param_def.get('max_value'),
                "description": param_def['description'],
                "required": param_name in param_info['required_params']
            }
            api_parameters.append(api_param)
        
        # Create default config
        default_config = {
            'symbols': ['BTCUSDT'],
            'position_size': 0.01
        }
        
        # Add defaults for required parameters
        for param_name in param_info['required_params']:
            if param_name not in default_config:
                param_def = param_info['parameters'].get(param_name, {})
                if param_def.get('default') is not None:
                    default_config[param_name] = param_def['default']
        
        response = {
            "parameters": api_parameters,
            "default_config": default_config,
            "required_params": param_info['required_params'],
            "optional_params": param_info['optional_params']
        }
        
        
        return response
        
    except Exception as e:
        logger.error("Failed to get strategy parameters", strategy=strategy_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get strategy parameters: {str(e)}")

@router.get("/strategies/{strategy_name}/presets")
async def get_strategy_presets(strategy_name: str):
    """Get predefined configuration presets for a strategy"""
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
        
        # Use the timeframe from the frontend request
        fetch_interval = request.timeframe
        logger.info("Fetching historical data with frontend timeframe",
                   symbol=request.symbol,
                   requested_timeframe=request.timeframe,
                   fetch_interval=fetch_interval)
        
        historical_data = await data_manager.get_historical_data(
            symbol=request.symbol,
            interval=fetch_interval,
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
        
        # IMPORTANT: Always override timeframes with frontend value
        # Remove any existing timeframes from frontend params and set our own
        if "timeframes" in strategy_config:
            del strategy_config["timeframes"]
        strategy_config["timeframes"] = [request.timeframe]
        
        logger.info("Strategy config for backtest", 
                   strategy=request.strategy_name,
                   symbol=request.symbol,
                   timeframe_from_frontend=request.timeframe,
                   strategy_config_timeframes=strategy_config["timeframes"],
                   original_frontend_params=request.parameters)
        
        # Ensure numeric parameters are properly typed
        numeric_params = ['fast_period', 'slow_period', 'rsi_period', 'lookback_periods']
        for param in numeric_params:
            if param in strategy_config and isinstance(strategy_config[param], str):
                try:
                    strategy_config[param] = int(strategy_config[param])
                except ValueError:
                    logger.warning(f"Could not convert {param} to int: {strategy_config[param]}")
        
        float_params = ['position_size', 'stop_loss_pct', 'take_profit_pct', 'rsi_oversold', 'rsi_overbought']
        for param in float_params:
            if param in strategy_config and isinstance(strategy_config[param], str):
                try:
                    strategy_config[param] = float(strategy_config[param])
                except ValueError:
                    logger.warning(f"Could not convert {param} to float: {strategy_config[param]}")
        
        strategy = create_strategy(request.strategy_name, strategy_config)
        
        session["message"] = "Running backtest..."
        session["progress"] = 40.0
        
        # Configure backtesting engine with new interface
        position_size = strategy_config.get('position_size', 0.1)  # Default 10% of capital
        
        # Use new BacktestEngine constructor interface (no BacktestConfig)
        engine = BacktestEngine(
            initial_balance=request.initial_capital,
            fast_mode=False  # Full mode for detailed results
        )
        
        # Set additional engine parameters directly
        engine.commission_rate = request.commission_rate
        engine.slippage_rate = request.slippage_rate
        engine.position_sizing = 'percentage'
        engine.position_size = position_size
        
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