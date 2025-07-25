from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
import structlog
from ..database.connection import db

logger = structlog.get_logger()

templates = Jinja2Templates(directory="src/web/templates")

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection established", 
                   total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed", 
                   total_connections=len(self.active_connections))
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)

websocket_manager = WebSocketManager()

async def get_dashboard_data() -> Dict:
    """Get dashboard data from database"""
    
    # Get recent trades
    trades_query = """
    SELECT symbol, side, entry_time, entry_price, exit_price, pnl, status
    FROM trades
    ORDER BY entry_time DESC
    LIMIT 20
    """
    recent_trades = await db.fetch_all(trades_query)
    
    # Get current positions
    positions_query = """
    SELECT symbol, quantity, avg_entry_price, current_price, unrealized_pnl
    FROM positions
    WHERE ABS(quantity) > 0
    """
    positions = await db.fetch_all(positions_query)
    
    # Get recent signals
    signals_query = """
    SELECT strategy_name, symbol, action, price, confidence, signal_time
    FROM signals
    ORDER BY signal_time DESC
    LIMIT 20
    """
    recent_signals = await db.fetch_all(signals_query)
    
    # Get performance metrics
    performance_query = """
    SELECT strategy_name, total_trades, winning_trades, total_pnl, win_rate
    FROM strategy_performance
    WHERE date >= $1
    ORDER BY date DESC
    """
    start_date = datetime.now().date() - timedelta(days=7)
    performance = await db.fetch_all(performance_query, start_date)
    
    # Get risk events
    risk_query = """
    SELECT event_type, severity, symbol, description, created_at
    FROM risk_events
    WHERE created_at >= $1
    ORDER BY created_at DESC
    LIMIT 10
    """
    start_time = datetime.now() - timedelta(hours=24)
    risk_events = await db.fetch_all(risk_query, start_time)
    
    return {
        'trades': recent_trades,
        'positions': positions,
        'signals': recent_signals,
        'performance': performance,
        'risk_events': risk_events,
        'last_updated': datetime.now()
    }

def setup_web_routes(app: FastAPI):
    """Setup web dashboard routes"""
    
    # Serve static files
    app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Main dashboard page"""
        data = await get_dashboard_data()
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "data": data
        })
    
    @app.get("/api/dashboard-data")
    async def api_dashboard_data():
        """API endpoint for dashboard data"""
        return await get_dashboard_data()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await websocket_manager.connect(websocket)
        try:
            while True:
                # Send periodic updates
                data = await get_dashboard_data()
                await websocket.send_text(json.dumps(data, default=str))
                await asyncio.sleep(5)  # Update every 5 seconds
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
    
    @app.get("/trades")
    async def trades_page(request: Request):
        """Trades history page"""
        trades_query = """
        SELECT * FROM trades
        ORDER BY entry_time DESC
        LIMIT 100
        """
        trades = await db.fetch_all(trades_query)
        
        return templates.TemplateResponse("trades.html", {
            "request": request,
            "trades": trades
        })
    
    @app.get("/performance")
    async def performance_page(request: Request):
        """Performance analytics page"""
        # Get performance data
        performance_query = """
        SELECT * FROM strategy_performance
        WHERE date >= $1
        ORDER BY date DESC
        """
        start_date = datetime.now().date() - timedelta(days=30)
        performance = await db.fetch_all(performance_query, start_date)
        
        return templates.TemplateResponse("performance.html", {
            "request": request,
            "performance": performance
        })
