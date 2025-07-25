from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
import asyncio
from contextlib import asynccontextmanager

from ..core.config import settings, config
from ..exchanges.binance_exchange import BinanceExchange
from ..database.connection import db
from ..logging.logger_config import setup_logging
from ..web.dashboard import setup_web_routes
from .routes import trading, strategies

# Setup logging
setup_logging(log_level=getattr(settings, 'log_level', 'INFO'))
logger = structlog.get_logger()

# Global exchange instance
exchange = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    global exchange
    
    # Startup
    logger.info("Starting Magicbot Trading System", version=getattr(settings, 'version', '2.0.0'))
    
    # Initialize database
    try:
        await db.initialize()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        # Continue without database for development
        logger.warning("Continuing without database connection")
    
    # Initialize exchange connection
    exchange = BinanceExchange()
    try:
        await exchange.connect()
        logger.info("Exchange connected")
    except Exception as e:
        logger.error("Exchange connection failed", error=str(e))
        logger.warning("Continuing without exchange connection")
    
    # Store instances in app state
    app.state.exchange = exchange
    app.state.db = db
    
    yield
    
    # Shutdown
    logger.info("Shutting down Magicbot Trading System")
    if exchange:
        await exchange.disconnect()
    await db.close()

app = FastAPI(
    title="Magicbot Trading System",
    description="Advanced Cryptocurrency Algorithmic Trading Platform",
    version=getattr(settings, 'version', '2.0.0'),
    debug=getattr(settings, 'debug', True),
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup web dashboard routes
setup_web_routes(app)

# Include API routers
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["strategies"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": "Magicbot Trading System",
        "version": getattr(settings, 'version', '2.0.0'),
        "status": "online",
        "exchange_connected": exchange is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test exchange connection
        exchange_status = "connected" if exchange else "disconnected"
        if exchange:
            try:
                await exchange.get_account_balance()
            except Exception as e:
                exchange_status = f"error: {str(e)}"
        
        # Test database connection
        db_status = "connected"
        try:
            await db.fetch_one("SELECT 1")
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "exchange": exchange_status,
            "database": db_status,
            "version": getattr(settings, 'version', '2.0.0')
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)