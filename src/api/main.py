from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
import asyncio
from contextlib import asynccontextmanager

from src.core.config import settings, config
from src.exchanges.binance_exchange import BinanceExchange
from src.api.routes import trading, strategies

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global exchange instance
exchange = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    global exchange
    
    # Startup
    logger.info("Starting Magicbot Trading System", version=settings.version)
    
    # Initialize exchange connection
    exchange = BinanceExchange()
    await exchange.connect()
    
    # Store exchange in app state
    app.state.exchange = exchange
    
    yield
    
    # Shutdown
    logger.info("Shutting down Magicbot Trading System")
    if exchange:
        await exchange.disconnect()

app = FastAPI(
    title="Magicbot Trading System",
    description="Advanced Cryptocurrency Algorithmic Trading Platform",
    version=settings.version,
    debug=settings.debug,
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

# Include routers
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["strategies"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": "Magicbot Trading System",
        "version": settings.version,
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
            # Try to get account balance as a connectivity test
            await exchange.get_account_balance()
        
        return {
            "status": "healthy",
            "exchange": exchange_status,
            "database": "connected",  # TODO: Add database health check
            "redis": "connected"      # TODO: Add Redis health check
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)