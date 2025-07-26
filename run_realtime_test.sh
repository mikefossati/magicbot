#!/bin/bash

# Real-time Day Trading Test Setup Script
# This script starts the web dashboard and real-time testing

echo "🤖 Magicbot Real-time Day Trading Test Setup"
echo "============================================="

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  No .env file found. Creating from template..."
    if [ -f "config/.env.template" ]; then
        cp config/.env.template .env
        echo "✅ Created .env file from template"
        echo "📝 Please edit .env file with your Binance testnet API credentials:"
        echo "   - BINANCE_API_KEY=your_testnet_api_key"
        echo "   - BINANCE_SECRET_KEY=your_testnet_secret_key"
        echo "   Then run this script again."
        exit 1
    else
        echo "❌ Error: No .env template found. Please create .env file manually."
        exit 1
    fi
fi

# Check if required API credentials are now available
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_SECRET_KEY" ]; then
    echo "❌ Error: BINANCE_API_KEY and BINANCE_SECRET_KEY not found in .env file"
    echo "📝 Please edit your .env file and add:"
    echo "   BINANCE_API_KEY=your_testnet_api_key"
    echo "   BINANCE_SECRET_KEY=your_testnet_secret_key"
    echo "🌐 Get testnet credentials from: https://testnet.binance.vision/"
    exit 1
fi

# Set default database URL if not provided
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="sqlite:///./trading.db"
    echo "📊 Using SQLite database: $DATABASE_URL"
fi

# Set default Redis URL if not provided
if [ -z "$REDIS_URL" ]; then
    export REDIS_URL="redis://localhost:6379"
    echo "📡 Using Redis: $REDIS_URL"
fi

echo "🌐 Binance Testnet: ENABLED"
echo "🔑 API Key: ${BINANCE_API_KEY:0:8}..."
echo "💾 Database: $DATABASE_URL"
echo "📡 Redis: $REDIS_URL"
echo ""

# Function to start the web dashboard
start_dashboard() {
    echo "🚀 Starting web dashboard..."
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    DASHBOARD_PID=$!
    echo "📊 Dashboard PID: $DASHBOARD_PID"
    
    # Wait for dashboard to start
    echo "⏳ Waiting for dashboard to start..."
    sleep 5
    
    # Check if dashboard is running
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Dashboard is running at http://localhost:8000"
    else
        echo "❌ Dashboard failed to start"
        exit 1
    fi
}

# Function to start the real-time test
start_realtime_test() {
    echo ""
    echo "🚀 Starting real-time day trading test..."
    python scripts/test_day_trading_realtime.py --duration 60 --symbols BTCUSDT ETHUSDT
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null
        echo "🔌 Dashboard stopped"
    fi
    echo "👋 Goodbye!"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start services
start_dashboard
start_realtime_test

# Keep the script running
wait