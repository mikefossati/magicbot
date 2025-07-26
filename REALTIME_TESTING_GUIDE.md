# Real-time Day Trading Strategy Testing Guide

This guide shows you how to test the day trading strategy in real-time using Binance testnet with the web dashboard.

## Prerequisites

1. **Binance Testnet Account**: 
   - Sign up at https://testnet.binance.vision/
   - Generate API keys (API Key and Secret Key)
   - Fund your testnet account with test USDT

2. **Environment Setup**:
   ```bash
   export BINANCE_API_KEY="your_testnet_api_key"
   export BINANCE_SECRET_KEY="your_testnet_secret_key"
   export DATABASE_URL="sqlite:///./trading.db"  # Optional
   export REDIS_URL="redis://localhost:6379"     # Optional
   ```

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Set your Binance testnet credentials
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_SECRET_KEY="your_testnet_secret_key"

# Run the automated setup script
./run_realtime_test.sh
```

### Option 2: Manual Setup
```bash
# 1. Start the web dashboard
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 2. In another terminal, run the real-time test
python scripts/test_day_trading_realtime.py --duration 60 --symbols BTCUSDT ETHUSDT
```

## What Happens During Testing

1. **Connection**: Connects to Binance testnet using your API credentials
2. **Data Fetching**: Continuously fetches 5-minute kline data for BTCUSDT and ETHUSDT
3. **Signal Generation**: Runs the day trading strategy every 30 seconds to generate signals
4. **Database Storage**: Saves all signals to the database
5. **Dashboard Updates**: Broadcasts real-time updates to the web dashboard via WebSocket
6. **Logging**: Provides detailed logging of all activities

## Strategy Features Being Tested

- **Multi-indicator Analysis**: EMA (8/21/50), RSI, MACD, Volume analysis
- **Support/Resistance**: Dynamic pivot levels
- **Risk Management**: ATR-based stop losses and take profits
- **Leverage Support**: Optional leverage with risk-adjusted position sizing
- **Session Management**: Trading time restrictions
- **Daily Trade Limits**: Maximum trades per day

## Accessing the Dashboard

1. Open your browser to: http://localhost:8000
2. You'll see:
   - **Portfolio Overview**: Active positions, trades, P&L, win rate
   - **Recent Signals**: Latest trading signals with confidence scores
   - **Risk Alerts**: Any risk management events
   - **Performance Chart**: Real-time P&L visualization
   - **Recent Trades Table**: Detailed trade history

## Configuration Options

### Test Duration
```bash
python scripts/test_day_trading_realtime.py --duration 120  # Run for 2 hours
```

### Different Symbols
```bash
python scripts/test_day_trading_realtime.py --symbols BTCUSDT ETHUSDT ADAUSDT DOTUSDT
```

### Strategy Parameters
Edit `config/development.yaml` under `day_trading_strategy` section:
```yaml
day_trading_strategy:
  symbols: ["BTCUSDT", "ETHUSDT"]
  fast_ema: 8
  medium_ema: 21
  slow_ema: 50
  rsi_period: 14
  stop_loss_pct: 1.5
  take_profit_pct: 2.5
  max_daily_trades: 3
  position_size: 0.02
  use_leverage: false
  leverage: 3.0
```

## Monitoring and Debugging

### Real-time Logs
The script provides detailed logging including:
- Connection status
- Market data fetching
- Signal generation details
- Risk management decisions
- Database operations

### Dashboard Features
- **Live Updates**: WebSocket connection provides real-time updates every 5 seconds
- **Connection Status**: Top-right indicator shows connection status
- **Performance Metrics**: Real-time P&L tracking and win rate calculations
- **Risk Monitoring**: Alerts for any risk management events

### Database Queries
You can query the database directly to analyze results:
```sql
-- View all signals
SELECT * FROM signals ORDER BY signal_time DESC LIMIT 10;

-- View strategy performance
SELECT * FROM strategy_performance WHERE strategy_name = 'day_trading_strategy';

-- View risk events
SELECT * FROM risk_events ORDER BY created_at DESC LIMIT 5;
```

## Safety Notes

⚠️ **This is testnet only** - No real money is involved
- Uses Binance testnet (https://testnet.binance.vision/)
- All trades are simulated with test funds
- Perfect for strategy validation and parameter tuning

## Troubleshooting

### Common Issues

1. **API Connection Failed**:
   - Verify your API credentials are correct
   - Ensure you're using testnet API keys (not mainnet)
   - Check your IP is whitelisted on Binance testnet

2. **Database Errors**:
   - Ensure SQLite database file has write permissions
   - Check DATABASE_URL environment variable

3. **Dashboard Not Loading**:
   - Verify port 8000 is not in use
   - Check firewall settings
   - Try accessing via http://127.0.0.1:8000

4. **No Signals Generated**:
   - Market conditions may not meet strategy criteria
   - Check if symbols have sufficient price movement
   - Review strategy parameters in config file

## Next Steps

After successful testing:
1. **Analyze Results**: Review signal quality and strategy performance
2. **Parameter Tuning**: Adjust strategy parameters based on results
3. **Risk Assessment**: Evaluate risk management effectiveness
4. **Production Planning**: Consider live trading implementation (with extreme caution)

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Review the configuration files
3. Ensure all prerequisites are met
4. Verify network connectivity to Binance testnet