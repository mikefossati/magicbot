# MagicBot 🤖

A modular cryptocurrency trading bot framework designed for algorithmic trading strategies, backtesting, and risk management.

## Features

- 🔧 **Modular Architecture**: Pluggable strategies, exchanges, and risk management
- 📊 **Backtesting Engine**: Comprehensive strategy testing with performance metrics
- 🛡️ **Risk Management**: Built-in position sizing and risk controls
- 🏪 **Exchange Support**: Currently supports Binance (more coming soon)
- 🔌 **REST API**: Full API for bot management and monitoring
- 🐳 **Docker Ready**: Containerized deployment with Docker Compose
- 📈 **Strategy Library**: Pre-built strategies like MA Crossover
- 🔍 **Real-time Data**: Live market data processing

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd magicbot

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp config/.env.template .env
# Edit .env with your configuration
```

### Configuration

1. Copy `config/.env.template` to `.env`
2. Add your exchange API keys
3. Configure database settings
4. Set risk management parameters

### Initialize Database

```bash
python scripts/setup_db.py
```

### Run a Backtest

```bash
python scripts/run_backtest.py \
  --strategy ma_crossover \
  --symbol BTCUSDT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --initial-balance 10000
```

### Start Trading (Dry Run)

```bash
python scripts/start_trading.py \
  --strategy ma_crossover \
  --symbol BTCUSDT \
  --dry-run
```

### Start API Server

```bash
python -m src.api.main
```

## Docker Deployment

### Development

```bash
docker-compose -f docker/docker-compose.dev.yml up
```

### Production

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Project Structure

```
magicbot/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── exchanges/         # Exchange integrations
│   ├── strategies/        # Trading strategies
│   ├── data/             # Data handling
│   ├── risk/             # Risk management
│   ├── backtesting/      # Backtesting engine
│   └── api/              # REST API
├── tests/                 # Test suite
├── config/               # Configuration files
├── scripts/              # Utility scripts
├── docker/               # Docker configuration
└── docs/                 # Documentation
```

## Strategies

### Built-in Strategies

- **MA Crossover**: Moving average crossover strategy
- More strategies coming soon...

### Creating Custom Strategies

Extend the `BaseStrategy` class in `src/strategies/base.py`:

```python
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def on_bar(self, bar):
        # Your strategy logic here
        pass
```

## Risk Management

- Position sizing based on account balance
- Maximum daily loss limits
- Stop-loss and take-profit orders
- Portfolio-level risk controls

## API Endpoints

- `GET /health` - Health check
- `GET /strategies` - List available strategies
- `POST /trading/start` - Start trading
- `POST /trading/stop` - Stop trading
- `GET /trading/status` - Get trading status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.

## Support

For questions and support, please open an issue on GitHub.
