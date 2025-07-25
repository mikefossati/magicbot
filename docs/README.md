# MagicBot Documentation

## Overview

MagicBot is a modular cryptocurrency trading bot framework designed for algorithmic trading strategies, backtesting, and risk management.

## Architecture

### Core Components

- **Core**: Configuration, database, events, and exception handling
- **Exchanges**: Exchange integrations (currently supports Binance)
- **Strategies**: Trading strategy implementations
- **Data**: Market data handling and storage
- **Risk**: Position sizing and risk management
- **Backtesting**: Strategy testing and performance analysis
- **API**: REST API for bot management

### Key Features

- Modular strategy system
- Built-in risk management
- Comprehensive backtesting engine
- Real-time market data processing
- RESTful API interface
- Docker containerization
- Extensive configuration options

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.template` to `.env` and configure
3. Initialize database: `python scripts/setup_db.py`
4. Run backtest: `python scripts/run_backtest.py --strategy ma_crossover --start-date 2023-01-01 --end-date 2023-12-31`
5. Start trading: `python scripts/start_trading.py --strategy ma_crossover --dry-run`

## Configuration

See `config/` directory for environment-specific configurations.

## Development

Use `docker-compose.dev.yml` for development environment with hot reloading.
