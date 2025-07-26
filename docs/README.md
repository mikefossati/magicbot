# MagicBot Professional Trading Platform 🚀

A professional-grade cryptocurrency trading platform with enterprise-level features including real-time monitoring, advanced risk management, persistent data storage, and comprehensive analytics.

## 🌟 Professional Features (Week 2)

### 🗄️ **TimescaleDB Integration**
- **Time-series optimized database** with automatic partitioning
- **Hypertables** for efficient market data storage and retrieval
- **Advanced indexing** for high-performance queries
- **Data retention policies** and automated cleanup

### ⚠️ **Advanced Risk Management**
- **Portfolio-level risk controls** with real-time monitoring
- **Position size limits** and exposure management
- **Drawdown tracking** with automatic position closure
- **Value-at-Risk (VaR)** calculations and correlation analysis
- **Risk violation logging** with automated alerts

### 🌐 **Real-time Web Dashboard**
- **Professional web interface** with live portfolio monitoring
- **WebSocket connections** for real-time data streaming
- **Interactive charts** with Chart.js integration
- **Portfolio overview** with P&L tracking
- **Trade history** and performance analytics
- **Risk alerts** and system status monitoring

### 📊 **Comprehensive Logging & Analytics**
- **Multi-tier logging system** (console, file, database)
- **Structured logging** with metadata and audit trails
- **Database-backed logs** for compliance and debugging
- **Trading-specific events** and performance tracking
- **System health monitoring** and error tracking

### 🔧 **Core Trading Features**
- **Modular strategy framework** with 4 built-in algorithms
  - Moving Average Crossover (trend following)
  - RSI Strategy (momentum reversal)
  - Bollinger Bands (mean reversion)
  - Breakout Strategy (range breakouts)
- **Advanced backtesting engine** with realistic simulation
- **Exchange integration** (Binance with more coming)
- **Real-time market data** processing and storage
- **Order management system** (coming in Week 4)
- **Portfolio tracking** with unrealized P&L

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (recommended for optimal compatibility)
- **PostgreSQL/TimescaleDB** for time-series data storage
- **Redis** (optional, for caching)
- **Virtual environment** (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd magicbot

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp config/.env.template .env
```

### Database Setup

**Option 1: Docker (Recommended)**
```bash
# Start TimescaleDB with Docker
docker run -d --name magicbot-db \
  -p 5432:5432 \
  -e POSTGRES_DB=magicbot \
  -e POSTGRES_USER=magicbot \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb:latest-pg15
```

**Option 2: Local Installation**
```bash
# Install TimescaleDB locally (macOS)
brew install timescaledb
# Follow TimescaleDB setup instructions
```

### Configuration

**1. Environment Variables (.env)**
```bash
# Database
DATABASE_URL=postgresql+asyncpg://magicbot:password@localhost:5432/magicbot

# Binance API (get from binance.com)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=true

# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
```

**2. Initialize Database Schema**
```bash
python scripts/setup_week2.py
```

**3. Test Installation**
```bash
python scripts/test_week2_setup.py
```

### Start the Platform

**1. Launch API Server & Dashboard**
```bash
python -m src.api.main
```

**2. Access Web Dashboard**
```
Open: http://localhost:8000
```

**3. Run Backtests**

**Single Strategy Testing:**
```bash
# Test MA crossover strategy (legacy script)
python scripts/run_backtest.py \
  --symbols BTCUSDT ETHUSDT \
  --fast-period 10 \
  --slow-period 30 \
  --interval 1h
```

**Multi-Strategy Comparison:**
```bash
# Compare all strategies side-by-side
python scripts/compare_strategies.py --all

# Compare specific strategies
python scripts/compare_strategies.py rsi_strategy bollinger_bands breakout_strategy

# Compare with custom parameters
python scripts/compare_strategies.py \
  ma_crossover rsi_strategy \
  --symbols BTCUSDT ETHUSDT ADAUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --capital 50000 \
  --interval 4h
```

**Advanced Multi-Strategy Portfolio:**
```bash
# Run portfolio backtest with multiple strategies
python scripts/run_multi_strategy_backtest.py \
  --strategies ma_crossover rsi_strategy bollinger_bands \
  --symbols BTCUSDT ETHUSDT \
  --initial-capital 100000 \
  --interval 1h

# Individual strategies only (skip portfolio)
python scripts/run_multi_strategy_backtest.py \
  --strategies rsi_strategy breakout_strategy \
  --no-portfolio
```

### Results Storage
- **JSON exports** for programmatic analysis
- **Text reports** for human-readable summaries
- **Database storage** for historical comparison
- **Chart exports** for presentations

### 📊 **Strategy Performance Metrics**

**Individual Strategy Report:**
```bash
=== Strategy Performance Report ===
Strategy: RSI Strategy
Period: 2024-01-01 to 2024-12-31
Symbol: BTCUSDT

📈 Returns:
- Total Return: +24.5%
- Annualized Return: +22.1%
- Max Drawdown: -8.3%
- Sharpe Ratio: 1.85

📊 Trade Statistics:
- Total Trades: 147
- Win Rate: 58.5%
- Avg Win: +2.1%
- Avg Loss: -1.4%
- Profit Factor: 1.67
```

**Multi-Strategy Comparison Report:**
```bash
📊 STRATEGY COMPARISON RESULTS
================================================================================
Strategy             Return %   Trades   Win %    Sharpe   Max DD %   Final $
--------------------------------------------------------------------------------
breakout_strategy        +28.7      89    64.0      2.12      -6.8    $12,870
rsi_strategy            +24.5     147    58.5      1.85      -8.3    $12,450
bollinger_bands         +18.3     203    55.2      1.67      -9.1    $11,830
ma_crossover            +15.2      67    52.2      1.43     -12.4    $11,520

🏆 BEST PERFORMERS
----------------------------------------
💰 Highest Return:   breakout_strategy (+28.7%)
📈 Best Sharpe:      breakout_strategy (2.12)
🎯 Best Win Rate:    breakout_strategy (64.0%)

💡 RECOMMENDATIONS
----------------------------------------
🎯 Consider combining trend-following and mean-reversion strategies
📊 Portfolio of top 2 strategies could yield ~26.6% return
```

**Multi-Strategy Portfolio Benefits:**
- **Diversification:** Reduced overall portfolio volatility
- **Risk-Adjusted Returns:** Higher Sharpe ratios through strategy mixing
- **Market Regime Coverage:** Different strategies excel in different conditions
- **Drawdown Reduction:** Portfolio max drawdown typically lower than individual strategies

### 🎯 **Quick Start Examples**

**Run the interactive example:**
```bash
python examples/multi_strategy_example.py
```

**Compare all strategies (recommended first step):**
```bash
python scripts/compare_strategies.py --all
```

**Test specific strategies:**
```bash
python scripts/compare_strategies.py rsi_strategy bollinger_bands breakout_strategy
```

## ⚠️ Risk Management System

### Portfolio-Level Controls
- **Position size limits** (% of portfolio)
- **Daily loss limits** with automatic shutdown
- **Maximum drawdown** protection
- **Concentration risk** monitoring

### Real-time Monitoring
- **VaR calculations** for portfolio risk
- **Correlation analysis** between positions
- **Exposure tracking** by symbol and strategy
- **Violation alerts** with immediate notifications

### Risk Metrics Dashboard
- **Current exposure** by symbol and strategy
- **Risk-adjusted returns** and Sharpe ratios
- **Drawdown analysis** with recovery tracking
- **Risk budget utilization** monitoring

## 📈 **Trading Strategies**

MagicBot includes 4 professional-grade trading strategies, each designed for different market conditions:

### 🔄 **Moving Average Crossover**
**Type:** Trend Following | **Best For:** Trending markets
```yaml
ma_crossover:
  fast_period: 10        # Short-term MA period
  slow_period: 30        # Long-term MA period
  position_size: 0.01    # 1% of portfolio per trade
```
- **Buy Signal:** Fast MA crosses above slow MA
- **Sell Signal:** Fast MA crosses below slow MA
- **Confidence:** Based on crossover strength and momentum

### ⚡ **RSI Strategy**
**Type:** Momentum Reversal | **Best For:** Volatile, mean-reverting markets
```yaml
rsi_strategy:
  rsi_period: 14         # RSI calculation period
  oversold: 30           # Oversold threshold
  overbought: 70         # Overbought threshold
  position_size: 0.01    # 1% of portfolio per trade
```
- **Buy Signal:** RSI ≤ 30 and showing upward momentum
- **Sell Signal:** RSI ≥ 70 and showing downward momentum
- **Confidence:** Increases with RSI extremes and momentum strength

### 📊 **Bollinger Bands Strategy**
**Type:** Mean Reversion | **Best For:** Range-bound markets
```yaml
bollinger_bands:
  period: 20             # Moving average period
  std_dev: 2.0           # Standard deviation multiplier
  mean_reversion_threshold: 0.02  # 2% reversion threshold
  position_size: 0.01    # 1% of portfolio per trade
```
- **Buy Signal:** Price near lower band (≤20%) with upward momentum
- **Sell Signal:** Price near upper band (≥80%) with downward momentum
- **Confidence:** Increases with band proximity and volatility

### 🚀 **Breakout Strategy**
**Type:** Range Breakout | **Best For:** Consolidation breakouts
```yaml
breakout_strategy:
  lookback_period: 20    # Period for support/resistance calculation
  breakout_threshold: 1.02  # 2% breakout threshold
  volume_confirmation: true # Require volume confirmation
  min_volatility: 0.005  # 0.5% minimum volatility filter
  position_size: 0.01    # 1% of portfolio per trade
```
- **Buy Signal:** Price breaks above resistance with volume
- **Sell Signal:** Price breaks below support with volume
- **Confidence:** Increases with breakout strength and volatility

### 🎯 **Strategy Selection Guide**

| Market Condition | Recommended Strategy | Reasoning |
|------------------|---------------------|-----------|
| **Strong Trend** | Moving Average Crossover | Captures sustained directional moves |
| **High Volatility** | RSI Strategy | Profits from momentum reversals |
| **Range-bound** | Bollinger Bands | Mean reversion in sideways markets |
| **Post-consolidation** | Breakout Strategy | Captures explosive moves after compression |
| **Mixed Conditions** | All Strategies | Diversified approach across market regimes |

### 🔧 **Strategy Configuration**

Enable multiple strategies simultaneously in `config/development.yaml`:

```yaml
strategies:
  enabled:
    - ma_crossover
    - rsi_strategy
    - bollinger_bands
    - breakout_strategy
  
  # Individual strategy configurations...
```

Each strategy includes:
- **Signal Generation:** Buy/Sell/Hold decisions
- **Confidence Scoring:** 0.0-1.0 confidence levels
- **Risk Integration:** Position sizing and risk limits
- **Metadata Tracking:** Detailed signal context for analysis

## 🗄️ Database Schema

### TimescaleDB Tables
- **market_data**: OHLCV data with hypertable optimization
- **trades**: Executed trades with P&L tracking
- **positions**: Current portfolio positions
- **signals**: Strategy signals and confidence levels
- **risk_events**: Risk violations and alerts
- **system_logs**: Comprehensive audit trails

### Performance Optimization
- **Hypertables** for time-series data
- **Indexes** for fast queries
- **Views** for common analytics
- **Functions** for portfolio calculations

## 📝 Logging & Monitoring

### Multi-Tier Logging
- **Console logs**: Real-time development feedback
- **File logs**: Persistent daily log files
- **Database logs**: Structured audit trails
- **Trading logs**: Specialized trading events

### Log Categories
- **System events**: Startup, shutdown, errors
- **Trading events**: Orders, fills, P&L updates
- **Risk events**: Violations, alerts, actions
- **Performance events**: Strategy metrics, benchmarks

## 🚀 Production Deployment

### Docker Compose (Recommended)
```bash
# Start full stack
docker-compose -f docker/docker-compose.yml up -d

# Scale services
docker-compose up --scale magicbot=3

# Monitor logs
docker-compose logs -f magicbot
```

### Manual Deployment
```bash
# Production environment
export APP_ENV=production
export DEBUG=false
export LOG_LEVEL=WARNING

# Start with gunicorn
gunicorn src.api.main:app --workers 4 --bind 0.0.0.0:8000
```

## 🧪 Testing

### Test Suite
```bash
# Run all tests
pytest tests/

# Test with coverage
pytest --cov=src --cov-report=html

# Test specific components
pytest tests/test_backtesting.py
pytest tests/test_risk_management.py
```

### Integration Tests
```bash
# Test database setup
python scripts/test_week2_setup.py

# Test exchange connectivity
python scripts/test_connection.py

# Test strategy signals
python scripts/test_strategy_signals.py
```

## 📚 Development Guide

### Adding New Strategies

**1. Create Strategy Class**
```python
# src/strategies/my_strategy.py
from .base import BaseStrategy, Signal
from decimal import Decimal
import structlog

logger = structlog.get_logger()

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        self.my_param = config.get('my_param', 10)
        super().__init__(config)
        self.last_signals = {}
    
    def validate_parameters(self):
        """Validate strategy parameters"""
        if self.my_param <= 0:
            logger.error("my_param must be positive")
            return False
        return True
    
    def get_required_data(self):
        """Define data requirements"""
        return {
            'timeframes': ['1h'],
            'lookback_periods': self.my_param + 5,
            'indicators': ['custom_indicator']
        }
    
    async def generate_signals(self, market_data):
        """Generate trading signals"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
                
            signal = await self._analyze_symbol(symbol, market_data[symbol])
            if signal:
                signals.append(signal)
        
        return signals
    
    async def _analyze_symbol(self, symbol, data):
        """Analyze individual symbol"""
        # Your custom analysis logic
        if len(data) < self.my_param:
            return None
        
        # Example signal generation
        current_price = Decimal(str(data[-1]['close']))
        
        return Signal(
            symbol=symbol,
            action='BUY',  # or 'SELL' or 'HOLD'
            quantity=self.position_size,
            confidence=0.75,
            price=current_price,
            metadata={'my_indicator': 123.45}
        )
```

**2. Register Strategy**
```python
# src/strategies/registry.py - Add to imports and registry
from .my_strategy import MyStrategy

STRATEGY_REGISTRY = {
    # ... existing strategies ...
    'my_strategy': MyStrategy,
}
```

**3. Add Configuration**
```yaml
# config/development.yaml
strategies:
  enabled:
    - my_strategy
  
  my_strategy:
    symbols: ["BTCUSDT", "ETHUSDT"]
    my_param: 15
    position_size: 0.01
```

### Custom Risk Rules
```python
from src.risk.portfolio_risk_manager import PortfolioRiskManager

class MyRiskManager(PortfolioRiskManager):
    def check_custom_rule(self, trade_request):
        # Your custom risk logic
        return True  # or False to reject
```

## 🛣️ Roadmap

### ✅ Week 2 Complete
- TimescaleDB integration with hypertables
- Advanced portfolio risk management system
- Real-time web dashboard with WebSocket streaming
- Comprehensive multi-tier logging system
- **4 Professional Trading Strategies:**
  - Moving Average Crossover (trend following)
  - RSI Strategy (momentum reversal)  
  - Bollinger Bands (mean reversion)
  - Breakout Strategy (range breakouts)

### 📋 Week 3 Planned
- Advanced market data pipelines with streaming
- Real-time WebSocket market feeds integration
- Enhanced portfolio analytics and performance tracking
- Strategy performance optimization and parameter tuning
- Machine learning strategy components
- Advanced technical indicators library

### 📋 Week 4 Planned
- Live trading integration with order management
- Advanced order types (stop-loss, take-profit, trailing stops)
- Production monitoring and alerting systems
- Deployment automation with Docker/Kubernetes
- Strategy marketplace and community features

## ⚖️ License & Disclaimer

**License**: MIT License - see LICENSE file for details.

**⚠️ Important Disclaimer**: 
This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always test strategies thoroughly with paper trading before risking real capital.

## 📞 Support & Community

- 📖 **Documentation**: Comprehensive guides in `docs/` directory
- 🐛 **Bug Reports**: GitHub Issues for bug reports and feature requests
- 💬 **Discussions**: GitHub Discussions for community support
- 📧 **Contact**: [Your contact information]

---

**Built with ❤️ for the algorithmic trading community**
