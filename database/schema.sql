-- Enable TimescaleDB extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Historical market data (optimized for TimescaleDB)
CREATE TABLE IF NOT EXISTS market_data (
    timestamp BIGINT NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    interval_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint that includes timestamp (required for hypertable)
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_unique 
ON market_data (timestamp, symbol, interval_type);

-- Create hypertable ONLY if table is not already a hypertable
DO $$
BEGIN
    -- Check if table is already a hypertable
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE table_name = 'market_data'
    ) THEN
        -- Convert to hypertable with 1-day chunks
        PERFORM create_hypertable(
            'market_data', 
            'timestamp',
            chunk_time_interval => 86400000000::bigint, -- 1 day in microseconds
            if_not_exists => TRUE
        );
    END IF;
END $$;

-- Additional indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
ON market_data (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_interval 
ON market_data (interval_type, timestamp DESC);

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_time TIMESTAMP NOT NULL,
    action VARCHAR(10) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy_time 
ON signals (strategy_name, signal_time DESC);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
ON signals (symbol, signal_time DESC);

-- Executed trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    pnl DECIMAL(20,8),
    pnl_percentage DECIMAL(10,4),
    commission DECIMAL(20,8) DEFAULT 0,
    slippage DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'OPEN',
    exchange_order_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_strategy_time 
ON trades (strategy_name, entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status 
ON trades (symbol, status);

CREATE INDEX IF NOT EXISTS idx_trades_entry_time 
ON trades (entry_time DESC);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    avg_entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Use a unique constraint instead of UNIQUE in table definition
CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_unique 
ON positions (symbol, strategy_name);

-- Risk events and violations
CREATE TABLE IF NOT EXISTS risk_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    strategy_name VARCHAR(100),
    description TEXT NOT NULL,
    current_value DECIMAL(20,8),
    threshold_value DECIMAL(20,8),
    action_taken VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_events_time 
ON risk_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_risk_events_severity 
ON risk_events (severity, created_at DESC);

-- Strategy performance metrics
CREATE TABLE IF NOT EXISTS strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(10,4) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4),
    win_rate DECIMAL(5,2),
    avg_trade_duration INTERVAL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Use a unique constraint instead of UNIQUE in table definition
CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_performance_unique 
ON strategy_performance (strategy_name, date);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    logger VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function VARCHAR(100),
    line_number INTEGER,
    exception TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_level_time 
ON system_logs (level, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_logs_logger_time 
ON system_logs (logger, created_at DESC);

-- Data retention policies (optional - uncomment if you want automatic cleanup)
-- Keep market data for 1 year
-- SELECT add_retention_policy('market_data', INTERVAL '1 year');

-- Keep system logs for 3 months  
-- SELECT add_retention_policy('system_logs', INTERVAL '3 months');

-- Create a view for easy querying of recent market data
CREATE OR REPLACE VIEW recent_market_data AS
SELECT 
    symbol,
    interval_type,
    timestamp,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    to_timestamp(timestamp / 1000) AS datetime
FROM market_data
WHERE timestamp >= EXTRACT(epoch FROM NOW() - INTERVAL '7 days') * 1000
ORDER BY timestamp DESC;

-- Create a view for trading performance summary
CREATE OR REPLACE VIEW trading_summary AS
SELECT 
    strategy_name,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
    COUNT(*) FILTER (WHERE pnl < 0) as losing_trades,
    ROUND(
        COUNT(*) FILTER (WHERE pnl > 0)::numeric / 
        NULLIF(COUNT(*), 0) * 100, 2
    ) as win_rate_pct,
    ROUND(SUM(pnl)::numeric, 2) as total_pnl,
    ROUND(AVG(pnl)::numeric, 2) as avg_pnl,
    ROUND(MAX(pnl)::numeric, 2) as best_trade,
    ROUND(MIN(pnl)::numeric, 2) as worst_trade
FROM trades
WHERE status = 'CLOSED'
GROUP BY strategy_name;

-- Create function to calculate portfolio value
CREATE OR REPLACE FUNCTION get_portfolio_value()
RETURNS TABLE(
    symbol VARCHAR,
    quantity DECIMAL,
    current_price DECIMAL,
    position_value DECIMAL,
    unrealized_pnl DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.symbol,
        p.quantity,
        p.current_price,
        (p.quantity * p.current_price) as position_value,
        p.unrealized_pnl
    FROM positions p
    WHERE ABS(p.quantity) > 0;
END;
$$ LANGUAGE plpgsql;