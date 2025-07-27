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

-- Parameter Optimization Tables

-- Optimization runs - Main table for tracking optimization jobs
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id TEXT PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    optimizer_type VARCHAR(50) NOT NULL,  -- 'grid_search', 'genetic', 'multi_objective'
    objective_type VARCHAR(50) NOT NULL,  -- 'maximize_return', 'maximize_sharpe', etc.
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) NOT NULL,  -- 'running', 'completed', 'failed', 'cancelled'
    config JSONB NOT NULL,  -- Optimizer configuration parameters
    parameter_space JSONB NOT NULL,  -- Parameter space definition
    best_parameters JSONB,  -- Best parameters found
    best_objective_value DECIMAL(20,8),  -- Best objective value achieved
    total_evaluations INTEGER DEFAULT 0,
    metadata JSONB,  -- Additional metadata (validation settings, etc.)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Parameter evaluations - Individual parameter test results
CREATE TABLE IF NOT EXISTS parameter_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    parameters JSONB NOT NULL,  -- Parameter values tested
    objective_value DECIMAL(20,8) NOT NULL,  -- Objective function result
    metrics JSONB NOT NULL,  -- Additional metrics (Sharpe, drawdown, etc.)
    validation_results JSONB,  -- Walk-forward validation results
    is_valid BOOLEAN NOT NULL,  -- Whether evaluation was successful
    evaluation_time DECIMAL(10,4) NOT NULL,  -- Time taken to evaluate (seconds)
    timestamp TIMESTAMP NOT NULL,
    iteration INTEGER NOT NULL  -- Iteration number in optimization
);

-- Validation results - Walk-forward and cross-validation data
CREATE TABLE IF NOT EXISTS validation_results (
    validation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    validation_type VARCHAR(50) NOT NULL,  -- 'walk_forward', 'cross_validation', 'monte_carlo'
    parameters JSONB NOT NULL,  -- Parameters being validated
    results JSONB NOT NULL,  -- Validation results and metrics
    robustness_score DECIMAL(5,4),  -- Overall robustness score (0-1)
    overfitting_detected BOOLEAN,  -- Whether overfitting was detected
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model registry - Deployed model versions and metadata
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,  -- Optimized parameters
    performance_metrics JSONB NOT NULL,  -- Backtest performance metrics
    validation_score DECIMAL(5,4),  -- Cross-validation score
    is_deployed BOOLEAN DEFAULT FALSE,
    deployment_date TIMESTAMP,
    metadata JSONB,  -- Additional model metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for optimization tables
CREATE INDEX IF NOT EXISTS idx_runs_strategy ON optimization_runs (strategy_name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON optimization_runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_start_time ON optimization_runs (start_time DESC);
CREATE INDEX IF NOT EXISTS idx_runs_optimizer_type ON optimization_runs (optimizer_type);

CREATE INDEX IF NOT EXISTS idx_evals_run_id ON parameter_evaluations (run_id);
CREATE INDEX IF NOT EXISTS idx_evals_objective ON parameter_evaluations (objective_value DESC);
CREATE INDEX IF NOT EXISTS idx_evals_timestamp ON parameter_evaluations (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_evals_iteration ON parameter_evaluations (iteration);
CREATE INDEX IF NOT EXISTS idx_evals_valid ON parameter_evaluations (is_valid);

CREATE INDEX IF NOT EXISTS idx_validation_run_id ON validation_results (run_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_results (validation_type);
CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_results (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_models_strategy ON model_registry (strategy_name);
CREATE INDEX IF NOT EXISTS idx_models_deployed ON model_registry (is_deployed);
CREATE INDEX IF NOT EXISTS idx_models_created ON model_registry (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_validation_score ON model_registry (validation_score DESC);

-- GIN indexes for JSONB columns to enable fast parameter searches
CREATE INDEX IF NOT EXISTS idx_runs_config_gin ON optimization_runs USING GIN (config);
CREATE INDEX IF NOT EXISTS idx_runs_parameter_space_gin ON optimization_runs USING GIN (parameter_space);
CREATE INDEX IF NOT EXISTS idx_runs_best_params_gin ON optimization_runs USING GIN (best_parameters);

CREATE INDEX IF NOT EXISTS idx_evals_parameters_gin ON parameter_evaluations USING GIN (parameters);
CREATE INDEX IF NOT EXISTS idx_evals_metrics_gin ON parameter_evaluations USING GIN (metrics);

CREATE INDEX IF NOT EXISTS idx_models_parameters_gin ON model_registry USING GIN (parameters);
CREATE INDEX IF NOT EXISTS idx_models_performance_gin ON model_registry USING GIN (performance_metrics);

-- Views for optimization analysis

-- Recent optimization runs summary
CREATE OR REPLACE VIEW optimization_summary AS
SELECT 
    strategy_name,
    optimizer_type,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    COUNT(*) FILTER (WHERE status = 'running') as running_runs,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'completed')::numeric / 
        NULLIF(COUNT(*), 0) * 100, 2
    ) as success_rate_pct,
    MAX(best_objective_value) as best_objective_found,
    AVG(total_evaluations) as avg_evaluations,
    MAX(start_time) as last_run_time
FROM optimization_runs
GROUP BY strategy_name, optimizer_type
ORDER BY last_run_time DESC;

-- Parameter sensitivity analysis view
CREATE OR REPLACE VIEW parameter_importance AS
WITH parameter_stats AS (
    SELECT 
        run_id,
        jsonb_object_keys(parameters) as param_name,
        objective_value
    FROM parameter_evaluations 
    WHERE is_valid = true
)
SELECT 
    r.strategy_name,
    r.optimizer_type,
    ps.param_name,
    COUNT(*) as evaluation_count,
    ROUND(STDDEV(ps.objective_value)::numeric, 6) as objective_variance,
    ROUND(AVG(ps.objective_value)::numeric, 6) as avg_objective,
    ROUND(MAX(ps.objective_value)::numeric, 6) as max_objective,
    ROUND(MIN(ps.objective_value)::numeric, 6) as min_objective
FROM parameter_stats ps
JOIN optimization_runs r ON ps.run_id = r.run_id
WHERE r.status = 'completed'
GROUP BY r.strategy_name, r.optimizer_type, ps.param_name
HAVING COUNT(*) >= 10  -- Only include parameters with sufficient data
ORDER BY objective_variance DESC;

-- Best performing models by strategy
CREATE OR REPLACE VIEW best_models_by_strategy AS
WITH ranked_models AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY strategy_name 
            ORDER BY validation_score DESC, created_at DESC
        ) as rank
    FROM model_registry
    WHERE validation_score IS NOT NULL
)
SELECT 
    strategy_name,
    model_id,
    model_name,
    model_version,
    validation_score,
    is_deployed,
    deployment_date,
    created_at
FROM ranked_models
WHERE rank <= 3  -- Top 3 models per strategy
ORDER BY strategy_name, rank;

-- Function to get optimization convergence data
CREATE OR REPLACE FUNCTION get_optimization_convergence(run_id_param TEXT)
RETURNS TABLE(
    iteration INTEGER,
    objective_value DECIMAL,
    running_best DECIMAL,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH ordered_evals AS (
        SELECT 
            iteration,
            objective_value,
            timestamp,
            MAX(objective_value) OVER (
                ORDER BY iteration 
                ROWS UNBOUNDED PRECEDING
            ) as running_best
        FROM parameter_evaluations
        WHERE run_id = run_id_param AND is_valid = true
        ORDER BY iteration
    )
    SELECT 
        oe.iteration,
        oe.objective_value,
        oe.running_best,
        oe.timestamp
    FROM ordered_evals oe;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze parameter correlation with objective
CREATE OR REPLACE FUNCTION analyze_parameter_correlation(
    run_id_param TEXT,
    param_name TEXT
) 
RETURNS TABLE(
    correlation_coefficient DECIMAL,
    p_value DECIMAL,
    sample_size INTEGER
) AS $$
BEGIN
    -- Note: This is a simplified correlation analysis
    -- For full statistical analysis, consider using R or Python integration
    RETURN QUERY
    WITH param_values AS (
        SELECT 
            (parameters ->> param_name)::numeric as param_value,
            objective_value
        FROM parameter_evaluations
        WHERE run_id = run_id_param 
        AND is_valid = true
        AND parameters ? param_name
        AND (parameters ->> param_name) ~ '^-?[0-9]*\.?[0-9]+$'  -- Numeric values only
    )
    SELECT 
        ROUND(
            CORR(param_value, objective_value)::numeric, 6
        ) as correlation_coefficient,
        NULL::DECIMAL as p_value,  -- Placeholder for statistical significance
        COUNT(*)::INTEGER as sample_size
    FROM param_values
    WHERE param_value IS NOT NULL;
END;
$$ LANGUAGE plpgsql;