#!/usr/bin/env python3
"""
Create optimization tables in the database with targeted SQL.

This script creates only the optimization tables with clean SQL.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database.connection import db
import structlog

logger = structlog.get_logger()

async def create_optimization_tables():
    """Create optimization tables in the database"""
    
    logger.info("Creating optimization tables...")
    
    try:
        # Initialize database connection
        await db.initialize()
        logger.info("Connected to database")
        
        # Create optimization_runs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id TEXT PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                optimizer_type VARCHAR(50) NOT NULL,
                objective_type VARCHAR(50) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status VARCHAR(20) NOT NULL,
                config JSONB NOT NULL,
                parameter_space JSONB NOT NULL,
                best_parameters JSONB,
                best_objective_value DECIMAL(20,8),
                total_evaluations INTEGER DEFAULT 0,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created optimization_runs table")
        
        # Create parameter_evaluations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS parameter_evaluations (
                evaluation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                parameters JSONB NOT NULL,
                objective_value DECIMAL(20,8) NOT NULL,
                metrics JSONB NOT NULL,
                validation_results JSONB,
                is_valid BOOLEAN NOT NULL,
                evaluation_time DECIMAL(10,4) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                iteration INTEGER NOT NULL
            )
        """)
        logger.info("Created parameter_evaluations table")
        
        # Create validation_results table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                validation_type VARCHAR(50) NOT NULL,
                parameters JSONB NOT NULL,
                results JSONB NOT NULL,
                robustness_score DECIMAL(5,4),
                overfitting_detected BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created validation_results table")
        
        # Create model_registry table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                model_name VARCHAR(100) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                strategy_name VARCHAR(100) NOT NULL,
                parameters JSONB NOT NULL,
                performance_metrics JSONB NOT NULL,
                validation_score DECIMAL(5,4),
                is_deployed BOOLEAN DEFAULT FALSE,
                deployment_date TIMESTAMP,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("Created model_registry table")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_runs_strategy ON optimization_runs (strategy_name)",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON optimization_runs (status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_start_time ON optimization_runs (start_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_evals_run_id ON parameter_evaluations (run_id)",
            "CREATE INDEX IF NOT EXISTS idx_evals_objective ON parameter_evaluations (objective_value DESC)",
            "CREATE INDEX IF NOT EXISTS idx_models_strategy ON model_registry (strategy_name)",
            "CREATE INDEX IF NOT EXISTS idx_models_deployed ON model_registry (is_deployed)"
        ]
        
        for index_sql in indexes:
            await db.execute(index_sql)
        
        logger.info("Created optimization indexes")
        
        # Verify tables were created
        tables_to_check = [
            'optimization_runs',
            'parameter_evaluations', 
            'validation_results',
            'model_registry'
        ]
        
        for table in tables_to_check:
            result = await db.fetch_one(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                table
            )
            
            if result and result['exists']:
                logger.info("Table verified", table=table)
            else:
                logger.error("Table not found", table=table)
                return False
        
        logger.info("All optimization tables created successfully!")
        return True
        
    except Exception as e:
        logger.error("Failed to create optimization tables", error=str(e))
        return False
    
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(create_optimization_tables())
