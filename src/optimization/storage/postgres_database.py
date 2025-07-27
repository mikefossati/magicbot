"""
Optimization Results Database - PostgreSQL Implementation

PostgreSQL-based storage for optimization results using the existing database infrastructure
with efficient querying and analysis capabilities.
"""

import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import structlog
from pathlib import Path

from ...database.connection import db

logger = structlog.get_logger()

@dataclass
class OptimizationRun:
    """Represents a complete optimization run"""
    run_id: str
    strategy_name: str
    optimizer_type: str
    objective_type: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    config: Dict[str, Any]
    parameter_space: Dict[str, Any]
    best_parameters: Optional[Dict[str, Any]]
    best_objective_value: Optional[float]
    total_evaluations: int
    metadata: Dict[str, Any]

@dataclass
class ParameterEvaluation:
    """Represents a single parameter evaluation"""
    evaluation_id: str
    run_id: str
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float]
    validation_results: Optional[Dict[str, Any]]
    is_valid: bool
    evaluation_time: float
    timestamp: datetime
    iteration: int

class OptimizationDatabase:
    """
    PostgreSQL database for storing optimization results using existing infrastructure.
    
    Features:
    - Persistent storage of optimization runs
    - Efficient parameter evaluation storage
    - Metadata and configuration tracking
    - Query interface for analysis
    - Data export capabilities
    """
    
    def __init__(self):
        """Initialize optimization database using existing connection"""
        self.db = db
    
    async def initialize_tables(self):
        """Create optimization tables if they don't exist"""
        
        # Optimization runs table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                optimizer_type TEXT NOT NULL,
                objective_type TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                config JSONB NOT NULL,
                parameter_space JSONB NOT NULL,
                best_parameters JSONB,
                best_objective_value REAL,
                total_evaluations INTEGER DEFAULT 0,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Parameter evaluations table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS parameter_evaluations (
                evaluation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                parameters JSONB NOT NULL,
                objective_value REAL NOT NULL,
                metrics JSONB NOT NULL,
                validation_results JSONB,
                is_valid BOOLEAN NOT NULL,
                evaluation_time REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                iteration INTEGER NOT NULL
            )
        """)
        
        # Validation results table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                validation_type TEXT NOT NULL,
                parameters JSONB NOT NULL,
                results JSONB NOT NULL,
                robustness_score REAL,
                overfitting_detected BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model registry table
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                parameters JSONB NOT NULL,
                performance_metrics JSONB NOT NULL,
                validation_score REAL,
                is_deployed BOOLEAN DEFAULT FALSE,
                deployment_date TIMESTAMP,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better query performance
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON optimization_runs (strategy_name)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON optimization_runs (status)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON optimization_runs (start_time)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_evals_run_id ON parameter_evaluations (run_id)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_evals_objective ON parameter_evaluations (objective_value)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_evals_timestamp ON parameter_evaluations (timestamp)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_models_strategy ON model_registry (strategy_name)")
        await self.db.execute("CREATE INDEX IF NOT EXISTS idx_models_deployed ON model_registry (is_deployed)")
        
        logger.info("Optimization database tables initialized")
    
    async def create_optimization_run(self, run: OptimizationRun) -> str:
        """Create a new optimization run record"""
        
        await self.db.execute("""
            INSERT INTO optimization_runs (
                run_id, strategy_name, optimizer_type, objective_type,
                start_time, end_time, status, config, parameter_space,
                best_parameters, best_objective_value, total_evaluations, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """, 
            run.run_id,
            run.strategy_name,
            run.optimizer_type,
            run.objective_type,
            run.start_time,
            run.end_time,
            run.status,
            json.dumps(run.config),
            json.dumps(run.parameter_space),
            json.dumps(run.best_parameters) if run.best_parameters else None,
            run.best_objective_value,
            run.total_evaluations,
            json.dumps(run.metadata)
        )
        
        logger.info("Created optimization run", run_id=run.run_id)
        return run.run_id
    
    async def update_optimization_run(self, run_id: str, updates: Dict[str, Any]):
        """Update an existing optimization run"""
        
        if not updates:
            return
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        param_count = 1
        
        for key, value in updates.items():
            if key in ['config', 'parameter_space', 'best_parameters', 'metadata']:
                # JSON fields
                set_clauses.append(f"{key} = ${param_count}")
                values.append(json.dumps(value) if value is not None else None)
            else:
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
            param_count += 1
        
        values.append(run_id)  # For WHERE clause
        
        query = f"UPDATE optimization_runs SET {', '.join(set_clauses)} WHERE run_id = ${param_count}"
        
        await self.db.execute(query, *values)
        
        logger.debug("Updated optimization run", run_id=run_id, updates=list(updates.keys()))
    
    async def add_parameter_evaluation(self, evaluation: ParameterEvaluation):
        """Add a parameter evaluation record"""
        
        await self.db.execute("""
            INSERT INTO parameter_evaluations (
                evaluation_id, run_id, parameters, objective_value, metrics,
                validation_results, is_valid, evaluation_time, timestamp, iteration
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
            evaluation.evaluation_id,
            evaluation.run_id,
            json.dumps(evaluation.parameters),
            evaluation.objective_value,
            json.dumps(evaluation.metrics),
            json.dumps(evaluation.validation_results) if evaluation.validation_results else None,
            evaluation.is_valid,
            evaluation.evaluation_time,
            evaluation.timestamp,
            evaluation.iteration
        )
    
    async def get_optimization_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get optimization run by ID"""
        
        row = await self.db.fetch_one("""
            SELECT run_id, strategy_name, optimizer_type, objective_type,
                   start_time, end_time, status, config, parameter_space,
                   best_parameters, best_objective_value, total_evaluations, metadata
            FROM optimization_runs WHERE run_id = $1
        """, run_id)
        
        if row:
            return OptimizationRun(
                run_id=row['run_id'],
                strategy_name=row['strategy_name'],
                optimizer_type=row['optimizer_type'],
                objective_type=row['objective_type'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                status=row['status'],
                config=row['config'] if isinstance(row['config'], dict) else json.loads(row['config']),
                parameter_space=row['parameter_space'] if isinstance(row['parameter_space'], dict) else json.loads(row['parameter_space']),
                best_parameters=row['best_parameters'] if row['best_parameters'] and isinstance(row['best_parameters'], dict) else (json.loads(row['best_parameters']) if row['best_parameters'] else None),
                best_objective_value=row['best_objective_value'],
                total_evaluations=row['total_evaluations'],
                metadata=row['metadata'] if isinstance(row['metadata'], dict) else (json.loads(row['metadata']) if row['metadata'] else {})
            )
        
        return None
    
    async def get_parameter_evaluations(
        self,
        run_id: str,
        limit: Optional[int] = None,
        valid_only: bool = False
    ) -> List[ParameterEvaluation]:
        """Get parameter evaluations for a run"""
        
        query = """
            SELECT evaluation_id, run_id, parameters, objective_value, metrics,
                   validation_results, is_valid, evaluation_time, timestamp, iteration
            FROM parameter_evaluations
            WHERE run_id = $1
        """
        
        params = [run_id]
        param_count = 2
        
        if valid_only:
            query += f" AND is_valid = TRUE"
        
        query += " ORDER BY objective_value DESC"
        
        if limit:
            query += f" LIMIT ${param_count}"
            params.append(limit)
        
        rows = await self.db.fetch_all(query, *params)
        
        evaluations = []
        for row in rows:
            evaluations.append(ParameterEvaluation(
                evaluation_id=row['evaluation_id'],
                run_id=row['run_id'],
                parameters=row['parameters'] if isinstance(row['parameters'], dict) else json.loads(row['parameters']),
                objective_value=row['objective_value'],
                metrics=row['metrics'] if isinstance(row['metrics'], dict) else json.loads(row['metrics']),
                validation_results=row['validation_results'] if row['validation_results'] and isinstance(row['validation_results'], dict) else (json.loads(row['validation_results']) if row['validation_results'] else None),
                is_valid=row['is_valid'],
                evaluation_time=row['evaluation_time'],
                timestamp=row['timestamp'],
                iteration=row['iteration']
            ))
        
        return evaluations
    
    async def get_optimization_runs(
        self,
        strategy_name: Optional[str] = None,
        status: Optional[str] = None,
        days_back: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[OptimizationRun]:
        """Get optimization runs with filters"""
        
        query = """
            SELECT run_id, strategy_name, optimizer_type, objective_type,
                   start_time, end_time, status, config, parameter_space,
                   best_parameters, best_objective_value, total_evaluations, metadata
            FROM optimization_runs
            WHERE 1=1
        """
        
        params = []
        param_count = 1
        
        if strategy_name:
            query += f" AND strategy_name = ${param_count}"
            params.append(strategy_name)
            param_count += 1
        
        if status:
            query += f" AND status = ${param_count}"
            params.append(status)
            param_count += 1
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query += f" AND start_time >= ${param_count}"
            params.append(cutoff_date)
            param_count += 1
        
        query += " ORDER BY start_time DESC"
        
        if limit:
            query += f" LIMIT ${param_count}"
            params.append(limit)
        
        rows = await self.db.fetch_all(query, *params)
        
        runs = []
        for row in rows:
            runs.append(OptimizationRun(
                run_id=row['run_id'],
                strategy_name=row['strategy_name'],
                optimizer_type=row['optimizer_type'],
                objective_type=row['objective_type'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                status=row['status'],
                config=row['config'] if isinstance(row['config'], dict) else json.loads(row['config']),
                parameter_space=row['parameter_space'] if isinstance(row['parameter_space'], dict) else json.loads(row['parameter_space']),
                best_parameters=row['best_parameters'] if row['best_parameters'] and isinstance(row['best_parameters'], dict) else (json.loads(row['best_parameters']) if row['best_parameters'] else None),
                best_objective_value=row['best_objective_value'],
                total_evaluations=row['total_evaluations'],
                metadata=row['metadata'] if isinstance(row['metadata'], dict) else (json.loads(row['metadata']) if row['metadata'] else {})
            ))
        
        return runs
    
    async def get_best_parameters_by_strategy(self, strategy_name: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get best parameter sets for a strategy"""
        
        rows = await self.db.fetch_all("""
            SELECT DISTINCT r.run_id, r.best_parameters, r.best_objective_value,
                   r.optimizer_type, r.start_time
            FROM optimization_runs r
            WHERE r.strategy_name = $1 AND r.best_parameters IS NOT NULL
            ORDER BY r.best_objective_value DESC
            LIMIT $2
        """, strategy_name, top_n)
        
        results = []
        for row in rows:
            results.append({
                'run_id': row['run_id'],
                'parameters': row['best_parameters'] if isinstance(row['best_parameters'], dict) else json.loads(row['best_parameters']),
                'objective_value': row['best_objective_value'],
                'optimizer_type': row['optimizer_type'],
                'optimization_date': row['start_time']
            })
        
        return results
    
    async def export_to_dataframe(self, run_id: str) -> pd.DataFrame:
        """Export parameter evaluations to pandas DataFrame"""
        
        rows = await self.db.fetch_all("""
            SELECT parameters, objective_value, metrics, is_valid, 
                   evaluation_time, timestamp, iteration
            FROM parameter_evaluations
            WHERE run_id = $1
            ORDER BY iteration
        """, run_id)
        
        if not rows:
            return pd.DataFrame()
        
        # Convert rows to list of dicts
        data = []
        for row in rows:
            data.append({
                'parameters': row['parameters'] if isinstance(row['parameters'], dict) else json.loads(row['parameters']),
                'objective_value': row['objective_value'],
                'metrics': row['metrics'] if isinstance(row['metrics'], dict) else json.loads(row['metrics']),
                'is_valid': row['is_valid'],
                'evaluation_time': row['evaluation_time'],
                'timestamp': row['timestamp'],
                'iteration': row['iteration']
            })
        
        df = pd.DataFrame(data)
        
        # Expand parameters into separate columns
        params_df = pd.json_normalize(df['parameters'])
        params_df.columns = [f"param_{col}" for col in params_df.columns]
        
        # Expand metrics into separate columns
        metrics_df = pd.json_normalize(df['metrics'])
        metrics_df.columns = [f"metric_{col}" for col in metrics_df.columns]
        
        # Combine all data
        result_df = pd.concat([
            df[['objective_value', 'is_valid', 'evaluation_time', 'timestamp', 'iteration']],
            params_df,
            metrics_df
        ], axis=1)
        
        return result_df
    
    async def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        # Count runs by status
        runs_by_status_rows = await self.db.fetch_all("""
            SELECT status, COUNT(*) as count FROM optimization_runs GROUP BY status
        """)
        runs_by_status = {row['status']: row['count'] for row in runs_by_status_rows}
        
        # Count runs by strategy
        runs_by_strategy_rows = await self.db.fetch_all("""
            SELECT strategy_name, COUNT(*) as count FROM optimization_runs GROUP BY strategy_name
        """)
        runs_by_strategy = {row['strategy_name']: row['count'] for row in runs_by_strategy_rows}
        
        # Total evaluations
        total_evaluations_row = await self.db.fetch_one("SELECT COUNT(*) as count FROM parameter_evaluations")
        total_evaluations = total_evaluations_row['count'] if total_evaluations_row else 0
        
        # Valid evaluations
        valid_evaluations_row = await self.db.fetch_one("SELECT COUNT(*) as count FROM parameter_evaluations WHERE is_valid = TRUE")
        valid_evaluations = valid_evaluations_row['count'] if valid_evaluations_row else 0
        
        # Date range
        date_range_row = await self.db.fetch_one("SELECT MIN(start_time) as earliest, MAX(start_time) as latest FROM optimization_runs")
        
        return {
            'total_runs': sum(runs_by_status.values()),
            'runs_by_status': runs_by_status,
            'runs_by_strategy': runs_by_strategy,
            'total_evaluations': total_evaluations,
            'valid_evaluations': valid_evaluations,
            'success_rate': valid_evaluations / total_evaluations if total_evaluations > 0 else 0,
            'date_range': {
                'earliest': date_range_row['earliest'] if date_range_row else None,
                'latest': date_range_row['latest'] if date_range_row else None
            }
        }
    
    async def cleanup_old_runs(self, days_to_keep: int = 90):
        """Clean up old optimization runs"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Get runs to delete
        old_runs = await self.db.fetch_all("""
            SELECT run_id FROM optimization_runs 
            WHERE start_time < $1 AND status != 'running'
        """, cutoff_date)
        
        old_run_ids = [row['run_id'] for row in old_runs]
        
        if old_run_ids:
            # Delete in proper order due to foreign key constraints
            # PostgreSQL will handle cascading deletes automatically due to ON DELETE CASCADE
            for run_id in old_run_ids:
                await self.db.execute("DELETE FROM optimization_runs WHERE run_id = $1", run_id)
            
            logger.info("Cleaned up old optimization runs",
                       deleted_runs=len(old_run_ids),
                       cutoff_date=cutoff_date)
        
        return len(old_run_ids)