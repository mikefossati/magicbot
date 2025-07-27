"""
Optimization Results Database

SQLite-based storage for optimization results with efficient querying
and analysis capabilities.
"""

import sqlite3
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import structlog
from pathlib import Path
from contextlib import contextmanager

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
    SQLite database for storing optimization results.
    
    Features:
    - Persistent storage of optimization runs
    - Efficient parameter evaluation storage
    - Metadata and configuration tracking
    - Query interface for analysis
    - Data export capabilities
    """
    
    def __init__(self, db_path: str = "optimization_results.db"):
        """
        Initialize optimization database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database_exists()
        self._create_tables()
    
    def _ensure_database_exists(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Optimization runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    run_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    optimizer_type TEXT NOT NULL,
                    objective_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    config TEXT NOT NULL,
                    parameter_space TEXT NOT NULL,
                    best_parameters TEXT,
                    best_objective_value REAL,
                    total_evaluations INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Parameter evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    objective_value REAL NOT NULL,
                    metrics TEXT NOT NULL,
                    validation_results TEXT,
                    is_valid BOOLEAN NOT NULL,
                    evaluation_time REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    iteration INTEGER NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id)
                )
            """)
            
            # Validation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    validation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    results TEXT NOT NULL,
                    robustness_score REAL,
                    overfitting_detected BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id)
                )
            """)
            
            # Model registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    validation_score REAL,
                    is_deployed BOOLEAN DEFAULT FALSE,
                    deployment_date TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON optimization_runs (strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON optimization_runs (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON optimization_runs (start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_run_id ON parameter_evaluations (run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_objective ON parameter_evaluations (objective_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_timestamp ON parameter_evaluations (timestamp)")
            
            conn.commit()
    
    def create_optimization_run(self, run: OptimizationRun) -> str:
        """Create a new optimization run record"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO optimization_runs (
                    run_id, strategy_name, optimizer_type, objective_type,
                    start_time, end_time, status, config, parameter_space,
                    best_parameters, best_objective_value, total_evaluations, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
            ))
            
            conn.commit()
        
        logger.info("Created optimization run", run_id=run.run_id)
        return run.run_id
    
    def update_optimization_run(self, run_id: str, updates: Dict[str, Any]):
        """Update an existing optimization run"""
        
        if not updates:
            return
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ['config', 'parameter_space', 'best_parameters', 'metadata']:
                # JSON fields
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value) if value is not None else None)
            else:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        values.append(run_id)  # For WHERE clause
        
        query = f"UPDATE optimization_runs SET {', '.join(set_clauses)} WHERE run_id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
        
        logger.debug("Updated optimization run", run_id=run_id, updates=list(updates.keys()))
    
    def add_parameter_evaluation(self, evaluation: ParameterEvaluation):
        """Add a parameter evaluation record"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO parameter_evaluations (
                    evaluation_id, run_id, parameters, objective_value, metrics,
                    validation_results, is_valid, evaluation_time, timestamp, iteration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
            ))
            
            conn.commit()
    
    def get_optimization_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Get optimization run by ID"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT run_id, strategy_name, optimizer_type, objective_type,
                       start_time, end_time, status, config, parameter_space,
                       best_parameters, best_objective_value, total_evaluations, metadata
                FROM optimization_runs WHERE run_id = ?
            """, (run_id,))
            
            row = cursor.fetchone()
            
            if row:
                return OptimizationRun(
                    run_id=row[0],
                    strategy_name=row[1],
                    optimizer_type=row[2],
                    objective_type=row[3],
                    start_time=datetime.fromisoformat(row[4]),
                    end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                    status=row[6],
                    config=json.loads(row[7]),
                    parameter_space=json.loads(row[8]),
                    best_parameters=json.loads(row[9]) if row[9] else None,
                    best_objective_value=row[10],
                    total_evaluations=row[11],
                    metadata=json.loads(row[12]) if row[12] else {}
                )
        
        return None
    
    def get_parameter_evaluations(
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
            WHERE run_id = ?
        """
        
        params = [run_id]
        
        if valid_only:
            query += " AND is_valid = TRUE"
        
        query += " ORDER BY objective_value DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            evaluations = []
            for row in cursor.fetchall():
                evaluations.append(ParameterEvaluation(
                    evaluation_id=row[0],
                    run_id=row[1],
                    parameters=json.loads(row[2]),
                    objective_value=row[3],
                    metrics=json.loads(row[4]),
                    validation_results=json.loads(row[5]) if row[5] else None,
                    is_valid=bool(row[6]),
                    evaluation_time=row[7],
                    timestamp=datetime.fromisoformat(row[8]),
                    iteration=row[9]
                ))
            
            return evaluations
    
    def get_optimization_runs(
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
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query += " AND start_time >= ?"
            params.append(cutoff_date)
        
        query += " ORDER BY start_time DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            runs = []
            for row in cursor.fetchall():
                runs.append(OptimizationRun(
                    run_id=row[0],
                    strategy_name=row[1],
                    optimizer_type=row[2],
                    objective_type=row[3],
                    start_time=datetime.fromisoformat(row[4]),
                    end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                    status=row[6],
                    config=json.loads(row[7]),
                    parameter_space=json.loads(row[8]),
                    best_parameters=json.loads(row[9]) if row[9] else None,
                    best_objective_value=row[10],
                    total_evaluations=row[11],
                    metadata=json.loads(row[12]) if row[12] else {}
                ))
            
            return runs
    
    def get_best_parameters_by_strategy(self, strategy_name: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get best parameter sets for a strategy"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT r.run_id, r.best_parameters, r.best_objective_value,
                       r.optimizer_type, r.start_time
                FROM optimization_runs r
                WHERE r.strategy_name = ? AND r.best_parameters IS NOT NULL
                ORDER BY r.best_objective_value DESC
                LIMIT ?
            """, (strategy_name, top_n))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'run_id': row[0],
                    'parameters': json.loads(row[1]),
                    'objective_value': row[2],
                    'optimizer_type': row[3],
                    'optimization_date': row[4]
                })
            
            return results
    
    def export_to_dataframe(self, run_id: str) -> pd.DataFrame:
        """Export parameter evaluations to pandas DataFrame"""
        
        query = """
            SELECT parameters, objective_value, metrics, is_valid, 
                   evaluation_time, timestamp, iteration
            FROM parameter_evaluations
            WHERE run_id = ?
            ORDER BY iteration
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(run_id,))
            
            # Parse JSON columns
            df['parameters'] = df['parameters'].apply(json.loads)
            df['metrics'] = df['metrics'].apply(json.loads)
            
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
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count runs by status
            cursor.execute("""
                SELECT status, COUNT(*) FROM optimization_runs GROUP BY status
            """)
            runs_by_status = dict(cursor.fetchall())
            
            # Count runs by strategy
            cursor.execute("""
                SELECT strategy_name, COUNT(*) FROM optimization_runs GROUP BY strategy_name
            """)
            runs_by_strategy = dict(cursor.fetchall())
            
            # Total evaluations
            cursor.execute("SELECT COUNT(*) FROM parameter_evaluations")
            total_evaluations = cursor.fetchone()[0]
            
            # Valid evaluations
            cursor.execute("SELECT COUNT(*) FROM parameter_evaluations WHERE is_valid = TRUE")
            valid_evaluations = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(start_time), MAX(start_time) FROM optimization_runs")
            date_range = cursor.fetchone()
            
            return {
                'total_runs': sum(runs_by_status.values()),
                'runs_by_status': runs_by_status,
                'runs_by_strategy': runs_by_strategy,
                'total_evaluations': total_evaluations,
                'valid_evaluations': valid_evaluations,
                'success_rate': valid_evaluations / total_evaluations if total_evaluations > 0 else 0,
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                }
            }
    
    def cleanup_old_runs(self, days_to_keep: int = 90):
        """Clean up old optimization runs"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get runs to delete
            cursor.execute("""
                SELECT run_id FROM optimization_runs 
                WHERE start_time < ? AND status != 'running'
            """, (cutoff_date,))
            
            old_run_ids = [row[0] for row in cursor.fetchall()]
            
            if old_run_ids:
                # Delete parameter evaluations
                placeholders = ','.join(['?' for _ in old_run_ids])
                cursor.execute(f"""
                    DELETE FROM parameter_evaluations 
                    WHERE run_id IN ({placeholders})
                """, old_run_ids)
                
                # Delete validation results
                cursor.execute(f"""
                    DELETE FROM validation_results 
                    WHERE run_id IN ({placeholders})
                """, old_run_ids)
                
                # Delete optimization runs
                cursor.execute(f"""
                    DELETE FROM optimization_runs 
                    WHERE run_id IN ({placeholders})
                """, old_run_ids)
                
                conn.commit()
                
                logger.info("Cleaned up old optimization runs",
                           deleted_runs=len(old_run_ids),
                           cutoff_date=cutoff_date)
            
            return len(old_run_ids)