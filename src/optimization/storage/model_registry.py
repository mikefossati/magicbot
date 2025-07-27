"""
Model Registry

Manages versioning, deployment, and lifecycle of optimized trading strategy models
with metadata tracking and performance monitoring.
"""

import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import structlog
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .postgres_database import OptimizationDatabase, OptimizationRun

logger = structlog.get_logger()

@dataclass
class ModelVersion:
    """Represents a versioned model in the registry"""
    model_id: str
    run_id: str
    model_name: str
    version: str
    strategy_name: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    validation_score: float
    is_deployed: bool
    deployment_date: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime

class ModelRegistry:
    """
    Registry for managing optimized trading strategy models.
    
    Features:
    - Model versioning and tracking
    - Deployment management
    - Performance monitoring
    - A/B testing support
    - Model comparison and rollback
    - Automated deployment based on criteria
    """
    
    def __init__(self, database: OptimizationDatabase, models_path: str = "models/"):
        """
        Initialize model registry.
        
        Args:
            database: Optimization database instance
            models_path: Directory to store serialized models
        """
        self.database = database
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    async def register_model(
        self,
        run_id: str,
        model_name: str,
        version: str,
        strategy_instance: Any,
        performance_metrics: Dict[str, float],
        validation_score: float,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            run_id: Optimization run ID
            model_name: Descriptive name for the model
            version: Version identifier (e.g., "1.0.0", "2024-01-15-v1")
            strategy_instance: The trained strategy instance
            performance_metrics: Performance metrics from backtesting
            validation_score: Validation score (e.g., walk-forward score)
            metadata: Additional metadata
            
        Returns:
            Model ID
        """
        
        # Get optimization run details
        run = await self.database.get_optimization_run(run_id)
        if not run:
            raise ValueError(f"Optimization run {run_id} not found")
        
        model_id = f"{model_name}_{version}_{run_id[:8]}"
        
        logger.info("Registering model",
                   model_id=model_id,
                   model_name=model_name,
                   version=version,
                   strategy=run.strategy_name)
        
        # Serialize and save strategy instance
        model_file_path = self.models_path / f"{model_id}.pkl"
        
        try:
            with open(model_file_path, 'wb') as f:
                pickle.dump({
                    'strategy': strategy_instance,
                    'parameters': run.best_parameters,
                    'metadata': metadata or {}
                }, f)
        except Exception as e:
            logger.error("Failed to serialize model", error=str(e))
            raise
        
        # Register in database
        await self.database.db.execute("""
            INSERT INTO model_registry (
                model_id, run_id, model_name, model_version, strategy_name,
                parameters, performance_metrics, validation_score, is_deployed,
                deployment_date, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, 
            model_id,
            run_id,
            model_name,
            version,
            run.strategy_name,
            json.dumps(run.best_parameters),
            json.dumps(performance_metrics),
            validation_score,
            False,
            None,
            json.dumps(metadata or {})
        )
        
        logger.info("Model registered successfully",
                   model_id=model_id,
                   file_path=str(model_file_path))
        
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model and its metadata.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (strategy_instance, model_metadata)
        """
        
        model_file_path = self.models_path / f"{model_id}.pkl"
        
        if not model_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        try:
            with open(model_file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Get model metadata from database
            model_info = self.get_model_info(model_id)
            
            return model_data['strategy'], {
                'parameters': model_data['parameters'],
                'model_metadata': model_data['metadata'],
                'registry_info': model_info
            }
            
        except Exception as e:
            logger.error("Failed to load model", model_id=model_id, error=str(e))
            raise
    
    def get_model_info(self, model_id: str) -> Optional[ModelVersion]:
        """Get model information from registry"""
        
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_id, run_id, model_name, model_version, strategy_name,
                       parameters, performance_metrics, validation_score, is_deployed,
                       deployment_date, metadata, created_at
                FROM model_registry WHERE model_id = ?
            """, (model_id,))
            
            row = cursor.fetchone()
            
            if row:
                return ModelVersion(
                    model_id=row[0],
                    run_id=row[1],
                    model_name=row[2],
                    version=row[3],
                    strategy_name=row[4],
                    parameters=json.loads(row[5]),
                    performance_metrics=json.loads(row[6]),
                    validation_score=row[7],
                    is_deployed=bool(row[8]),
                    deployment_date=datetime.fromisoformat(row[9]) if row[9] else None,
                    metadata=json.loads(row[10]) if row[10] else {},
                    created_at=datetime.fromisoformat(row[11])
                )
        
        return None
    
    async def list_models(
        self,
        strategy_name: Optional[str] = None,
        model_name: Optional[str] = None,
        deployed_only: bool = False,
        limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """
        List models with optional filters.
        
        Args:
            strategy_name: Filter by strategy name
            model_name: Filter by model name
            deployed_only: Only return deployed models
            limit: Maximum number of models to return
            
        Returns:
            List of model versions
        """
        
        query = """
            SELECT model_id, run_id, model_name, model_version, strategy_name,
                   parameters, performance_metrics, validation_score, is_deployed,
                   deployment_date, metadata, created_at
            FROM model_registry
            WHERE 1=1
        """
        
        params = []
        
        param_count = 1
        if strategy_name:
            query += f" AND strategy_name = ${param_count}"
            params.append(strategy_name)
            param_count += 1
        
        if model_name:
            query += f" AND model_name = ${param_count}"
            params.append(model_name)
            param_count += 1
        
        if deployed_only:
            query += " AND is_deployed = TRUE"
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT ${param_count}"
            params.append(limit)
        
        rows = await self.database.db.fetch_all(query, *params)
        
        models = []
        for row in rows:
            models.append(ModelVersion(
                    model_id=row[0],
                    run_id=row[1],
                    model_name=row[2],
                    version=row[3],
                    strategy_name=row[4],
                    parameters=json.loads(row[5]),
                    performance_metrics=json.loads(row[6]),
                    validation_score=row[7],
                    is_deployed=bool(row[8]),
                    deployment_date=datetime.fromisoformat(row[9]) if row[9] else None,
                    metadata=json.loads(row[10]) if row[10] else {},
                    created_at=datetime.fromisoformat(row[11])
                ))
            
            return models
    
    def deploy_model(self, model_id: str, force: bool = False) -> bool:
        """
        Deploy a model to production.
        
        Args:
            model_id: Model to deploy
            force: Force deployment even if validation fails
            
        Returns:
            True if deployment successful
        """
        
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        # Validation checks
        if not force:
            if not self._validate_deployment(model_info):
                logger.warning("Model failed deployment validation", model_id=model_id)
                return False
        
        # Undeploy existing models for the same strategy
        self._undeploy_strategy_models(model_info.strategy_name)
        
        # Deploy the new model
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_registry 
                SET is_deployed = TRUE, deployment_date = ?
                WHERE model_id = ?
            """, (datetime.now(), model_id))
            
            conn.commit()
        
        logger.info("Model deployed successfully",
                   model_id=model_id,
                   strategy=model_info.strategy_name)
        
        return True
    
    def undeploy_model(self, model_id: str):
        """Undeploy a model from production"""
        
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_registry 
                SET is_deployed = FALSE, deployment_date = NULL
                WHERE model_id = ?
            """, (model_id,))
            
            conn.commit()
        
        logger.info("Model undeployed", model_id=model_id)
    
    def get_deployed_model(self, strategy_name: str) -> Optional[ModelVersion]:
        """Get currently deployed model for a strategy"""
        
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT model_id, run_id, model_name, model_version, strategy_name,
                       parameters, performance_metrics, validation_score, is_deployed,
                       deployment_date, metadata, created_at
                FROM model_registry 
                WHERE strategy_name = ? AND is_deployed = TRUE
                ORDER BY deployment_date DESC
                LIMIT 1
            """, (strategy_name,))
            
            row = cursor.fetchone()
            
            if row:
                return ModelVersion(
                    model_id=row[0],
                    run_id=row[1],
                    model_name=row[2],
                    version=row[3],
                    strategy_name=row[4],
                    parameters=json.loads(row[5]),
                    performance_metrics=json.loads(row[6]),
                    validation_score=row[7],
                    is_deployed=bool(row[8]),
                    deployment_date=datetime.fromisoformat(row[9]) if row[9] else None,
                    metadata=json.loads(row[10]) if row[10] else {},
                    created_at=datetime.fromisoformat(row[11])
                )
        
        return None
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models across various metrics.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparison analysis
        """
        
        models = []
        for model_id in model_ids:
            model_info = self.get_model_info(model_id)
            if model_info:
                models.append(model_info)
        
        if len(models) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Create comparison matrix
        comparison_data = []
        
        for model in models:
            model_data = {
                'model_id': model.model_id,
                'model_name': model.model_name,
                'version': model.version,
                'strategy_name': model.strategy_name,
                'validation_score': model.validation_score,
                'is_deployed': model.is_deployed,
                'created_at': model.created_at
            }
            
            # Add performance metrics
            for metric, value in model.performance_metrics.items():
                model_data[f'metric_{metric}'] = value
            
            comparison_data.append(model_data)
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical comparison
        metric_columns = [col for col in df.columns if col.startswith('metric_')]
        
        comparison_stats = {}
        for metric in metric_columns:
            values = df[metric].dropna()
            if len(values) > 0:
                comparison_stats[metric] = {
                    'best_model': df.loc[df[metric].idxmax(), 'model_id'],
                    'worst_model': df.loc[df[metric].idxmin(), 'model_id'],
                    'mean': values.mean(),
                    'std': values.std(),
                    'range': values.max() - values.min()
                }
        
        # Ranking by validation score
        df_sorted = df.sort_values('validation_score', ascending=False)
        rankings = []
        
        for idx, row in df_sorted.iterrows():
            rankings.append({
                'rank': len(rankings) + 1,
                'model_id': row['model_id'],
                'model_name': row['model_name'],
                'validation_score': row['validation_score']
            })
        
        return {
            'models_compared': len(models),
            'comparison_data': comparison_data,
            'metric_statistics': comparison_stats,
            'rankings': rankings,
            'best_model': rankings[0] if rankings else None,
            'deployment_status': {
                'deployed_models': len([m for m in models if m.is_deployed]),
                'total_models': len(models)
            }
        }
    
    def auto_deploy_best_model(
        self,
        strategy_name: str,
        min_validation_score: float = 0.7,
        min_improvement: float = 0.05,
        max_age_days: int = 30
    ) -> Optional[str]:
        """
        Automatically deploy the best performing model for a strategy.
        
        Args:
            strategy_name: Strategy to deploy model for
            min_validation_score: Minimum validation score threshold
            min_improvement: Minimum improvement over current deployed model
            max_age_days: Maximum age of model to consider
            
        Returns:
            Model ID if deployed, None otherwise
        """
        
        # Get candidate models
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        candidates = []
        for model in self.list_models(strategy_name=strategy_name):
            if (model.validation_score >= min_validation_score and 
                model.created_at >= cutoff_date and
                not model.is_deployed):
                candidates.append(model)
        
        if not candidates:
            logger.info("No candidate models found for auto-deployment",
                       strategy=strategy_name)
            return None
        
        # Find best candidate
        best_candidate = max(candidates, key=lambda x: x.validation_score)
        
        # Check improvement over current deployed model
        current_deployed = self.get_deployed_model(strategy_name)
        
        if current_deployed:
            improvement = best_candidate.validation_score - current_deployed.validation_score
            if improvement < min_improvement:
                logger.info("Best candidate does not meet improvement threshold",
                           strategy=strategy_name,
                           improvement=improvement,
                           threshold=min_improvement)
                return None
        
        # Deploy the best candidate
        if self.deploy_model(best_candidate.model_id):
            logger.info("Auto-deployed best model",
                       strategy=strategy_name,
                       model_id=best_candidate.model_id,
                       validation_score=best_candidate.validation_score)
            return best_candidate.model_id
        
        return None
    
    def cleanup_old_models(self, days_to_keep: int = 90, keep_deployed: bool = True):
        """
        Clean up old models and their files.
        
        Args:
            days_to_keep: Number of days of models to keep
            keep_deployed: Whether to keep deployed models regardless of age
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            # Find models to delete
            query = """
                SELECT model_id FROM model_registry 
                WHERE created_at < ?
            """
            params = [cutoff_date]
            
            if keep_deployed:
                query += " AND is_deployed = FALSE"
            
            cursor.execute(query, params)
            old_model_ids = [row[0] for row in cursor.fetchall()]
            
            if old_model_ids:
                # Delete model files
                deleted_files = 0
                for model_id in old_model_ids:
                    model_file = self.models_path / f"{model_id}.pkl"
                    if model_file.exists():
                        model_file.unlink()
                        deleted_files += 1
                
                # Delete from database
                placeholders = ','.join(['?' for _ in old_model_ids])
                cursor.execute(f"""
                    DELETE FROM model_registry 
                    WHERE model_id IN ({placeholders})
                """, old_model_ids)
                
                conn.commit()
                
                logger.info("Cleaned up old models",
                           deleted_models=len(old_model_ids),
                           deleted_files=deleted_files,
                           cutoff_date=cutoff_date)
        
        return len(old_model_ids)
    
    def get_model_performance_history(
        self,
        strategy_name: str,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """
        Get performance history for models of a strategy.
        
        Args:
            strategy_name: Strategy to analyze
            days_back: Number of days to look back
            
        Returns:
            Performance history analysis
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        models = []
        for model in self.list_models(strategy_name=strategy_name):
            if model.created_at >= cutoff_date:
                models.append(model)
        
        if not models:
            return {'error': f'No models found for strategy {strategy_name}'}
        
        # Performance trends over time
        performance_data = []
        
        for model in models:
            model_data = {
                'model_id': model.model_id,
                'created_at': model.created_at,
                'validation_score': model.validation_score,
                'is_deployed': model.is_deployed
            }
            
            # Add performance metrics
            for metric, value in model.performance_metrics.items():
                model_data[metric] = value
            
            performance_data.append(model_data)
        
        df = pd.DataFrame(performance_data)
        df = df.sort_values('created_at')
        
        # Calculate trends
        trends = {}
        metric_columns = [col for col in df.columns 
                         if col not in ['model_id', 'created_at', 'is_deployed']]
        
        for metric in metric_columns:
            values = df[metric].dropna()
            if len(values) > 1:
                # Simple linear trend
                x = np.arange(len(values))
                z = np.polyfit(x, values, 1)
                trends[metric] = {
                    'slope': z[0],
                    'improving': z[0] > 0,
                    'latest_value': values.iloc[-1],
                    'best_value': values.max(),
                    'worst_value': values.min()
                }
        
        # Deployment analysis
        deployed_models = df[df['is_deployed'] == True]
        deployment_stats = {
            'total_deployments': len(deployed_models),
            'latest_deployment': deployed_models['created_at'].max() if not deployed_models.empty else None,
            'deployment_frequency': len(deployed_models) / max(1, days_back) * 30  # per month
        }
        
        return {
            'strategy_name': strategy_name,
            'analysis_period': {
                'days_back': days_back,
                'total_models': len(models),
                'date_range': {
                    'start': df['created_at'].min(),
                    'end': df['created_at'].max()
                }
            },
            'performance_trends': trends,
            'deployment_statistics': deployment_stats,
            'model_timeline': performance_data
        }
    
    def _validate_deployment(self, model_info: ModelVersion) -> bool:
        """Validate if a model is suitable for deployment"""
        
        # Basic validation criteria
        if model_info.validation_score < 0.5:
            logger.warning("Model validation score too low for deployment",
                          model_id=model_info.model_id,
                          score=model_info.validation_score)
            return False
        
        # Check if model file exists
        model_file = self.models_path / f"{model_info.model_id}.pkl"
        if not model_file.exists():
            logger.error("Model file not found",
                        model_id=model_info.model_id,
                        path=str(model_file))
            return False
        
        # Additional validation can be added here
        # - Minimum performance thresholds
        # - Strategy-specific requirements
        # - Risk management checks
        
        return True
    
    def _undeploy_strategy_models(self, strategy_name: str):
        """Undeploy all models for a strategy"""
        
        with self.database._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE model_registry 
                SET is_deployed = FALSE, deployment_date = NULL
                WHERE strategy_name = ? AND is_deployed = TRUE
            """, (strategy_name,))
            
            conn.commit()
    
    def _get_connection(self):
        """Get database connection (delegate to database)"""
        return self.database._get_connection()