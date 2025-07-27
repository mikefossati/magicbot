"""
Optimization API Routes

REST API endpoints for managing parameter optimization jobs, monitoring progress,
and retrieving results.
"""

from typing import Dict, List, Any, Optional
import asyncio
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from pydantic import BaseModel, Field
import structlog

from ...optimization import (
    GridSearchOptimizer, 
    GeneticOptimizer, 
    MultiObjectiveOptimizer,
    ParameterSpace,
    CommonParameterSpaces,
    CommonObjectives,
    OptimizationConfig,
    WalkForwardValidator
)
from ...optimization.storage import OptimizationDatabase, ResultsAnalyzer, ModelRegistry
from ...strategies import get_strategy_factory
from ...database.connection import get_historical_data

logger = structlog.get_logger()

router = APIRouter(prefix="/optimization", tags=["optimization"])

# Global optimization job manager
optimization_jobs: Dict[str, Dict[str, Any]] = {}

class OptimizationJobRequest(BaseModel):
    """Request model for starting optimization jobs"""
    strategy_name: str = Field(..., description="Name of the strategy to optimize")
    optimizer_type: str = Field("grid_search", description="Type of optimizer (grid_search, genetic, multi_objective)")
    start_date: str = Field(..., description="Start date for backtesting (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtesting (YYYY-MM-DD)")
    parameter_space: Optional[Dict[str, Any]] = Field(None, description="Custom parameter space definition")
    objectives: List[str] = Field(["maximize_return"], description="Optimization objectives")
    config: Optional[Dict[str, Any]] = Field(None, description="Optimizer configuration")
    validation_enabled: bool = Field(True, description="Enable walk-forward validation")
    validation_config: Optional[Dict[str, Any]] = Field(None, description="Validation configuration")

class OptimizationJobResponse(BaseModel):
    """Response model for optimization job creation"""
    job_id: str
    status: str
    strategy_name: str
    optimizer_type: str
    created_at: datetime
    estimated_duration_minutes: Optional[int] = None

class OptimizationJobStatus(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    current_iteration: int
    total_iterations: int
    best_result: Optional[Dict[str, Any]] = None
    start_time: datetime
    elapsed_time_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    error_message: Optional[str] = None

class OptimizationResults(BaseModel):
    """Response model for optimization results"""
    job_id: str
    run_id: str
    strategy_name: str
    optimizer_type: str
    best_parameters: Dict[str, Any]
    best_objective_value: float
    total_evaluations: int
    optimization_time_seconds: float
    validation_results: Optional[Dict[str, Any]] = None
    convergence_analysis: Optional[Dict[str, Any]] = None

# Dependency to get optimization database
def get_optimization_db() -> OptimizationDatabase:
    return OptimizationDatabase()

# Dependency to get results analyzer
def get_results_analyzer(db: OptimizationDatabase = Depends(get_optimization_db)) -> ResultsAnalyzer:
    return ResultsAnalyzer(db)

# Dependency to get model registry
def get_model_registry(db: OptimizationDatabase = Depends(get_optimization_db)) -> ModelRegistry:
    return ModelRegistry(db)

@router.post("/jobs", response_model=OptimizationJobResponse)
async def start_optimization_job(
    request: OptimizationJobRequest,
    background_tasks: BackgroundTasks,
    db: OptimizationDatabase = Depends(get_optimization_db)
):
    """Start a new optimization job"""
    
    job_id = str(uuid.uuid4())
    
    logger.info("Starting optimization job",
               job_id=job_id,
               strategy=request.strategy_name,
               optimizer=request.optimizer_type)
    
    # Validate strategy exists
    try:
        strategy_factory = get_strategy_factory(request.strategy_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {str(e)}")
    
    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Create job entry
    job_info = {
        "job_id": job_id,
        "strategy_name": request.strategy_name,
        "optimizer_type": request.optimizer_type,
        "status": "queued",
        "progress": 0.0,
        "current_iteration": 0,
        "total_iterations": 0,
        "created_at": datetime.now(),
        "start_time": None,
        "error_message": None,
        "best_result": None,
        "run_id": None
    }
    
    optimization_jobs[job_id] = job_info
    
    # Start optimization in background
    background_tasks.add_task(
        run_optimization_job,
        job_id,
        request,
        strategy_factory,
        start_date,
        end_date,
        db
    )
    
    # Estimate duration (rough estimate based on optimizer type)
    estimated_duration = {
        "grid_search": 30,
        "genetic": 60,
        "multi_objective": 90
    }.get(request.optimizer_type, 45)
    
    return OptimizationJobResponse(
        job_id=job_id,
        status="queued",
        strategy_name=request.strategy_name,
        optimizer_type=request.optimizer_type,
        created_at=job_info["created_at"],
        estimated_duration_minutes=estimated_duration
    )

async def run_optimization_job(
    job_id: str,
    request: OptimizationJobRequest,
    strategy_factory,
    start_date: datetime,
    end_date: datetime,
    db: OptimizationDatabase
):
    """Background task to run optimization job"""
    
    try:
        job_info = optimization_jobs[job_id]
        job_info["status"] = "running"
        job_info["start_time"] = datetime.now()
        
        logger.info("Executing optimization job", job_id=job_id)
        
        # Get historical data
        historical_data = await get_historical_data(start_date, end_date)
        
        # Create parameter space
        if request.parameter_space:
            parameter_space = ParameterSpace(request.parameter_space)
        else:
            parameter_space = getattr(CommonParameterSpaces, request.strategy_name, lambda: CommonParameterSpaces.momentum_strategy())()
        
        # Create objectives
        objectives = []
        for obj_name in request.objectives:
            objective_func = getattr(CommonObjectives, obj_name, lambda: CommonObjectives.maximize_return())
            objectives.append(objective_func())
        
        # Create optimizer configuration
        config_dict = request.config or {}
        config = OptimizationConfig(**config_dict)
        
        # Create optimizer
        if request.optimizer_type == "grid_search":
            optimizer = GridSearchOptimizer(
                parameter_space=parameter_space,
                objective=objectives[0],
                config=config
            )
        elif request.optimizer_type == "genetic":
            optimizer = GeneticOptimizer(
                parameter_space=parameter_space,
                objective=objectives[0],
                config=config
            )
        elif request.optimizer_type == "multi_objective":
            optimizer = MultiObjectiveOptimizer(
                parameter_space=parameter_space,
                objectives=objectives,
                config=config
            )
        else:
            raise ValueError(f"Unknown optimizer type: {request.optimizer_type}")
        
        # Set up progress callback
        def progress_callback(iteration: int, total: int, best_result=None):
            job_info["current_iteration"] = iteration
            job_info["total_iterations"] = total
            job_info["progress"] = iteration / total if total > 0 else 0
            if best_result:
                job_info["best_result"] = {
                    "parameters": best_result.parameters,
                    "objective_value": best_result.objective_value,
                    "metrics": best_result.metrics
                }
        
        optimizer.set_progress_callback(progress_callback)
        
        # Run optimization
        result = await optimizer.optimize(
            strategy_factory=strategy_factory,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Get run ID from optimizer
        run_id = optimizer.state.run_id if hasattr(optimizer.state, 'run_id') else str(uuid.uuid4())
        job_info["run_id"] = run_id
        
        # Run validation if enabled
        validation_results = None
        if request.validation_enabled:
            logger.info("Running walk-forward validation", job_id=job_id)
            
            validation_config = request.validation_config or {}
            validator = WalkForwardValidator(**validation_config)
            
            validation_results = await validator.validate_parameters(
                parameters=result.parameters,
                strategy_factory=strategy_factory,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
        
        # Update job status
        job_info["status"] = "completed"
        job_info["progress"] = 1.0
        job_info["validation_results"] = validation_results
        job_info["final_result"] = {
            "parameters": result.parameters,
            "objective_value": result.objective_value,
            "metrics": result.metrics,
            "is_valid": result.is_valid
        }
        
        logger.info("Optimization job completed successfully",
                   job_id=job_id,
                   run_id=run_id,
                   objective_value=result.objective_value)
        
    except Exception as e:
        logger.error("Optimization job failed", job_id=job_id, error=str(e))
        job_info["status"] = "failed"
        job_info["error_message"] = str(e)

@router.get("/jobs/{job_id}/status", response_model=OptimizationJobStatus)
async def get_job_status(job_id: str):
    """Get status of an optimization job"""
    
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = optimization_jobs[job_id]
    
    # Calculate timing information
    start_time = job_info.get("start_time") or job_info["created_at"]
    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    
    estimated_remaining = None
    if job_info["status"] == "running" and job_info["progress"] > 0:
        estimated_total = elapsed_seconds / job_info["progress"]
        estimated_remaining = max(0, estimated_total - elapsed_seconds)
    
    return OptimizationJobStatus(
        job_id=job_id,
        status=job_info["status"],
        progress=job_info["progress"],
        current_iteration=job_info["current_iteration"],
        total_iterations=job_info["total_iterations"],
        best_result=job_info.get("best_result"),
        start_time=start_time,
        elapsed_time_seconds=elapsed_seconds,
        estimated_remaining_seconds=estimated_remaining,
        error_message=job_info.get("error_message")
    )

@router.get("/jobs/{job_id}/results", response_model=OptimizationResults)
async def get_job_results(
    job_id: str,
    include_analysis: bool = Query(False, description="Include convergence analysis"),
    db: OptimizationDatabase = Depends(get_optimization_db),
    analyzer: ResultsAnalyzer = Depends(get_results_analyzer)
):
    """Get results of a completed optimization job"""
    
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    run_id = job_info.get("run_id")
    if not run_id:
        raise HTTPException(status_code=500, detail="Run ID not found")
    
    # Get optimization run from database
    run = await db.get_optimization_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Optimization run not found in database")
    
    # Get convergence analysis if requested
    convergence_analysis = None
    if include_analysis:
        convergence_analysis = await analyzer.analyze_convergence(run_id)
    
    optimization_time = 0
    if run.end_time and run.start_time:
        optimization_time = (run.end_time - run.start_time).total_seconds()
    
    return OptimizationResults(
        job_id=job_id,
        run_id=run_id,
        strategy_name=run.strategy_name,
        optimizer_type=run.optimizer_type,
        best_parameters=run.best_parameters or {},
        best_objective_value=run.best_objective_value or 0,
        total_evaluations=run.total_evaluations,
        optimization_time_seconds=optimization_time,
        validation_results=job_info.get("validation_results"),
        convergence_analysis=convergence_analysis
    )

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running optimization job"""
    
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = optimization_jobs[job_id]
    
    if job_info["status"] not in ["queued", "running"]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    job_info["status"] = "cancelled"
    
    logger.info("Optimization job cancelled", job_id=job_id)
    
    return {"message": "Job cancelled successfully"}

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(50, description="Maximum number of jobs to return")
):
    """List optimization jobs"""
    
    jobs = []
    for job_info in optimization_jobs.values():
        if status and job_info["status"] != status:
            continue
        if strategy_name and job_info["strategy_name"] != strategy_name:
            continue
        
        jobs.append({
            "job_id": job_info["job_id"],
            "status": job_info["status"],
            "strategy_name": job_info["strategy_name"],
            "optimizer_type": job_info["optimizer_type"],
            "progress": job_info["progress"],
            "created_at": job_info["created_at"]
        })
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"jobs": jobs[:limit]}

@router.get("/runs")
async def list_optimization_runs(
    strategy_name: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    days_back: Optional[int] = Query(7),
    limit: Optional[int] = Query(50),
    db: OptimizationDatabase = Depends(get_optimization_db)
):
    """List optimization runs from database"""
    
    runs = await db.get_optimization_runs(
        strategy_name=strategy_name,
        status=status,
        days_back=days_back,
        limit=limit
    )
    
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "strategy_name": run.strategy_name,
                "optimizer_type": run.optimizer_type,
                "status": run.status,
                "best_objective_value": run.best_objective_value,
                "total_evaluations": run.total_evaluations,
                "start_time": run.start_time,
                "end_time": run.end_time
            }
            for run in runs
        ]
    }

@router.get("/runs/{run_id}/analysis")
async def get_run_analysis(
    run_id: str,
    analysis_type: str = Query("sensitivity", description="Type of analysis (sensitivity, convergence, comparison)"),
    analyzer: ResultsAnalyzer = Depends(get_results_analyzer)
):
    """Get detailed analysis of an optimization run"""
    
    if analysis_type == "sensitivity":
        result = await analyzer.analyze_parameter_sensitivity(run_id)
    elif analysis_type == "convergence":
        result = await analyzer.analyze_convergence(run_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid analysis type")
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/runs/compare")
async def compare_runs(
    run_ids: List[str],
    metrics: Optional[List[str]] = None,
    analyzer: ResultsAnalyzer = Depends(get_results_analyzer)
):
    """Compare multiple optimization runs"""
    
    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 runs for comparison")
    
    result = await analyzer.compare_optimization_runs(run_ids, metrics)
    
    return result

@router.get("/strategies/{strategy_name}/performance")
async def get_strategy_performance(
    strategy_name: str,
    days_back: int = Query(30),
    analyzer: ResultsAnalyzer = Depends(get_results_analyzer)
):
    """Get performance analysis for a specific strategy"""
    
    result = await analyzer.analyze_strategy_performance(strategy_name, days_back)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@router.get("/database/stats")
async def get_database_stats(db: OptimizationDatabase = Depends(get_optimization_db)):
    """Get optimization database statistics"""
    
    return await db.get_database_statistics()

@router.post("/database/cleanup")
async def cleanup_database(
    days_to_keep: int = Query(90, description="Number of days of data to keep"),
    db: OptimizationDatabase = Depends(get_optimization_db)
):
    """Clean up old optimization data"""
    
    deleted_count = await db.cleanup_old_runs(days_to_keep)
    
    return {
        "message": f"Cleaned up {deleted_count} old optimization runs",
        "days_kept": days_to_keep
    }

# Model Registry Endpoints

@router.post("/models/register")
async def register_model(
    run_id: str,
    model_name: str,
    version: str,
    performance_metrics: Dict[str, float],
    validation_score: float,
    metadata: Optional[Dict[str, Any]] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Register a new model version"""
    
    # Note: In a real implementation, you'd need to serialize the actual strategy instance
    # For now, we'll create a placeholder
    strategy_instance = {"placeholder": "strategy_data"}
    
    try:
        model_id = await registry.register_model(
            run_id=run_id,
            model_name=model_name,
            version=version,
            strategy_instance=strategy_instance,
            performance_metrics=performance_metrics,
            validation_score=validation_score,
            metadata=metadata or {}
        )
        
        return {"model_id": model_id, "message": "Model registered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models")
async def list_models(
    strategy_name: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    deployed_only: bool = Query(False),
    limit: Optional[int] = Query(50),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """List registered models"""
    
    models = await registry.list_models(
        strategy_name=strategy_name,
        model_name=model_name,
        deployed_only=deployed_only,
        limit=limit
    )
    
    return {
        "models": [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "strategy_name": model.strategy_name,
                "validation_score": model.validation_score,
                "is_deployed": model.is_deployed,
                "created_at": model.created_at
            }
            for model in models
        ]
    }

@router.post("/models/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    force: bool = Query(False),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Deploy a model to production"""
    
    try:
        success = await registry.deploy_model(model_id, force=force)
        
        if success:
            return {"message": "Model deployed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Model failed deployment validation")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/models/{model_id}/undeploy")
async def undeploy_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Undeploy a model from production"""
    
    try:
        await registry.undeploy_model(model_id)
        return {"message": "Model undeployed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/compare")
async def compare_models(
    model_ids: List[str] = Query(...),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Compare multiple models"""
    
    result = await registry.compare_models(model_ids)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/strategies/{strategy_name}/auto-deploy")
async def auto_deploy_strategy(
    strategy_name: str,
    min_validation_score: float = Query(0.7),
    min_improvement: float = Query(0.05),
    max_age_days: int = Query(30),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """Automatically deploy the best model for a strategy"""
    
    model_id = await registry.auto_deploy_best_model(
        strategy_name=strategy_name,
        min_validation_score=min_validation_score,
        min_improvement=min_improvement,
        max_age_days=max_age_days
    )
    
    if model_id:
        return {
            "message": "Model auto-deployed successfully",
            "model_id": model_id
        }
    else:
        return {
            "message": "No suitable model found for auto-deployment"
        }