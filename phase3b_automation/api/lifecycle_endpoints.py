"""
FastAPI Endpoints for Experiment Lifecycle Management
Phase 3B API Integration
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import asyncio

from api.models import (
    ExperimentPipelineRequest,
    ExperimentScheduleRequest, 
    ExperimentStatusResponse,
    ResourceStatusResponse,
    ExperimentListResponse,
    HealthCheckResponse
)
from core.experiment_lifecycle_manager import ExperimentLifecycleManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

def get_lifecycle_manager(request: Request) -> ExperimentLifecycleManager:
    """Dependency to get lifecycle manager from app state"""
    return request.app.state.lifecycle_manager

@router.post("/experiments/create-pipeline", 
             response_model=Dict[str, str],
             summary="Create Automated Experiment Pipeline",
             description="Create a complete automated experiment pipeline with intelligent scheduling")
async def create_experiment_pipeline(
    pipeline_request: ExperimentPipelineRequest,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Create automated experiment pipeline"""
    try:
        # Convert Pydantic model to dict
        experiment_config = pipeline_request.dict()
        
        # Create the pipeline
        experiment_id = await manager.create_experiment_pipeline(experiment_config)
        
        # Auto-schedule if requested
        if pipeline_request.auto_schedule:
            try:
                await manager.schedule_experiment(experiment_id)
                status = "created_and_scheduled"
            except Exception as e:
                logger.warning(f"Failed to auto-schedule experiment {experiment_id}: {e}")
                status = "created_but_scheduling_failed"
        else:
            status = "created"
        
        return {
            "experiment_id": experiment_id,
            "status": status,
            "message": f"Experiment pipeline created successfully: {experiment_id}",
            "next_steps": [
                f"Monitor progress: GET /automation/experiments/{experiment_id}/status",
                "View all experiments: GET /automation/experiments",
                f"Schedule manually: POST /automation/experiments/{experiment_id}/schedule"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating experiment pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create experiment pipeline: {str(e)}"
        )

@router.post("/experiments/{experiment_id}/schedule",
             response_model=Dict[str, str],
             summary="Schedule Experiment",
             description="Schedule an experiment for execution with optional custom configuration")
async def schedule_experiment(
    experiment_id: str,
    schedule_request: Optional[ExperimentScheduleRequest] = None,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Schedule experiment for execution"""
    try:
        # Check if experiment exists
        if experiment_id not in manager.experiments:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found"
            )
        
        # Get current experiment
        experiment = manager.experiments[experiment_id]
        
        # Check if already scheduled
        if experiment["state"].value in ["scheduled", "queued", "running", "completed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Experiment {experiment_id} is already {experiment['state'].value}"
            )
        
        # Create schedule config if provided
        schedule_config = None
        if schedule_request:
            from core.scheduler import ScheduleConfig
            from core.resource_manager import ResourceRequirement, ResourceType, Priority
            
            # Convert schedule request to ScheduleConfig
            resource_requirements = []
            if schedule_request.resource_requirements:
                for req_dict in schedule_request.resource_requirements:
                    resource_requirements.append(
                        ResourceRequirement(
                            resource_type=ResourceType(req_dict["resource_type"]),
                            amount=req_dict["amount"],
                            unit=req_dict["unit"],
                            priority=Priority(req_dict.get("priority", 2))
                        )
                    )
            
            schedule_config = ScheduleConfig(
                start_time=schedule_request.start_time,
                end_time=schedule_request.end_time,
                max_duration=schedule_request.max_duration,
                priority=Priority(schedule_request.priority),
                dependencies=schedule_request.dependencies or [],
                resource_requirements=resource_requirements,
                auto_start=schedule_request.auto_start
            )
        
        # Schedule the experiment
        await manager.schedule_experiment(experiment_id, schedule_config)
        
        return {
            "experiment_id": experiment_id,
            "status": "scheduled",
            "message": f"Experiment {experiment_id} scheduled successfully",
            "scheduled_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling experiment {experiment_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to schedule experiment: {str(e)}"
        )

@router.get("/experiments/{experiment_id}/status",
           response_model=ExperimentStatusResponse,
           summary="Get Experiment Status",
           description="Get comprehensive status and progress information for an experiment")
async def get_experiment_status(
    experiment_id: str,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Get comprehensive experiment status"""
    try:
        status_info = manager.get_experiment_status(experiment_id)
        
        if "error" in status_info:
            raise HTTPException(
                status_code=404,
                detail=status_info["error"]
            )
        
        return ExperimentStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for experiment {experiment_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiment status: {str(e)}"
        )

@router.get("/experiments",
           response_model=ExperimentListResponse, 
           summary="List All Experiments",
           description="Get status and summary information for all experiments")
async def list_all_experiments(
    state_filter: Optional[str] = None,
    owner_filter: Optional[str] = None,
    team_filter: Optional[str] = None,
    limit: int = 100,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """List all experiments with optional filtering"""
    try:
        all_experiments_info = manager.get_all_experiments()
        experiments = all_experiments_info["experiments"]
        
        # Apply filters
        if state_filter:
            experiments = [exp for exp in experiments if exp["state"] == state_filter]
        
        if owner_filter:
            experiments = [exp for exp in experiments if exp["owner"] == owner_filter]
        
        if team_filter:
            experiments = [exp for exp in experiments if exp["team"] == team_filter]
        
        # Apply limit
        experiments = experiments[:limit]
        
        return ExperimentListResponse(
            experiments=experiments,
            total_count=len(experiments),
            resource_utilization=all_experiments_info["resource_utilization"],
            scheduler_queue_length=all_experiments_info["scheduler_queue_length"],
            filters_applied={
                "state": state_filter,
                "owner": owner_filter,
                "team": team_filter,
                "limit": limit
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list experiments: {str(e)}"
        )

@router.post("/experiments/{experiment_id}/pause",
            response_model=Dict[str, str],
            summary="Pause Experiment",
            description="Pause a running experiment")
async def pause_experiment(
    experiment_id: str,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Pause a running experiment"""
    try:
        if experiment_id not in manager.experiments:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found"
            )
        
        experiment = manager.experiments[experiment_id]
        
        # Check if experiment can be paused
        if experiment["state"].value != "running":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause experiment in {experiment['state'].value} state"
            )
        
        # Pause the experiment (transition state)
        from core.state_machine import ExperimentState
        if manager.state_machine.transition_state(
            experiment_id,
            ExperimentState.RUNNING,
            ExperimentState.PAUSED,
            "Manual pause request"
        ):
            experiment["state"] = ExperimentState.PAUSED
            
            return {
                "experiment_id": experiment_id,
                "status": "paused",
                "message": f"Experiment {experiment_id} paused successfully",
                "paused_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to pause experiment"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing experiment {experiment_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pause experiment: {str(e)}"
        )

@router.delete("/experiments/{experiment_id}",
              response_model=Dict[str, str],
              summary="Cancel Experiment",
              description="Cancel an experiment and cleanup resources")
async def cancel_experiment(
    experiment_id: str,
    force: bool = False,
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Cancel an experiment"""
    try:
        if experiment_id not in manager.experiments:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment {experiment_id} not found"
            )
        
        experiment = manager.experiments[experiment_id]
        current_state = experiment["state"]
        
        # Check if experiment can be cancelled
        if current_state.value in ["completed", "failed", "cancelled"] and not force:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel experiment in {current_state.value} state (use force=true to override)"
            )
        
        # Stop monitoring if running
        if current_state.value in ["running", "paused"]:
            manager.monitor.stop_monitoring(experiment_id)
        
        # Release resources
        manager.resource_manager.release_resources(experiment_id)
        
        # Transition to cancelled state
        from core.state_machine import ExperimentState
        if manager.state_machine.transition_state(
            experiment_id,
            current_state,
            ExperimentState.CANCELLED,
            "Manual cancellation request"
        ):
            experiment["state"] = ExperimentState.CANCELLED
            experiment["cancelled_at"] = datetime.now()
            
            return {
                "experiment_id": experiment_id,
                "status": "cancelled",
                "message": f"Experiment {experiment_id} cancelled successfully",
                "cancelled_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel experiment"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling experiment {experiment_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel experiment: {str(e)}"
        )

@router.get("/resources/status",
           response_model=ResourceStatusResponse,
           summary="Get Resource Status",
           description="Get current resource utilization and allocation information")
async def get_resource_status(
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Get resource status and utilization"""
    try:
        utilization = manager.resource_manager.get_resource_utilization()
        allocations = manager.resource_manager.get_experiment_allocations()
        
        return ResourceStatusResponse(
            utilization=utilization,
            allocations=allocations,
            optimization_recommendations=[],
            total_active_allocations=len(manager.resource_manager.allocations),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting resource status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resource status: {str(e)}"
        )

@router.get("/health",
           response_model=HealthCheckResponse,
           summary="Health Check",
           description="Check health status of all lifecycle management components")
async def health_check(
    manager: ExperimentLifecycleManager = Depends(get_lifecycle_manager)
):
    """Health check for lifecycle management components"""
    try:
        # Check component health
        components_health = {
            "lifecycle_manager": manager is not None and manager._running,
            "resource_manager": manager.resource_manager is not None,
            "scheduler": manager.scheduler is not None,
            "state_machine": manager.state_machine is not None,
            "monitor": manager.monitor is not None
        }
        
        # Determine overall health
        all_healthy = all(components_health.values())
        overall_status = "healthy" if all_healthy else "degraded"
        
        # Get system statistics
        stats = {
            "active_experiments": len(manager.experiments),
            "scheduled_experiments": len(manager.scheduler.scheduled_experiments),
            "resource_utilization": manager.resource_manager.get_resource_utilization()
        }
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="3.1.0-BETA",
            components=components_health,
            statistics=stats
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="3.1.0-BETA",
            components={},
            statistics={},
            error=str(e)
        )