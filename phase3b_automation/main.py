"""
Pydantic Models for Phase 3B API
Request and Response models for experiment lifecycle management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class PriorityEnum(str, Enum):
    """Priority levels for API"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ResourceTypeEnum(str, Enum):
    """Resource types for API"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    MODEL_SLOTS = "model_slots"

class ExperimentStateEnum(str, Enum):
    """Experiment states for API"""
    CREATED = "created"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

# Request Models

class ResourceRequirementRequest(BaseModel):
    """Resource requirement specification for API"""
    resource_type: ResourceTypeEnum
    amount: float = Field(..., gt=0, description="Amount of resource required")
    unit: str = Field(..., description="Unit of measurement")
    max_amount: Optional[float] = Field(None, gt=0, description="Maximum amount if scalable")
    priority: PriorityEnum = PriorityEnum.MEDIUM

    @field_validator('max_amount')
    @classmethod
    def validate_max_amount(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None and info.data.get('amount') is not None and v < info.data['amount']:
            raise ValueError('max_amount cannot be less than amount')
        return v

class ExperimentPipelineRequest(BaseModel):
    """Request model for creating experiment pipeline"""
    name: str = Field(..., min_length=1, max_length=200, description="Experiment name")
    description: str = Field("", max_length=1000, description="Experiment description")
    owner: str = Field(..., description="Experiment owner")
    team: str = Field(..., description="Team responsible for experiment")
    project: str = Field("default", description="Project name")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    
    # Business context
    business_impact: str = Field("medium", description="Business impact level")
    priority: int = Field(2, ge=1, le=4, description="Priority level (1=low, 4=critical)")
    
    # Technical configuration
    compute_requirement: float = Field(10.0, gt=0, description="Compute resources needed")
    storage_requirement: float = Field(50.0, gt=0, description="Storage resources needed")
    
    # Scheduling
    auto_schedule: bool = Field(True, description="Automatically schedule after creation")
    start_time: Optional[datetime] = Field(None, description="Preferred start time")
    end_time: Optional[datetime] = Field(None, description="Required completion time")
    max_duration_hours: Optional[int] = Field(None, gt=0, le=168, description="Maximum duration in hours")
    
    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Experiment dependencies")
    
    # Success criteria
    success_criteria: Dict[str, Any] = Field(default_factory=dict, description="Success criteria")
    
    # Phase 3A integration
    phase3a_config: Dict[str, Any] = Field(default_factory=dict, description="Phase 3A specific configuration")

    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v: Optional[datetime], info) -> Optional[datetime]:
        if v is not None and info.data.get('start_time') is not None:
            if v <= info.data['start_time']:
                raise ValueError('end_time must be after start_time')
        return v

class ExperimentScheduleRequest(BaseModel):
    """Request model for scheduling experiment"""
    start_time: Optional[datetime] = Field(None, description="Scheduled start time")
    end_time: Optional[datetime] = Field(None, description="Required completion time")
    max_duration: Optional[timedelta] = Field(None, description="Maximum duration")
    priority: int = Field(2, ge=1, le=4, description="Priority level")
    dependencies: Optional[List[str]] = Field(None, description="Experiment dependencies")
    resource_requirements: Optional[List[Dict[str, Any]]] = Field(None, description="Custom resource requirements")
    auto_start: bool = Field(True, description="Start automatically when scheduled")

# Response Models

class ExperimentMetadataResponse(BaseModel):
    """Experiment metadata for responses"""
    experiment_id: str
    name: str
    description: str
    owner: str
    team: str
    project: str
    tags: List[str]
    business_impact: Optional[str]
    success_criteria: Dict[str, Any]
    created_at: str
    updated_at: str

class ExperimentProgressResponse(BaseModel):
    """Experiment progress information"""
    experiment_id: str
    state: ExperimentStateEnum
    progress_percentage: float = Field(..., ge=0, le=100)
    current_phase: str
    metrics: Dict[str, Any]
    alerts: List[str]
    last_update: str
    estimated_completion: Optional[str]

class StateTransitionResponse(BaseModel):
    """State transition history entry"""
    timestamp: str
    from_state: ExperimentStateEnum
    to_state: ExperimentStateEnum
    reason: str

class ExperimentStatusResponse(BaseModel):
    """Comprehensive experiment status response"""
    experiment_id: str
    metadata: ExperimentMetadataResponse
    state: ExperimentStateEnum
    progress: Optional[ExperimentProgressResponse]
    state_history: List[StateTransitionResponse]
    resource_utilization: Dict[str, Any]

class ExperimentSummaryResponse(BaseModel):
    """Summary information for experiment list"""
    experiment_id: str
    name: str
    state: ExperimentStateEnum
    created_at: str
    owner: str
    team: str
    progress_percentage: Optional[float] = None

class ExperimentListResponse(BaseModel):
    """Response for listing experiments"""
    experiments: List[ExperimentSummaryResponse]
    total_count: int
    resource_utilization: Dict[str, Any]
    scheduler_queue_length: int
    filters_applied: Dict[str, Any]

class ResourceUtilizationResponse(BaseModel):
    """Resource utilization information"""
    used: float
    available: float
    limit: float
    utilization_percentage: float

class ResourceAllocationResponse(BaseModel):
    """Resource allocation information"""
    experiment_id: str
    allocated_at: str
    priority: str
    resources: List[Dict[str, Any]]

class OptimizationRecommendation(BaseModel):
    """Resource optimization recommendation"""
    type: str = Field(..., description="Recommendation type: warning, info, optimization")
    resource: Optional[str] = Field(None, description="Affected resource")
    message: str = Field(..., description="Recommendation message")
    current_usage: Optional[float] = Field(None, description="Current usage")
    limit: Optional[float] = Field(None, description="Resource limit")

class ResourceStatusResponse(BaseModel):
    """Comprehensive resource status response"""
    utilization: Dict[str, ResourceUtilizationResponse]
    allocations: Dict[str, Any]
    optimization_recommendations: List[OptimizationRecommendation]
    total_active_allocations: int
    timestamp: str

class ComponentHealthResponse(BaseModel):
    """Health status of individual components"""
    lifecycle_manager: bool
    resource_manager: bool
    scheduler: bool
    state_machine: bool
    monitor: bool

class SystemStatisticsResponse(BaseModel):
    """System statistics for health check"""
    active_experiments: int
    scheduled_experiments: int
    resource_utilization: Dict[str, Any]

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    timestamp: str
    version: str
    components: ComponentHealthResponse
    statistics: SystemStatisticsResponse
    error: Optional[str] = Field(None, description="Error message if unhealthy")

# Metrics Models

class ExperimentMetrics(BaseModel):
    """Experiment-related metrics"""
    total: int
    by_state: Dict[str, int]
    by_team: Dict[str, int]
    by_owner: Dict[str, int]

class SchedulerMetrics(BaseModel):
    """Scheduler-related metrics"""
    queue_length: int
    completed_experiments: int

class SystemMetrics(BaseModel):
    """System-level metrics"""
    uptime_seconds: float
    version: str
    timestamp: str

class MetricsResponse(BaseModel):
    """Comprehensive metrics response"""
    experiments: ExperimentMetrics
    resources: Dict[str, ResourceUtilizationResponse]
    scheduler: SchedulerMetrics
    system: SystemMetrics

# Error Models

class ErrorDetail(BaseModel):
    """Error detail information"""
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: ErrorDetail
    timestamp: str
    request_id: Optional[str] = None