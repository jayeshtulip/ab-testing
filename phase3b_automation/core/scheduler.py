"""
Intelligent Scheduler for Phase 3B
Priority-based scheduling with dependency management
"""

import logging
import heapq
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .resource_manager import ResourceManager, ResourceRequirement

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Experiment priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ScheduleConfig:
    """Experiment scheduling configuration"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_duration: Optional[timedelta] = None
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    auto_start: bool = True
    retry_config: Optional[Dict[str, Any]] = None

@dataclass
class ExperimentMetadata:
    """Comprehensive experiment metadata"""
    experiment_id: str
    name: str
    description: str
    owner: str
    team: str
    project: str
    tags: List[str] = field(default_factory=list)
    business_impact: Optional[str] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class ExperimentScheduler:
    """Intelligent experiment scheduling with priority and dependency management"""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.scheduled_experiments = []  # Priority queue
        self.dependency_graph: Dict[str, List[str]] = {}
        self.completed_experiments: set = set()
        
    def schedule_experiment(self, experiment_id: str, schedule_config: ScheduleConfig, 
                          metadata: ExperimentMetadata) -> bool:
        """Schedule experiment with intelligent resource and dependency management"""
        
        # Check resource availability
        if not self.resource_manager.can_allocate_resources(schedule_config.resource_requirements):
            logger.warning(f"Cannot schedule {experiment_id}: insufficient resources")
            return False
        
        # Calculate priority score (higher score = higher priority)
        priority_score = self._calculate_priority_score(schedule_config, metadata)
        
        # Add to scheduler queue
        scheduled_time = schedule_config.start_time or datetime.now()
        heapq.heappush(self.scheduled_experiments, (
            -priority_score,  # Negative for max heap behavior
            scheduled_time.timestamp(),
            experiment_id,
            schedule_config,
            metadata
        ))
        
        # Track dependencies
        if schedule_config.dependencies:
            self.dependency_graph[experiment_id] = schedule_config.dependencies
        
        logger.info(f"Experiment {experiment_id} scheduled with priority {priority_score}")
        return True
    
    def get_next_experiment(self) -> Optional[tuple]:
        """Get next experiment ready for execution"""
        while self.scheduled_experiments:
            priority_score, scheduled_time, experiment_id, config, metadata = self.scheduled_experiments[0]
            
            # Check if it's time to run
            if datetime.now().timestamp() < scheduled_time:
                break
            
            # Check dependencies
            if self._dependencies_satisfied(experiment_id):
                return heapq.heappop(self.scheduled_experiments)[2:]  # Return without priority and time
            else:
                # Re-schedule for later if dependencies not met
                heapq.heappop(self.scheduled_experiments)
                later_time = datetime.now() + timedelta(minutes=5)
                heapq.heappush(self.scheduled_experiments, (
                    priority_score, later_time.timestamp(), experiment_id, config, metadata
                ))
        
        return None
    
    def _calculate_priority_score(self, config: ScheduleConfig, metadata: ExperimentMetadata) -> float:
        """Calculate intelligent priority score based on multiple factors"""
        base_score = config.priority.value * 10
        
        # Business impact modifier
        impact_modifier = {
            "critical": 20,
            "high": 15,
            "medium": 10,
            "low": 5
        }.get(metadata.business_impact, 10)
        
        # Urgency modifier (closer to deadline = higher priority)
        urgency_modifier = 0
        if config.end_time:
            time_remaining = (config.end_time - datetime.now()).total_seconds()
            if time_remaining > 0:
                urgency_modifier = max(0, 10 - (time_remaining / 86400))  # Daily urgency increase
        
        # Team modifier (certain teams get priority)
        team_modifier = {
            "ml-platform": 5,
            "data-science": 3,
            "research": 2
        }.get(metadata.team.lower(), 1)
        
        return base_score + impact_modifier + urgency_modi