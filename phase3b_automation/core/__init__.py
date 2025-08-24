"""
Phase 3B Core Components
Automated experiment lifecycle management
"""

from .experiment_lifecycle_manager import ExperimentLifecycleManager
from .resource_manager import ResourceManager, ResourceType, ResourceRequirement
from .state_machine import ExperimentStateMachine, ExperimentState
from .scheduler import ExperimentScheduler, Priority, ScheduleConfig
from .monitor import ExperimentMonitor, ExperimentProgress

__all__ = [
    'ExperimentLifecycleManager',
    'ResourceManager',
    'ResourceType', 
    'ResourceRequirement',
    'ExperimentStateMachine',
    'ExperimentState',
    'ExperimentScheduler',
    'Priority',
    'ScheduleConfig',
    'ExperimentMonitor',
    'ExperimentProgress'
]

__version__ = "3.1.0-BETA"