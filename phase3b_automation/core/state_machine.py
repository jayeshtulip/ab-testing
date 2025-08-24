"""
State Machine for Phase 3B
Manages experiment state transitions with validation
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class ExperimentState(Enum):
    """Experiment lifecycle states"""
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

class ExperimentStateMachine:
    """Manages experiment state transitions with validation"""
    
    VALID_TRANSITIONS = {
        ExperimentState.CREATED: [ExperimentState.SCHEDULED, ExperimentState.CANCELLED],
        ExperimentState.SCHEDULED: [ExperimentState.QUEUED, ExperimentState.CANCELLED],
        ExperimentState.QUEUED: [ExperimentState.RUNNING, ExperimentState.CANCELLED],
        ExperimentState.RUNNING: [ExperimentState.PAUSED, ExperimentState.COMPLETING, ExperimentState.FAILED],
        ExperimentState.PAUSED: [ExperimentState.RUNNING, ExperimentState.CANCELLED],
        ExperimentState.COMPLETING: [ExperimentState.COMPLETED, ExperimentState.FAILED],
        ExperimentState.COMPLETED: [ExperimentState.ARCHIVED],
        ExperimentState.FAILED: [ExperimentState.SCHEDULED, ExperimentState.CANCELLED],
        ExperimentState.CANCELLED: [ExperimentState.SCHEDULED],
        ExperimentState.ARCHIVED: []
    }
    
    def __init__(self):
        self.state_history: Dict[str, List[Tuple]] = {}
    
    def transition_state(self, experiment_id: str, current_state: ExperimentState, 
                        new_state: ExperimentState, reason: str = "") -> bool:
        """Safely transition experiment state with validation"""
        if new_state not in self.VALID_TRANSITIONS.get(current_state, []):
            logger.error(f"Invalid state transition for {experiment_id}: {current_state} -> {new_state}")
            return False
        
        # Record state transition
        if experiment_id not in self.state_history:
            self.state_history[experiment_id] = []
        
        self.state_history[experiment_id].append((
            datetime.now(),
            current_state,
            new_state,
            reason
        ))
        
        logger.info(f"Experiment {experiment_id} transitioned: {current_state} -> {new_state}")
        return True
    
    def get_state_history(self, experiment_id: str) -> List[Tuple]:
        """Get complete state transition history"""
        return self.state_history.get(experiment_id, [])