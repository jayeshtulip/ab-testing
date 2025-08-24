"""
Real-time Experiment Monitor for Phase 3B
Progress tracking and alerting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .state_machine import ExperimentState

logger = logging.getLogger(__name__)

@dataclass
class ExperimentProgress:
    """Real-time experiment progress tracking"""
    experiment_id: str
    state: ExperimentState
    progress_percentage: float = 0.0
    current_phase: str = "initialization"
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None

class ExperimentMonitor:
    """Real-time experiment monitoring and progress tracking"""
    
    def __init__(self):
        self.active_experiments: Dict[str, ExperimentProgress] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_monitoring(self, experiment_id: str, metadata):
        """Start real-time monitoring for experiment"""
        progress = ExperimentProgress(
            experiment_id=experiment_id,
            state=ExperimentState.RUNNING,
            current_phase="initialization"
        )
        self.active_experiments[experiment_id] = progress
        
        # Start monitoring task
        self.monitoring_tasks[experiment_id] = asyncio.create_task(
            self._monitor_experiment_loop(experiment_id)
        )
        
        logger.info(f"Started monitoring for experiment {experiment_id}")
    
    async def _monitor_experiment_loop(self, experiment_id: str):
        """Continuous monitoring loop for experiment"""
        while experiment_id in self.active_experiments:
            try:
                # Update progress metrics
                await self._update_experiment_progress(experiment_id)
                
                # Check for alerts
                await self._check_experiment_alerts(experiment_id)
                
                # Update estimated completion
                self._update_completion_estimate(experiment_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring experiment {experiment_id}: {e}")
                break
    
    async def _update_experiment_progress(self, experiment_id: str):
        """Update experiment progress metrics"""
        if experiment_id not in self.active_experiments:
            return
        
        progress = self.active_experiments[experiment_id]
        
        # Simulate progress calculation (integrate with your Phase 3A monitoring)
        elapsed_time = (datetime.now() - progress.last_update).total_seconds()
        progress.progress_percentage = min(100.0, progress.progress_percentage + elapsed_time / 60)
        
        # Update current phase based on progress
        if progress.progress_percentage < 20:
            progress.current_phase = "data_preparation"
        elif progress.progress_percentage < 50:
            progress.current_phase = "model_training"
        elif progress.progress_percentage < 80:
            progress.current_phase = "validation"
        else:
            progress.current_phase = "finalization"
        
        progress.last_update = datetime.now()
    
    async def _check_experiment_alerts(self, experiment_id: str):
        """Check for experiment alerts and anomalies"""
        progress = self.active_experiments[experiment_id]
        
        # Clear old alerts
        progress.alerts = []
        
        # Example alert conditions
        if progress.progress_percentage > 0 and progress.progress_percentage < 10:
            time_stuck = (datetime.now() - progress.last_update).total_seconds()
            if time_stuck > 300:  # 5 minutes
                progress.alerts.append("Experiment appears to be stuck in initialization")
        
        # Performance alerts (integrate with your metrics)
        # This would connect to your Phase 3A statistical monitoring
    
    def _update_completion_estimate(self, experiment_id: str):
        """Update estimated completion time"""
        progress = self.active_experiments[experiment_id]
        
        if progress.progress_percentage > 5:  # Need some progress to estimate
            time_elapsed = (datetime.now() - progress.last_update).total_seconds()
            estimated_total_time = time_elapsed / (progress.progress_percentage / 100)
            remaining_time = estimated_total_time - time_elapsed
            progress.estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
    
    def stop_monitoring(self, experiment_id: str):
        """Stop monitoring for completed experiment"""
        if experiment_id in self.monitoring_tasks:
            self.monitoring_tasks[experiment_id].cancel()
            del self.monitoring_tasks[experiment_id]
        
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
        
        logger.info(f"Stopped monitoring for experiment {experiment_id}")
    
    def get_experiment_progress(self, experiment_id: str) -> Optional[ExperimentProgress]:
        """Get current experiment progress"""
        return self.active_experiments.get(experiment_id)