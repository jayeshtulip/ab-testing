"""
Phase 3B: Experiment Lifecycle Manager
Main orchestrator for automated experiment lifecycle management
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from .resource_manager import ResourceManager, ResourceRequirement, ResourceType, Priority
from .scheduler import ExperimentScheduler, ScheduleConfig, ExperimentMetadata
from .state_machine import ExperimentStateMachine, ExperimentState
from .monitor import ExperimentMonitor

logger = logging.getLogger(__name__)

class ExperimentLifecycleManager:
    """
    Main experiment lifecycle manager with automated scheduling and resource optimization
    """
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.scheduler = ExperimentScheduler(self.resource_manager)
        self.state_machine = ExperimentStateMachine()
        self.monitor = ExperimentMonitor()
        self.experiments: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._running = False
        self._scheduler_task = None
    
    async def start(self):
        """Start the lifecycle manager"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Experiment Lifecycle Manager started")
    
    async def stop(self):
        """Stop the lifecycle manager"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        self.executor.shutdown(wait=True)
        logger.info("Experiment Lifecycle Manager stopped")
    
    async def create_experiment_pipeline(self, experiment_config: Dict[str, Any]) -> str:
        """Create complete automated experiment pipeline"""
        experiment_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=experiment_config.get("name", f"Experiment-{experiment_id[:8]}"),
            description=experiment_config.get("description", ""),
            owner=experiment_config.get("owner", "system"),
            team=experiment_config.get("team", "default"),
            project=experiment_config.get("project", "default"),
            tags=experiment_config.get("tags", []),
            business_impact=experiment_config.get("business_impact", "medium"),
            success_criteria=experiment_config.get("success_criteria", {})
        )
        
        # Create schedule configuration
        schedule_config = ScheduleConfig(
            start_time=experiment_config.get("start_time"),
            end_time=experiment_config.get("end_time"),
            max_duration=experiment_config.get("max_duration"),
            priority=Priority(experiment_config.get("priority", 2)),
            dependencies=experiment_config.get("dependencies", []),
            resource_requirements=[
                ResourceRequirement(
                    resource_type=ResourceType.COMPUTE,
                    amount=experiment_config.get("compute_requirement", 10.0),
                    unit="cores"
                ),
                ResourceRequirement(
                    resource_type=ResourceType.MODEL_SLOTS,
                    amount=1.0,
                    unit="slots"
                )
            ],
            auto_start=experiment_config.get("auto_start", True)
        )
        
        # Store experiment
        self.experiments[experiment_id] = {
            "metadata": metadata,
            "schedule_config": schedule_config,
            "config": experiment_config,
            "state": ExperimentState.CREATED,
            "created_at": datetime.now()
        }
        
        logger.info(f"Created experiment pipeline: {experiment_id}")
        return experiment_id
    
    async def schedule_experiment(self, experiment_id: str, schedule_config: Optional[ScheduleConfig] = None):
        """Schedule experiment for execution"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        config = schedule_config or experiment["schedule_config"]
        
        # Update state
        if self.state_machine.transition_state(
            experiment_id, 
            experiment["state"], 
            ExperimentState.SCHEDULED,
            "Automated scheduling"
        ):
            experiment["state"] = ExperimentState.SCHEDULED
        
        # Schedule with scheduler
        success = self.scheduler.schedule_experiment(
            experiment_id, config, experiment["metadata"]
        )
        
        if success:
            logger.info(f"Experiment {experiment_id} scheduled successfully")
        else:
            # Revert state
            experiment["state"] = ExperimentState.CREATED
            raise RuntimeError(f"Failed to schedule experiment {experiment_id}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop for processing queued experiments"""
        while self._running:
            try:
                # Get next experiment to execute
                next_experiment = self.scheduler.get_next_experiment()
                
                if next_experiment:
                    experiment_id, config, metadata = next_experiment
                    await self._execute_experiment(experiment_id, config, metadata)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_experiment(self, experiment_id: str, config: ScheduleConfig, 
                                 metadata: ExperimentMetadata):
        """Execute experiment with full lifecycle management"""
        experiment = self.experiments[experiment_id]
        
        try:
            # Allocate resources
            if not self.resource_manager.allocate_resources(experiment_id, config.resource_requirements):
                logger.error(f"Failed to allocate resources for {experiment_id}")
                return
            
            # Transition to running state
            if self.state_machine.transition_state(
                experiment_id, 
                ExperimentState.SCHEDULED, 
                ExperimentState.RUNNING,
                "Starting execution"
            ):
                experiment["state"] = ExperimentState.RUNNING
            
            # Start monitoring
            await self.monitor.start_monitoring(experiment_id, metadata)
            
            # Execute experiment (integrate with your Phase 3A experiment execution)
            await self._run_experiment_logic(experiment_id, experiment["config"])
            
            # Transition to completing
            if self.state_machine.transition_state(
                experiment_id,
                ExperimentState.RUNNING,
                ExperimentState.COMPLETING,
                "Experiment execution completed"
            ):
                experiment["state"] = ExperimentState.COMPLETING
            
            # Finalize and cleanup
            await self._finalize_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Error executing experiment {experiment_id}: {e}")
            await self._handle_experiment_failure(experiment_id, str(e))
    
    async def _run_experiment_logic(self, experiment_id: str, config: Dict[str, Any]):
        """
        Execute the actual experiment logic
        This integrates with your Phase 3A statistical testing infrastructure
        """
        # Simulate experiment execution (replace with your Phase 3A integration)
        logger.info(f"Starting experiment execution for {experiment_id}")
        
        # This would call your Phase 3A endpoints:
        # - Configure experiment
        # - Start A/B testing
        # - Monitor statistical significance
        # - Apply early stopping criteria
        
        # Simulate progressive execution
        for phase in ["initialization", "data_collection", "analysis", "validation"]:
            logger.info(f"Experiment {experiment_id} in phase: {phase}")
            await asyncio.sleep(2)  # Simulate work
        
        logger.info(f"Experiment execution completed for {experiment_id}")
    
    async def _finalize_experiment(self, experiment_id: str):
        """Finalize experiment and cleanup resources"""
        experiment = self.experiments[experiment_id]
        
        # Stop monitoring
        self.monitor.stop_monitoring(experiment_id)
        
        # Release resources
        self.resource_manager.release_resources(experiment_id)
        
        # Mark as completed in scheduler
        self.scheduler.mark_experiment_completed(experiment_id)
        
        # Transition to completed state
        if self.state_machine.transition_state(
            experiment_id,
            ExperimentState.COMPLETING,
            ExperimentState.COMPLETED,
            "Experiment finalized successfully"
        ):
            experiment["state"] = ExperimentState.COMPLETED
            experiment["completed_at"] = datetime.now()
        
        logger.info(f"Experiment {experiment_id} finalized successfully")
    
    async def _handle_experiment_failure(self, experiment_id: str, error_message: str):
        """Handle experiment failure with cleanup"""
        experiment = self.experiments[experiment_id]
        
        # Stop monitoring
        self.monitor.stop_monitoring(experiment_id)
        
        # Release resources
        self.resource_manager.release_resources(experiment_id)
        
        # Transition to failed state
        if self.state_machine.transition_state(
            experiment_id,
            experiment["state"],
            ExperimentState.FAILED,
            f"Experiment failed: {error_message}"
        ):
            experiment["state"] = ExperimentState.FAILED
            experiment["error_message"] = error_message
            experiment["failed_at"] = datetime.now()
        
        logger.error(f"Experiment {experiment_id} failed: {error_message}")
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment status"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        progress = self.monitor.get_experiment_progress(experiment_id)
        state_history = self.state_machine.get_state_history(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "metadata": experiment["metadata"].__dict__,
            "state": experiment["state"].value,
            "progress": progress.__dict__ if progress else None,
            "state_history": [
                {
                    "timestamp": entry[0].isoformat(),
                    "from_state": entry[1].value,
                    "to_state": entry[2].value,
                    "reason": entry[3]
                } for entry in state_history
            ],
            "resource_utilization": self.resource_manager.get_resource_utilization()
        }
    
    def get_all_experiments(self) -> Dict[str, Any]:
        """Get status of all experiments"""
        return {
            "experiments": [
                {
                    "experiment_id": exp_id,
                    "name": exp["metadata"].name,
                    "state": exp["state"].value,
                    "created_at": exp["created_at"].isoformat(),
                    "owner": exp["metadata"].owner,
                    "team": exp["metadata"].team
                }
                for exp_id, exp in self.experiments.items()
            ],
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "scheduler_queue_length": len(self.scheduler.scheduled_experiments)
        }