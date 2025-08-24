"""
Resource Manager for Phase 3B
Intelligent resource allocation and management
"""

import logging
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Resource types for experiment execution"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    MODEL_SLOTS = "model_slots"

class Priority(Enum):
    """Resource allocation priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float
    unit: str
    max_amount: Optional[float] = None
    priority: Priority = Priority.MEDIUM
    
    def __post_init__(self):
        """Validate resource requirement"""
        if self.amount <= 0:
            raise ValueError("Resource amount must be positive")
        if self.max_amount and self.max_amount < self.amount:
            raise ValueError("Max amount cannot be less than required amount")

@dataclass
class ResourceAllocation:
    """Track resource allocation for experiments"""
    experiment_id: str
    requirements: List[ResourceRequirement]
    allocated_at: datetime
    priority: Priority

class ResourceManager:
    """Intelligent resource allocation and management"""
    
    def __init__(self, 
                 compute_limit: float = 100.0,
                 storage_limit: float = 1000.0,
                 network_limit: float = 50.0,
                 model_slots_limit: int = 10):
        """
        Initialize resource manager with limits
        
        Args:
            compute_limit: Maximum compute resources (cores)
            storage_limit: Maximum storage resources (GB)
            network_limit: Maximum network bandwidth (Mbps)
            model_slots_limit: Maximum concurrent model slots
        """
        self.resource_limits = {
            ResourceType.COMPUTE: float(compute_limit),
            ResourceType.STORAGE: float(storage_limit),
            ResourceType.NETWORK: float(network_limit),
            ResourceType.MODEL_SLOTS: float(model_slots_limit)
        }
        
        self.allocated_resources = {
            ResourceType.COMPUTE: 0.0,
            ResourceType.STORAGE: 0.0,
            ResourceType.NETWORK: 0.0,
            ResourceType.MODEL_SLOTS: 0.0
        }
        
        # Track allocations per experiment
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Resource usage history for optimization
        self.usage_history: List[Dict] = []
        
        logger.info("Resource Manager initialized with limits: %s", self.resource_limits)
    
    def check_resource_availability(self, requirements: List[ResourceRequirement]) -> Dict[str, bool]:
        """
        Check if resources are available for allocation
        
        Returns:
            Dict mapping resource types to availability status
        """
        availability = {}
        
        for req in requirements:
            available = self.resource_limits[req.resource_type] - self.allocated_resources[req.resource_type]
            is_available = available >= req.amount
            availability[req.resource_type.value] = is_available
            
            if not is_available:
                logger.warning(
                    "Insufficient %s: needed %.2f, available %.2f", 
                    req.resource_type.value, req.amount, available
                )
        
        return availability
    
    def can_allocate_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if all requested resources can be allocated"""
        availability = self.check_resource_availability(requirements)
        return all(availability.values())
    
    def allocate_resources(self, experiment_id: str, requirements: List[ResourceRequirement], 
                          priority: Priority = Priority.MEDIUM) -> bool:
        """
        Allocate resources for experiment
        
        Args:
            experiment_id: Unique experiment identifier
            requirements: List of resource requirements
            priority: Allocation priority
            
        Returns:
            True if allocation successful, False otherwise
        """
        # Check if experiment already has allocations
        if experiment_id in self.allocations:
            logger.warning("Experiment %s already has resource allocations", experiment_id)
            return False
        
        # Validate requirements
        try:
            for req in requirements:
                if not isinstance(req, ResourceRequirement):
                    raise ValueError("Invalid resource requirement type")
        except Exception as e:
            logger.error("Invalid resource requirements: %s", e)
            return False
        
        # Check availability
        if not self.can_allocate_resources(requirements):
            logger.warning("Cannot allocate resources for experiment %s", experiment_id)
            return False
        
        # Allocate resources
        try:
            for req in requirements:
                self.allocated_resources[req.resource_type] += req.amount
            
            # Track allocation
            allocation = ResourceAllocation(
                experiment_id=experiment_id,
                requirements=requirements,
                allocated_at=datetime.now(),
                priority=priority
            )
            self.allocations[experiment_id] = allocation
            
            # Log allocation
            logger.info(
                "Resources allocated for experiment %s: %s", 
                experiment_id, 
                {req.resource_type.value: req.amount for req in requirements}
            )
            
            # Record usage
            self._record_allocation(experiment_id, requirements, priority)
            
            return True
            
        except Exception as e:
            logger.error("Error allocating resources for experiment %s: %s", experiment_id, e)
            # Rollback partial allocation
            self._rollback_allocation(requirements)
            return False
    
    def release_resources(self, experiment_id: str) -> bool:
        """
        Release resources after experiment completion
        
        Args:
            experiment_id: Experiment to release resources for
            
        Returns:
            True if release successful, False otherwise
        """
        if experiment_id not in self.allocations:
            logger.warning("No resource allocation found for experiment %s", experiment_id)
            return False
        
        allocation = self.allocations[experiment_id]
        
        try:
            # Release each resource
            for req in allocation.requirements:
                self.allocated_resources[req.resource_type] -= req.amount
                # Ensure we don't go negative due to floating point errors
                self.allocated_resources[req.resource_type] = max(0, self.allocated_resources[req.resource_type])
            
            # Remove allocation record
            del self.allocations[experiment_id]
            
            logger.info("Resources released for experiment %s", experiment_id)
            
            # Record release
            self._record_release(experiment_id, allocation)
            
            return True
            
        except Exception as e:
            logger.error("Error releasing resources for experiment %s: %s", experiment_id, e)
            return False
    
    def get_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """
        Get current resource utilization statistics
        
        Returns:
            Dict with utilization percentages and absolute values
        """
        utilization = {}
        
        for resource_type in ResourceType:
            limit = self.resource_limits[resource_type]
            used = self.allocated_resources[resource_type]
            available = limit - used
            percentage = (used / limit) * 100 if limit > 0 else 0
            
            utilization[resource_type.value] = {
                "used": used,
                "available": available,
                "limit": limit,
                "utilization_percentage": round(percentage, 2)
            }
        
        return utilization
    
    def get_experiment_allocations(self, experiment_id: Optional[str] = None) -> Dict:
        """
        Get resource allocations for specific experiment or all experiments
        
        Args:
            experiment_id: Optional specific experiment ID
            
        Returns:
            Dict with allocation information
        """
        if experiment_id:
            if experiment_id in self.allocations:
                allocation = self.allocations[experiment_id]
                return {
                    "experiment_id": experiment_id,
                    "allocated_at": allocation.allocated_at.isoformat(),
                    "priority": allocation.priority.name,
                    "resources": [
                        {
                            "type": req.resource_type.value,
                            "amount": req.amount,
                            "unit": req.unit,
                            "priority": req.priority.name
                        }
                        for req in allocation.requirements
                    ]
                }
            else:
                return {"error": f"No allocation found for experiment {experiment_id}"}
        else:
            # Return all allocations
            return {
                "total_allocations": len(self.allocations),
                "allocations": [
                    {
                        "experiment_id": exp_id,
                        "allocated_at": allocation.allocated_at.isoformat(),
                        "priority": allocation.priority.name,
                        "resource_count": len(allocation.requirements)
                    }
                    for exp_id, allocation in self.allocations.items()
                ]
            }
    
    def optimize_allocations(self) -> Dict[str, Any]:
        """
        Analyze resource usage and provide optimization recommendations
        
        Returns:
            Dict with optimization recommendations
        """
        utilization = self.get_resource_utilization()
        recommendations = []
        
        # Check for overutilization
        for resource_type, stats in utilization.items():
            if stats["utilization_percentage"] > 90:
                recommendations.append({
                    "type": "warning",
                    "resource": resource_type,
                    "message": f"{resource_type} utilization is {stats['utilization_percentage']:.1f}% - consider scaling up",
                    "current_usage": stats["used"],
                    "limit": stats["limit"]
                })
            elif stats["utilization_percentage"] < 20:
                recommendations.append({
                    "type": "info",
                    "resource": resource_type,
                    "message": f"{resource_type} utilization is low at {stats['utilization_percentage']:.1f}% - resources could be reallocated",
                    "current_usage": stats["used"],
                    "limit": stats["limit"]
                })
        
        # Check for resource fragmentation
        if len(self.allocations) > 0:
            avg_allocation_size = sum(
                sum(req.amount for req in allocation.requirements)
                for allocation in self.allocations.values()
            ) / len(self.allocations)
            
            if avg_allocation_size < 5:  # Small allocations
                recommendations.append({
                    "type": "optimization",
                    "message": "Many small resource allocations detected - consider batch processing",
                    "average_allocation_size": avg_allocation_size,
                    "total_allocations": len(self.allocations)
                })
        
        return {
            "utilization": utilization,
            "recommendations": recommendations,
            "total_active_allocations": len(self.allocations),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _rollback_allocation(self, requirements: List[ResourceRequirement]):
        """Rollback partial resource allocation on error"""
        for req in requirements:
            if self.allocated_resources[req.resource_type] >= req.amount:
                self.allocated_resources[req.resource_type] -= req.amount
    
    def _record_allocation(self, experiment_id: str, requirements: List[ResourceRequirement], 
                          priority: Priority):
        """Record resource allocation for history tracking"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "allocate",
            "experiment_id": experiment_id,
            "priority": priority.name,
            "resources": [
                {
                    "type": req.resource_type.value,
                    "amount": req.amount,
                    "unit": req.unit
                }
                for req in requirements
            ]
        }
        self.usage_history.append(record)
        
        # Keep only last 1000 records
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
    
    def _record_release(self, experiment_id: str, allocation: ResourceAllocation):
        """Record resource release for history tracking"""
        duration = (datetime.now() - allocation.allocated_at).total_seconds()
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": "release", 
            "experiment_id": experiment_id,
            "duration_seconds": duration,
            "priority": allocation.priority.name,
            "resources": [
                {
                    "type": req.resource_type.value,
                    "amount": req.amount,
                    "unit": req.unit
                }
                for req in allocation.requirements
            ]
        }
        self.usage_history.append(record)
    
    def get_usage_history(self, limit: int = 100) -> List[Dict]:
        """Get recent resource usage history"""
        return self.usage_history[-limit:] if self.usage_history else []
    
    def reset_resources(self):
        """Reset all resource allocations (use with caution)"""
        logger.warning("Resetting all resource allocations")
        self.allocated_resources = {resource_type: 0.0 for resource_type in ResourceType}
        self.allocations.clear()
        self.usage_history.clear()