"""
Fixed Resource Manager for Phase 3B
Manages compute, storage, and other resources for experiments
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

class ResourceType(str, Enum):
    """Types of resources"""
    COMPUTE = "compute"
    STORAGE = "storage"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"

class Priority(str, Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float
    unit: str
    priority: Priority = Priority.MEDIUM
    max_amount: Optional[float] = None

@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    allocation_id: str
    experiment_id: str
    resource_type: ResourceType
    allocated_amount: float
    unit: str
    allocated_at: datetime
    priority: Priority
    status: str = "active"

class ResourceManager:
    """Manages experiment resources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Resource pools
        self.resource_pools = self._initialize_resource_pools()
        
        # Active allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Allocation history
        self.allocation_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default resource configuration"""
        return {
            "compute_limit": 100.0,  # CPU cores
            "storage_limit": 1000.0,  # GB
            "memory_limit": 500.0,   # GB
            "gpu_limit": 10.0,       # GPU units
            "network_limit": 1000.0, # Mbps
            "allocation_timeout_minutes": 30,
            "cleanup_interval_minutes": 60
        }
    
    def _initialize_resource_pools(self) -> Dict[ResourceType, Dict[str, float]]:
        """Initialize resource pools"""
        return {
            ResourceType.COMPUTE: {
                "total": self.config["compute_limit"],
                "used": 0.0,
                "reserved": 0.0
            },
            ResourceType.STORAGE: {
                "total": self.config["storage_limit"],
                "used": 0.0,
                "reserved": 0.0
            },
            ResourceType.MEMORY: {
                "total": self.config["memory_limit"],
                "used": 0.0,
                "reserved": 0.0
            },
            ResourceType.GPU: {
                "total": self.config["gpu_limit"],
                "used": 0.0,
                "reserved": 0.0
            },
            ResourceType.NETWORK: {
                "total": self.config["network_limit"],
                "used": 0.0,
                "reserved": 0.0
            }
        }
    
    async def allocate_resources(
        self, 
        experiment_id: str, 
        requirements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Allocate resources for an experiment"""
        
        try:
            self.logger.info(f"Allocating resources for experiment {experiment_id}")
            
            # Convert requirements to ResourceRequirement objects
            resource_reqs = []
            for req in requirements:
                resource_reqs.append(ResourceRequirement(
                    resource_type=ResourceType(req["resource_type"]),
                    amount=req["amount"],
                    unit=req["unit"],
                    priority=Priority(req.get("priority", "medium"))
                ))
            
            # Check if allocation is possible
            if not self.can_allocate_resources(resource_reqs):
                return {
                    "success": False,
                    "error": "Insufficient resources available",
                    "available_resources": await self.get_utilization()
                }
            
            # Perform allocation
            allocations = []
            for req in resource_reqs:
                allocation = await self._allocate_single_resource(experiment_id, req)
                allocations.append(allocation)
            
            result = {
                "success": True,
                "experiment_id": experiment_id,
                "allocations": [asdict(alloc) for alloc in allocations],
                "allocated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully allocated resources for {experiment_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Resource allocation failed for {experiment_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def can_allocate_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated"""
        for req in requirements:
            pool = self.resource_pools.get(req.resource_type)
            if not pool:
                return False
            
            available = pool["total"] - pool["used"] - pool["reserved"]
            if available < req.amount:
                return False
        
        return True
    
    async def _allocate_single_resource(
        self, 
        experiment_id: str, 
        requirement: ResourceRequirement
    ) -> ResourceAllocation:
        """Allocate a single resource"""
        
        allocation_id = f"alloc_{experiment_id}_{requirement.resource_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Update resource pool
        pool = self.resource_pools[requirement.resource_type]
        pool["used"] += requirement.amount
        
        # Create allocation record
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            experiment_id=experiment_id,
            resource_type=requirement.resource_type,
            allocated_amount=requirement.amount,
            unit=requirement.unit,
            allocated_at=datetime.now(),
            priority=requirement.priority,
            status="active"
        )
        
        # Store allocation
        self.allocations[allocation_id] = allocation
        self.allocation_history.append(allocation)
        
        return allocation
    
    async def deallocate_resources(self, experiment_id: str) -> Dict[str, Any]:
        """Deallocate all resources for an experiment"""
        
        try:
            deallocated = []
            
            # Find all allocations for this experiment
            for allocation_id, allocation in list(self.allocations.items()):
                if allocation.experiment_id == experiment_id:
                    # Update resource pool
                    pool = self.resource_pools[allocation.resource_type]
                    pool["used"] -= allocation.allocated_amount
                    
                    # Mark as deallocated
                    allocation.status = "deallocated"
                    deallocated.append(allocation_id)
                    
                    # Remove from active allocations
                    del self.allocations[allocation_id]
            
            self.logger.info(f"Deallocated {len(deallocated)} resources for {experiment_id}")
            
            return {
                "success": True,
                "experiment_id": experiment_id,
                "deallocated_count": len(deallocated),
                "deallocated_allocations": deallocated
            }
            
        except Exception as e:
            self.logger.error(f"Resource deallocation failed for {experiment_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        
        utilization = {}
        
        for resource_type, pool in self.resource_pools.items():
            total = pool["total"]
            used = pool["used"]
            available = total - used
            utilization_pct = (used / total * 100) if total > 0 else 0
            
            utilization[resource_type.value] = {
                "total": total,
                "used": used,
                "available": available,
                "utilization_percentage": round(utilization_pct, 2)
            }
        
        return utilization
    
    async def get_allocation_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get allocation status for an experiment"""
        
        experiment_allocations = [
            asdict(alloc) for alloc in self.allocations.values()
            if alloc.experiment_id == experiment_id
        ]
        
        return {
            "experiment_id": experiment_id,
            "active_allocations": len(experiment_allocations),
            "allocations": experiment_allocations
        }
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get resource optimization recommendations"""
        
        recommendations = []
        
        for resource_type, pool in self.resource_pools.items():
            utilization_pct = (pool["used"] / pool["total"] * 100) if pool["total"] > 0 else 0
            
            if utilization_pct > 90:
                recommendations.append({
                    "type": "warning",
                    "resource": resource_type.value,
                    "message": f"{resource_type.value} utilization is high ({utilization_pct:.1f}%)",
                    "current_usage": pool["used"],
                    "limit": pool["total"]
                })
            elif utilization_pct < 10:
                recommendations.append({
                    "type": "info",
                    "resource": resource_type.value,
                    "message": f"{resource_type.value} utilization is low ({utilization_pct:.1f}%) - consider reducing limits",
                    "current_usage": pool["used"],
                    "limit": pool["total"]
                })
        
        return recommendations
    
    async def check_availability(self) -> Dict[str, Any]:
        """Check resource availability"""
        
        availability = {}
        all_available = True
        
        for resource_type, pool in self.resource_pools.items():
            available = pool["total"] - pool["used"]
            is_available = available > 0
            
            availability[resource_type.value] = {
                "available": is_available,
                "available_amount": available,
                "total_capacity": pool["total"]
            }
            
            if not is_available:
                all_available = False
        
        return {
            "all_resources_available": all_available,
            "resource_availability": availability,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for resource manager"""
        
        try:
            utilization = await self.get_utilization()
            
            # Check for critical resource usage
            critical_resources = []
            for resource_type, usage in utilization.items():
                if usage["utilization_percentage"] > 95:
                    critical_resources.append(resource_type)
            
            healthy = len(critical_resources) == 0
            
            return {
                "healthy": healthy,
                "critical_resources": critical_resources,
                "active_allocations": len(self.allocations),
                "resource_utilization": utilization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Example usage
async def test_resource_manager():
    """Test the resource manager"""
    
    manager = ResourceManager()
    
    # Test health check
    health = await manager.health_check()
    print(f"💚 Health Check: {health}")
    
    # Test resource allocation
    requirements = [
        {
            "resource_type": "compute",
            "amount": 10.0,
            "unit": "cores",
            "priority": "high"
        },
        {
            "resource_type": "storage", 
            "amount": 50.0,
            "unit": "GB",
            "priority": "medium"
        }
    ]
    
    result = await manager.allocate_resources("test_exp_001", requirements)
    print(f"📦 Allocation Result: {result}")
    
    # Test utilization
    utilization = await manager.get_utilization()
    print(f"📊 Utilization: {utilization}")
    
    # Test deallocation
    dealloc_result = await manager.deallocate_resources("test_exp_001")
    print(f"🗑️ Deallocation Result: {dealloc_result}")

if __name__ == "__main__":
    asyncio.run(test_resource_manager())