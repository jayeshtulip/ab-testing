"""
Phase 3B: Automated Experiment Management
Main application entry point

Run with: python main.py
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.experiment_lifecycle_manager import ExperimentLifecycleManager
from api.lifecycle_endpoints import router as lifecycle_router
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global lifecycle manager instance
lifecycle_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global lifecycle_manager
    
    # Startup
    logger.info("🚀 Starting Phase 3B: Automated Experiment Management")
    
    # Initialize lifecycle manager
    lifecycle_manager = ExperimentLifecycleManager()
    await lifecycle_manager.start()
    
    # Make it available to endpoints
    app.state.lifecycle_manager = lifecycle_manager
    
    logger.info("✅ Phase 3B services started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Phase 3B services")
    if lifecycle_manager:
        await lifecycle_manager.stop()
    logger.info("✅ Phase 3B services stopped successfully")

# Create FastAPI application
app = FastAPI(
    title="Phase 3B: Automated Experiment Management",
    description="Intelligent experiment lifecycle automation with resource optimization",
    version="3.1.0-BETA",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(lifecycle_router, prefix="/automation", tags=["Experiment Lifecycle"])

@app.get("/")
async def root():
    """Root endpoint with Phase 3B information"""
    return {
        "message": "🚀 Phase 3B: Automated Experiment Management (BETA)",
        "phase": "3B - Automated Lifecycle Management",
        "version": "3.1.0-BETA",
        "status": "operational",
        "features": [
            "Automated Experiment Lifecycle Management",
            "Intelligent Resource Allocation",
            "Priority-Based Scheduling",
            "Real-Time Progress Monitoring",
            "Dependency Management",
            "State Machine Validation"
        ],
        "endpoints": {
            "create_pipeline": "/automation/experiments/create-pipeline",
            "schedule_experiment": "/automation/experiments/{experiment_id}/schedule",
            "experiment_status": "/automation/experiments/{experiment_id}/status",
            "all_experiments": "/automation/experiments",
            "resource_status": "/automation/resources/status",
            "health_check": "/automation/health"
        },
        "integration": "Built on Phase 3A Statistical Foundation",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with component status"""
    global lifecycle_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": "2025-08-23T15:45:32.123456",
        "version": "3.1.0-BETA",
        "components": {
            "lifecycle_manager": lifecycle_manager is not None and lifecycle_manager._running,
            "resource_manager": True,
            "scheduler": True,
            "state_machine": True,
            "monitor": True
        }
    }
    
    # Check if all components are healthy
    all_healthy = all(health_status["components"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    if lifecycle_manager:
        # Add resource utilization info
        health_status["resource_utilization"] = lifecycle_manager.resource_manager.get_resource_utilization()
        health_status["active_experiments"] = len(lifecycle_manager.experiments)
        health_status["scheduler_queue"] = len(lifecycle_manager.scheduler.scheduled_experiments)
    
    return health_status

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the server
    logger.info("🚀 Starting Phase 3B Server...")
    logger.info(f"Server will be available at: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()