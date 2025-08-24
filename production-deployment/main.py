"""
A/B Testing API Service - Production Ready with Prometheus Metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import asyncpg
import logging
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import uvicorn
from contextlib import asynccontextmanager
import time

# Prometheus metrics imports - NEW
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import prometheus_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DB_HOST = os.getenv("DB_HOST", "postgres.loan-default.svc.cluster.local")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "ab_experiments")
    DB_USER = os.getenv("DB_USER", "mlops_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")
    DB_SCHEMA = os.getenv("DB_SCHEMA", "ab_testing")
    
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "loan_default_rf_vs_gb")
    CONTROL_MODEL = os.getenv("CONTROL_MODEL", "random_forest_v1.0")
    TREATMENT_MODEL = os.getenv("TREATMENT_MODEL", "gradient_boost_v1.0")
    DEFAULT_TRAFFIC_SPLIT = float(os.getenv("DEFAULT_TRAFFIC_SPLIT", "0.5"))
    
    API_PORT = int(os.getenv("API_PORT", "8003"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

config = Config()

# Prometheus metrics - NEW
REQUEST_COUNT = Counter(
    'ab_testing_requests_total', 
    'Total A/B testing requests', 
    ['method', 'endpoint', 'status_code', 'experiment_group']
)

REQUEST_DURATION = Histogram(
    'ab_testing_request_duration_seconds', 
    'A/B testing request duration', 
    ['method', 'endpoint', 'experiment_group']
)

PREDICTIONS_COUNT = Counter(
    'ab_testing_predictions_total',
    'Total predictions made',
    ['experiment_group', 'model_version', 'prediction']
)

PREDICTION_PROBABILITY = Histogram(
    'ab_testing_prediction_probability',
    'Distribution of prediction probabilities',
    ['experiment_group', 'model_version']
)

RISK_SCORE_DISTRIBUTION = Histogram(
    'ab_testing_risk_score_distribution',
    'Distribution of risk scores',
    ['experiment_group'],
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

ACTIVE_EXPERIMENTS = Gauge(
    'ab_testing_active_experiments_total', 
    'Number of active A/B testing experiments'
)

DATABASE_CONNECTIONS = Gauge(
    'ab_testing_database_connections_active', 
    'Active database connections'
)

MODEL_PREDICTION_TIME = Histogram(
    'ab_testing_model_prediction_seconds',
    'Time taken for model prediction',
    ['model_version']
)

# Pydantic models
class LoanRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    annual_income: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    debt_to_income_ratio: float = Field(..., ge=0, le=1)
    employment_length: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    user_id: str
    experiment_group: str
    model_version: str
    prediction: int
    probability: float
    risk_score: float
    timestamp: datetime

# Database connection
db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = await asyncpg.create_pool(
                host=config.DB_HOST,
                port=config.DB_PORT,
                database=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                min_size=2,
                max_size=10
            )
            logger.info("Database pool created successfully")
            # Update database connections gauge - NEW
            DATABASE_CONNECTIONS.set(db_pool.get_size())
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    return db_pool

# A/B Testing Manager
class ABTestingManager:
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        logger.info("Initializing A/B test models...")
        
        # Simple synthetic data for demo
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = (X.sum(axis=1) > 0).astype(int)
        
        # Control model
        self.models[config.CONTROL_MODEL] = RandomForestClassifier(n_estimators=50, random_state=42)
        self.models[config.CONTROL_MODEL].fit(X, y)
        
        # Treatment model
        self.models[config.TREATMENT_MODEL] = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.models[config.TREATMENT_MODEL].fit(X, y)
        
        logger.info("Models initialized successfully")
    
    def assign_group(self, user_id: str) -> str:
        # Consistent hash-based assignment
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "control" if hash_val % 2 == 0 else "treatment"
    
    def predict(self, features: np.ndarray, group: str):
        model_version = config.CONTROL_MODEL if group == "control" else config.TREATMENT_MODEL
        model = self.models[model_version]
        
        # Time the prediction - NEW
        start_time = time.time()
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        prediction_time = time.time() - start_time
        
        # Record prediction timing metric - NEW
        MODEL_PREDICTION_TIME.labels(model_version=model_version).observe(prediction_time)
        
        return prediction, probability, model_version

ab_manager = ABTestingManager()

# Metrics middleware - NEW
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting A/B Testing API...")
    try:
        await get_db_pool()
        # Set active experiments gauge - NEW
        ACTIVE_EXPERIMENTS.set(1)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
    yield
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

app = FastAPI(
    title="A/B Testing API Service",
    description="Production A/B Testing for Loan Default Prediction",
    version="1.1.0",  # Updated version
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware - NEW
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for all requests"""
    start_time = time.time()
    
    # Get experiment group from request state if available
    experiment_group = "unknown"
    
    response = await call_next(request)
    
    # Try to get experiment group from response if it's a prediction
    if hasattr(response, '_experiment_group'):
        experiment_group = response._experiment_group
    
    # Calculate request duration
    process_time = time.time() - start_time
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        experiment_group=experiment_group
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
        experiment_group=experiment_group
    ).observe(process_time)
    
    return response

@app.get("/health")
async def health_check():
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {
            "status": "healthy", 
            "timestamp": datetime.now(),
            "version": "1.1.0"  # Updated version
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    return {
        "status": "ready",
        "models_loaded": len(ab_manager.models),
        "experiment": config.EXPERIMENT_NAME
    }

# NEW - Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_default(request: LoanRequest):
    try:
        # Assign user to group
        group = ab_manager.assign_group(request.user_id)
        
        # Prepare features
        features = np.array([[
            request.annual_income / 100000,  # Normalize
            request.credit_score / 850,      # Normalize  
            request.loan_amount / 100000,    # Normalize
            request.debt_to_income_ratio,
            request.employment_length / 10   # Normalize
        ]])
        
        # Make prediction
        prediction, probability, model_version = ab_manager.predict(features, group)
        
        # Calculate risk score
        risk_score = min(100.0, 
            probability * 70 + 
            (850 - request.credit_score) / 8.5 + 
            request.debt_to_income_ratio * 20
        )
        
        # Record prediction metrics - NEW
        PREDICTIONS_COUNT.labels(
            experiment_group=group,
            model_version=model_version,
            prediction=str(prediction)
        ).inc()
        
        PREDICTION_PROBABILITY.labels(
            experiment_group=group,
            model_version=model_version
        ).observe(probability)
        
        RISK_SCORE_DISTRIBUTION.labels(
            experiment_group=group
        ).observe(risk_score)
        
        response = PredictionResponse(
            user_id=request.user_id,
            experiment_group=group,
            model_version=model_version,
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_score=round(float(risk_score), 1),
            timestamp=datetime.now()
        )
        
        logger.info(f"User {request.user_id} -> {group}: pred={prediction}, prob={probability:.3f}, risk={risk_score:.1f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/experiment/status")
async def get_experiment_status():
    return {
        "experiment_name": config.EXPERIMENT_NAME,
        "status": "active",
        "control_model": config.CONTROL_MODEL,
        "treatment_model": config.TREATMENT_MODEL,
        "traffic_split": config.DEFAULT_TRAFFIC_SPLIT,
        "models_ready": len(ab_manager.models) == 2
    }

# NEW - Metrics summary endpoint for debugging
@app.get("/experiment/metrics-summary")
async def get_metrics_summary():
    """Get a summary of current metrics for debugging"""
    return {
        "active_experiments": 1,
        "models_loaded": len(ab_manager.models),
        "experiment_name": config.EXPERIMENT_NAME,
        "metrics_available": [
            "ab_testing_requests_total",
            "ab_testing_request_duration_seconds", 
            "ab_testing_predictions_total",
            "ab_testing_prediction_probability",
            "ab_testing_risk_score_distribution",
            "ab_testing_model_prediction_seconds"
        ],
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.ENVIRONMENT == "development"
    )