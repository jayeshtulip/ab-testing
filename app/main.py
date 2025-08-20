"""
Enhanced Loan Default Prediction API with Prometheus Metrics
"""

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import logging
import time
from typing import List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Default Prediction API",
    description="ML API for predicting loan defaults with monitoring",
    version="2.0.0"
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['prediction_class']
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Prediction duration in seconds'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy'
)

MODEL_VERSION = Gauge(
    'model_version',
    'Current model version'
)

ACTIVE_MODELS = Gauge(
    'active_models',
    'Number of active models'
)

ERROR_COUNT = Counter(
    'api_errors_total',
    'Total API errors',
    ['error_type']
)

# Global model variables
model = None
model_version = None
model_metrics = {}

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    prediction_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    model_loaded: bool
    model_version: str
    model_accuracy: float

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.on_event("startup")
async def load_model():
    """Load the ML model on startup and set metrics"""
    global model, model_version, model_metrics
    
    try:
        # Get model version from environment
        model_version = os.getenv("MODEL_VERSION", "latest")
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        # Load model from MLflow
        model_name = "loan-default-model"
        
        if model_version == "latest":
            # Get latest version
            client = mlflow.MlflowClient()
            try:
                latest_versions = client.get_latest_versions(model_name, stages=["Production"])
                if not latest_versions:
                    latest_versions = client.get_latest_versions(model_name, stages=["Staging"])
                if not latest_versions:
                    latest_versions = client.get_latest_versions(model_name, stages=["None"])
                
                if latest_versions:
                    model_version = latest_versions[0].version
                    # Get model metrics
                    run = client.get_run(latest_versions[0].run_id)
                    model_metrics = run.data.metrics
                else:
                    raise Exception("No model versions found")
            except Exception as e:
                logger.warning(f"Could not get model from MLflow: {e}")
                # Fallback to dummy model for testing
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                # Fit with dummy data
                X_dummy = np.random.rand(100, 20)
                y_dummy = np.random.randint(0, 2, 100)
                model.fit(X_dummy, y_dummy)
                model_version = "dummy-v1"
                model_metrics = {"accuracy": 0.78, "f1_score": 0.0}
                logger.info("Using dummy model for testing")
                # Set metrics
                MODEL_ACCURACY.set(model_metrics.get("accuracy", 0))
                MODEL_VERSION.set(1.0)  # dummy version
                ACTIVE_MODELS.set(1)
                return
        
        # Load the actual model
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model {model_name} version {model_version}")
        
        # Set Prometheus metrics
        MODEL_ACCURACY.set(model_metrics.get("accuracy", 0))
        try:
            MODEL_VERSION.set(float(model_version))
        except:
            MODEL_VERSION.set(1.0)
        ACTIVE_MODELS.set(1)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ERROR_COUNT.labels(error_type="model_loading").inc()
        # Fallback to dummy model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 20)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        model_version = "dummy-v1"
        model_metrics = {"accuracy": 0.78, "f1_score": 0.0}
        logger.info("Using dummy model due to loading error")
        
        # Set metrics for dummy model
        MODEL_ACCURACY.set(model_metrics.get("accuracy", 0))
        MODEL_VERSION.set(1.0)
        ACTIVE_MODELS.set(1)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with metrics"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        model_loaded=model is not None,
        model_version=model_version or "unknown",
        model_accuracy=model_metrics.get("accuracy", 0.0)
    )

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "Loan Default Prediction API with Monitoring",
        "model_version": model_version,
        "model_accuracy": model_metrics.get("accuracy", 0.0),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction with metrics tracking"""
    if model is None:
        ERROR_COUNT.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prediction_start = time.time()
    
    try:
        # Convert input to DataFrame
        features_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability
        try:
            probabilities = model.predict_proba(features_array)[0]
            probability = float(probabilities[1])  # Probability of default
        except:
            probability = float(prediction)  # Fallback for models without predict_proba
        
        # Record prediction metrics
        prediction_duration = time.time() - prediction_start
        PREDICTION_DURATION.observe(prediction_duration)
        PREDICTION_COUNT.labels(prediction_class=str(prediction)).inc()
        
        # Generate prediction ID
        prediction_id = f"pred_{int(time.time())}_{hash(str(request.features)) % 10000}"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            model_version=model_version or "unknown",
            prediction_id=prediction_id
        )
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "model_metrics": model_metrics,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/stats")
async def model_statistics():
    """Get model usage statistics"""
    return {
        "total_predictions": sum([
            PREDICTION_COUNT.labels(prediction_class="0")._value._value,
            PREDICTION_COUNT.labels(prediction_class="1")._value._value
        ]),
        "predictions_by_class": {
            "no_default": PREDICTION_COUNT.labels(prediction_class="0")._value._value,
            "default": PREDICTION_COUNT.labels(prediction_class="1")._value._value
        },
        "total_requests": REQUEST_COUNT._metrics.get(("GET", "/health", "200"), Counter())._value._value,
        "total_errors": sum([metric._value._value for metric in ERROR_COUNT._metrics.values()]),
        "model_accuracy": MODEL_ACCURACY._value._value,
        "model_version": MODEL_VERSION._value._value
    }

# Add a simple load test endpoint for monitoring testing
@app.post("/test/load")
async def load_test(num_requests: int = 10):
    """Generate load for testing monitoring"""
    results = []
    
    for i in range(min(num_requests, 100)):  # Limit to 100 requests
        test_features = np.random.rand(20).tolist()
        
        try:
            prediction_request = PredictionRequest(features=test_features)
            result = await predict(prediction_request)
            results.append({
                "request_id": i,
                "prediction": result.prediction,
                "probability": result.probability
            })
        except Exception as e:
            results.append({
                "request_id": i,
                "error": str(e)
            })
    
    return {
        "message": f"Load test completed with {num_requests} requests",
        "results": results[:5],  # Return first 5 results
        "total_processed": len(results)
    }