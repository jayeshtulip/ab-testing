"""
Loan Default Prediction API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Default Prediction API",
    description="ML API for predicting loan defaults",
    version="1.0.0"
)

# Global model variable
model = None
model_version = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.on_event("startup")
async def load_model():
    """Load the ML model on startup"""
    global model, model_version
    
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
                logger.info("Using dummy model for testing")
                return
        
        # Load the actual model
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Successfully loaded model {model_name} version {model_version}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to dummy model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 20)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        model_version = "dummy-v1"
        logger.info("Using dummy model due to loading error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Loan Default Prediction API",
        "model_version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }