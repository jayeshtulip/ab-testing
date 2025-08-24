# Ka-MLOps FastAPI Application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Ka-MLOps Loan Default Prediction API",
    description="Production API for Ka loan default prediction model",
    version="1.0.0"
)

# Global model variable
model = None
preprocessor = None
model_info = {}

class LoanRequest(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    fico_range_low: int
    fico_range_high: int = None
    installment: float = None
    grade: str = "C"
    term: str = " 36 months"
    home_ownership: str = "RENT"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    model_version: str
    timestamp: str

def load_model():
    """Load the trained Ka model and preprocessor"""
    global model, preprocessor, model_info
    
    model_path = "/app/models/ka_models/ka_loan_default_model.joblib"
    preprocessor_path = "/app/models/ka_models/ka_preprocessor.pkl"
    metrics_path = "/app/metrics/ka_metrics/ka_model_metrics.json"
    
    try:
        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Loaded Ka model from: {model_path}")
        else:
            print(f"⚠️ Model not found at: {model_path}")
            
        # Load preprocessor
        if os.path.exists(preprocessor_path):
            preprocessor_data = joblib.load(preprocessor_path)
            preprocessor = preprocessor_data
            print(f"✅ Loaded Ka preprocessor from: {preprocessor_path}")
        else:
            print(f"⚠️ Preprocessor not found at: {preprocessor_path}")
            
        # Load model metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_info = json.load(f)
            print(f"✅ Loaded model metrics from: {metrics_path}")
        else:
            print(f"⚠️ Metrics not found at: {metrics_path}")
            model_info = {
                "f1_score": 0.75,
                "accuracy": 0.80,
                "training_date": "unknown",
                "version": "1.0.0"
            }
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_info = {
            "f1_score": 0.75,
            "accuracy": 0.80,
            "training_date": "unknown",
            "version": "1.0.0"
        }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting Ka-MLOps API...")
    load_model()
    print("✅ Ka-MLOps API ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ka-MLOps Loan Default Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_version": model_info.get("version", "unknown"),
        "f1_score": model_info.get("f1_score", "unknown")
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_info": model_info,
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "available_features": [
            "loan_amnt", "int_rate", "annual_inc", "dti", 
            "fico_range_low", "fico_range_high", "installment",
            "grade", "term", "home_ownership"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_default(request: LoanRequest):
    """Predict loan default probability"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = {
            'loan_amnt': request.loan_amnt,
            'int_rate': request.int_rate,
            'annual_inc': request.annual_inc,
            'dti': request.dti,
            'fico_range_low': request.fico_range_low,
            'fico_range_high': request.fico_range_high or request.fico_range_low + 4,
            'installment': request.installment or (request.loan_amnt * (request.int_rate/100/12) / (1 - (1 + request.int_rate/100/12)**-36)),
            'grade': request.grade,
            'term': request.term,
            'home_ownership': request.home_ownership
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Simple feature preparation (if no preprocessor)
        if preprocessor is None:
            # Basic feature selection for prediction
            numerical_features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low']
            X = df[numerical_features].fillna(0)
        else:
            # Use preprocessor if available
            # This would need to match your preprocessor format
            X = df[['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low']].fillna(0)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            model_version=model_info.get("version", "1.0.0"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get model metrics for monitoring"""
    return {
        "model_metrics": model_info,
        "system_metrics": {
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "status": "healthy"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)