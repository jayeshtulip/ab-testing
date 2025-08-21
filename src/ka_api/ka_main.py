# Ka-MLOps FastAPI Application (Fixed imports)
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import sys

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import Ka modules
from ka_api.ka_schemas import KaLoanRequest, KaPredictionResponse, KaHealthResponse
from ka_modules.ka_data_preprocessing import KaLendingClubPreprocessor

# Initialize Ka FastAPI app
app = FastAPI(
    title="Ka Loan Default Prediction API",
    description="Ka-MLOps API for predicting loan defaults using advanced ML",
    version="1.0.0",
    docs_url="/ka-docs",
    redoc_url="/ka-redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Ka model and preprocessor
ka_model = None
ka_preprocessor = None
ka_feature_columns = None

@app.on_event("startup")
async def load_ka_models():
    '''Load Ka ML model and preprocessor on startup'''
    global ka_model, ka_preprocessor, ka_feature_columns
    
    try:
        print(" Loading Ka models...")
        
        # Get project root directory
        project_root = src_dir.parent
        
        # Load Ka trained model
        model_path = project_root / "models" / "ka_models" / "ka_loan_default_model.pkl"
        if model_path.exists():
            ka_model = joblib.load(model_path)
            print(" Ka model loaded successfully!")
        else:
            print(f" Ka model not found at: {model_path}")
            raise FileNotFoundError(f"Ka model file not found: {model_path}")
        
        # Load Ka preprocessor
        preprocessor_path = project_root / "models" / "ka_models" / "ka_preprocessor.pkl"
        if preprocessor_path.exists():
            ka_preprocessor = KaLendingClubPreprocessor()
            ka_preprocessor.load_preprocessor(preprocessor_path)
            print(" Ka preprocessor loaded successfully!")
        else:
            print(f" Ka preprocessor not found at: {preprocessor_path}")
            raise FileNotFoundError(f"Ka preprocessor file not found: {preprocessor_path}")
        
        # Get feature columns
        ka_feature_columns = ka_preprocessor.numerical_features + ka_preprocessor.categorical_features
        print(f" Ka system ready with {len(ka_feature_columns)} features!")
        
    except Exception as e:
        print(f" Error loading Ka models: {e}")
        raise e

@app.get("/", response_model=dict)
async def ka_root():
    '''Ka root endpoint'''
    return {
        "message": "Ka Loan Default Prediction API v1.0",
        "system": "Ka-MLOps",
        "docs": "/ka-docs",
        "health": "/ka-health",
        "predict": "/ka-predict"
    }

@app.get("/ka-health", response_model=KaHealthResponse)
async def ka_health_check():
    '''Ka health check endpoint'''
    return KaHealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=ka_model is not None,
        ka_system="operational"
    )

@app.post("/ka-predict", response_model=KaPredictionResponse)
async def ka_predict_default(request: KaLoanRequest):
    '''Ka predict loan default probability'''
    
    if ka_model is None or ka_preprocessor is None:
        raise HTTPException(status_code=503, detail="Ka models not loaded")
    
    try:
        print(" Processing Ka prediction request...")
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Add derived features that our preprocessor expects
        input_data['loan_status'] = 'Unknown'  # Placeholder for preprocessing
        
        # Preprocess the input using Ka preprocessor
        X_processed = ka_preprocessor.prepare_features(input_data, is_training=False)
        
        # Make prediction using Ka model
        prediction_proba = ka_model.predict_proba(X_processed)[0]
        default_probability = prediction_proba[1]  # Probability of default
        
        # Determine prediction and confidence
        if default_probability > 0.5:
            prediction = "rejected"
        else:
            prediction = "approved"
        
        # Calculate confidence level
        if default_probability < 0.3 or default_probability > 0.7:
            confidence = "high"
        elif default_probability < 0.4 or default_probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Get risk factors
        risk_factors = _get_ka_risk_factors(input_data, default_probability)
        
        print(f" Ka prediction completed: {prediction} ({default_probability:.3f})")
        
        return KaPredictionResponse(
            prediction=prediction,
            default_probability=round(default_probability, 4),
            confidence=confidence,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        print(f" Ka prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Ka prediction error: {str(e)}")

def _get_ka_risk_factors(input_data, default_probability):
    '''Identify Ka risk factors based on input data'''
    risk_factors = []
    
    # Check specific risk conditions for Ka model
    if input_data['fico_range_low'].iloc[0] < 650:
        risk_factors.append("low_credit_score")
    
    if input_data['dti'].iloc[0] > 25:
        risk_factors.append("high_debt_to_income")
    
    if input_data['delinq_2yrs'].iloc[0] > 0:
        risk_factors.append("recent_delinquencies")
    
    if input_data['inq_last_6mths'].iloc[0] > 3:
        risk_factors.append("multiple_credit_inquiries")
    
    if input_data['pub_rec'].iloc[0] > 0:
        risk_factors.append("public_records")
    
    if input_data['revol_util'].iloc[0] > 80:
        risk_factors.append("high_credit_utilization")
    
    if input_data['grade'].iloc[0] in ['E', 'F', 'G']:
        risk_factors.append("low_loan_grade")
    
    # If no specific risk factors and high default probability, add general factors
    if not risk_factors and default_probability > 0.4:
        risk_factors = ["overall_risk_profile", "credit_history_concerns"]
    
    # If low risk but some factors, keep it simple
    if not risk_factors:
        risk_factors = ["minimal_risk_detected"]
    
    return risk_factors[:5]  # Return top 5 risk factors

@app.get("/ka-model-info")
async def ka_model_info():
    '''Get Ka model information'''
    if ka_model is None:
        raise HTTPException(status_code=503, detail="Ka model not loaded")
    
    return {
        "model_type": "KaRandomForestClassifier",
        "features_count": len(ka_feature_columns) if ka_feature_columns else 0,
        "model_parameters": {
            "n_estimators": ka_model.n_estimators,
            "max_depth": ka_model.max_depth,
            "class_weight": str(ka_model.class_weight)
        },
        "ka_system": "advanced_lending_ml",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    print(" Starting Ka-MLOps API Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
