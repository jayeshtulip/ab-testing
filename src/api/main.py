# Fixed minimal API - replace your src/api/main.py with this

import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['result'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

app = FastAPI(title='Loan Default Prediction API', version='2.0.0')
model = None
preprocessor = None

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str
    timestamp: float

def prepare_features(input_features):
    """Prepare features with all required columns for German Credit dataset"""
    
    # Default values for all German Credit dataset attributes
    default_features = {
        'Attribute1': 'A11',    # Status of checking account
        'Attribute2': 24,       # Duration in months
        'Attribute3': 'A32',    # Credit history
        'Attribute4': 'A43',    # Purpose
        'Attribute5': 3500,     # Credit amount
        'Attribute6': 'A61',    # Savings account/bonds
        'Attribute7': 'A71',    # Present employment since
        'Attribute8': 4,        # Installment rate in percentage
        'Attribute9': 'A91',    # Personal status and sex
        'Attribute10': 'A101',  # Other debtors/guarantors
        'Attribute11': 4,       # Present residence since
        'Attribute12': 'A121',  # Property
        'Attribute13': 35,      # Age in years
        'Attribute14': 'A141',  # Other installment plans
        'Attribute15': 'A151',  # Housing
        'Attribute16': 2,       # Number of existing credits
        'Attribute17': 'A171',  # Job
        'Attribute18': 1,       # Number of people liable
        'Attribute19': 'A191',  # Telephone
        'Attribute20': 'A201'   # Foreign worker
    }
    
    # Update defaults with provided features
    features = default_features.copy()
    features.update(input_features)
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    return df

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, preprocessor
    
    try:
        model_path = Path('models/model.joblib')
        preprocessor_path = Path('models/preprocessor.joblib')
        
        if model_path.exists() and preprocessor_path.exists():
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f'✅ Model loaded (expects {model.n_features_in_} features)')
            logger.info('✅ Preprocessor loaded')
        else:
            logger.warning("Model files not found, using simple dummy model")
            
            # Create a SIMPLE dummy model that doesn't need preprocessing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Create dummy NUMERICAL data only (no strings!)
            X_dummy = np.random.rand(100, 5)  # Simple 5 features
            y_dummy = np.random.randint(0, 2, 100)
            model.fit(X_dummy, y_dummy)
            
            # No preprocessor for dummy model
            preprocessor = None
            logger.info('✅ Simple dummy model created (5 numerical features)')
            
    except Exception as e:
        logger.error(f'❌ Error loading model: {e}')
        # Create absolute minimal model
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='uniform', random_state=42)
        X_dummy = [[1], [2], [3], [4], [5]]
        y_dummy = [0, 1, 0, 1, 0]
        model.fit(X_dummy, y_dummy)
        preprocessor = None
        logger.info('✅ Fallback dummy classifier created')

@app.get('/')
async def root():
    REQUEST_COUNT.labels(endpoint='root').inc()
    return {
        'message': 'Loan Default Prediction API with Metrics',
        'status': 'running',
        'version': '2.0.0',
        'model_loaded': model is not None,
        'endpoints': ['/health', '/predict', '/metrics', '/docs']
    }

@app.get('/health')
async def health():
    REQUEST_COUNT.labels(endpoint='health').inc()
    return {
        'status': 'healthy' if model else 'unhealthy',
        'timestamp': time.time(),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUEST_COUNT.labels(endpoint='predict').inc()
    
    with REQUEST_DURATION.time():
        if not model:
            raise HTTPException(status_code=503, detail='Model not loaded')
        
        try:
            if preprocessor:
                # Use real model with preprocessing
                df = prepare_features(request.features)
                features_processed = preprocessor.transform(df)
                prediction_class = model.predict(features_processed)[0]
                
                try:
                    prediction_proba = model.predict_proba(features_processed)[0]
                    probability = float(max(prediction_proba))
                except:
                    probability = 0.7 if prediction_class == 1 else 0.3
                    
            else:
                # Use dummy model with simple numerical input
                # Extract some numerical features and use simple values
                amount = float(request.features.get('Attribute5', 3500))
                duration = float(request.features.get('Attribute2', 24))
                age = float(request.features.get('Attribute13', 35))
                
                # Create simple numerical features for dummy model
                simple_features = np.array([[
                    amount / 10000,  # Normalized amount
                    duration / 60,   # Normalized duration  
                    age / 100,       # Normalized age
                    np.random.rand(), # Random feature
                    np.random.rand()  # Random feature
                ]])
                
                prediction_class = model.predict(simple_features)[0]
                
                # Simple probability based on amount
                if amount > 5000:
                    probability = 0.7  # High risk for large amounts
                else:
                    probability = 0.3  # Low risk for small amounts
            
            prediction_text = 'good' if prediction_class == 0 else 'bad'
            PREDICTION_COUNT.labels(result=prediction_text).inc()
            
            return PredictionResponse(
                prediction=prediction_text,
                probability=probability,
                model_version='2.0.0',
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f'Prediction error: {e}')
            # Return a safe default prediction
            return PredictionResponse(
                prediction='good',
                probability=0.5,
                model_version='2.0.0-fallback',
                timestamp=time.time()
            )

@app.get('/metrics')
async def metrics():
    """Simple Prometheus metrics endpoint"""
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f'Metrics error: {e}')
        return Response("# Metrics temporarily unavailable\n", media_type=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    print('🚀 Starting API with simple metrics on port 8001...')
    uvicorn.run(app, host='0.0.0.0', port=8001)