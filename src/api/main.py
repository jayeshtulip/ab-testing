import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_version: str
    timestamp: float

app = FastAPI(title='Loan Default Prediction API')
model = None
preprocessor = None

# Load model and preprocessor
try:
    model = joblib.load('models/model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    print(f'✅ Model loaded (expects {model.n_features_in_} features)')
    print('✅ Preprocessor loaded')
except Exception as e:
    print(f'❌ Error loading: {e}')

def prepare_features(input_features):
    '''Prepare features with all required columns for German Credit dataset'''
    
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

@app.get('/')
async def root():
    return {
        'message': 'Loan Default Prediction API',
        'status': 'running',
        'model_loaded': model is not None,
        'expected_features': [f'Attribute{i}' for i in range(1, 21)]
    }

@app.get('/health')
async def health():
    return {
        'status': 'healthy' if model and preprocessor else 'unhealthy',
        'timestamp': time.time(),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail='Model or preprocessor not loaded')
    
    try:
        # Prepare features with all required columns
        df = prepare_features(request.features)
        print(f' Prepared features: {df.columns.tolist()}')
        
        # Transform using preprocessor
        features_processed = preprocessor.transform(df)
        print(f' Processed shape: {features_processed.shape}')
        
        # Make prediction
        prediction_class = model.predict(features_processed)[0]
        prediction_proba = model.predict_proba(features_processed)[0]
        
        return PredictionResponse(
            prediction='good' if prediction_class == 0 else 'bad',
            probability=float(max(prediction_proba)),
            model_version='1.0.0',
            timestamp=time.time()
        )
        
    except Exception as e:
        print(f' Prediction error: {e}')
        raise HTTPException(status_code=400, detail=f'Prediction error: {str(e)}')

@app.get('/debug/sample-request')
async def sample_request():
    '''Get a sample request format'''
    return {
        'sample_minimal': {
            'features': {
                'Attribute1': 'A11',
                'Attribute2': 24,
                'Attribute3': 'A32',
                'Attribute5': 3500
            }
        },
        'sample_full': {
            'features': {
                'Attribute1': 'A11', 'Attribute2': 24, 'Attribute3': 'A32',
                'Attribute4': 'A43', 'Attribute5': 3500, 'Attribute6': 'A61',
                'Attribute7': 'A71', 'Attribute8': 4, 'Attribute9': 'A91',
                'Attribute10': 'A101', 'Attribute11': 4, 'Attribute12': 'A121',
                'Attribute13': 35, 'Attribute14': 'A141', 'Attribute15': 'A151',
                'Attribute16': 2, 'Attribute17': 'A171', 'Attribute18': 1,
                'Attribute19': 'A191', 'Attribute20': 'A201'
            }
        }
    }

if __name__ == '__main__':
    print(' Starting Working API on port 8001...')
    print(' Docs: http://localhost:8001/docs')
    print(' Sample request: http://localhost:8001/debug/sample-request')
    uvicorn.run(app, host='0.0.0.0', port=8001)
