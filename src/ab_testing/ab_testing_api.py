"""
A/B Testing API with detailed error logging
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import numpy as np
from datetime import datetime
import logging
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    user_id: str
    features: List[float]
    experiment_id: Optional[int] = None

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: float
    probability: Optional[float] = None
    model_version: str = "dummy"
    experiment_group: str = "control"

app = FastAPI(
    title="A/B Testing API - Windows Edition",
    description="Simple A/B Testing API for loan default prediction",
    version="1.0.0"
)

# Global variables
MODEL = None
SCALER = None
MODEL_LOADED = False

def initialize_model():
    """Initialize model with detailed logging"""
    global MODEL, SCALER, MODEL_LOADED
    
    try:
        logger.info("Starting model initialization...")
        
        # Check if sklearn is available
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            logger.info("Successfully imported sklearn")
        except ImportError as e:
            logger.error(f"Failed to import sklearn: {e}")
            raise
        
        # Check if numpy is working
        logger.info("Testing numpy...")
        np.random.seed(42)
        X_dummy = np.random.random((100, 5))
        y_dummy = np.random.choice([0, 1], 100)
        logger.info(f"Generated dummy data: X shape {X_dummy.shape}, y shape {y_dummy.shape}")
        
        # Initialize models
        logger.info("Creating scaler...")
        scaler = StandardScaler()
        
        logger.info("Creating RandomForest model...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Fit scaler
        logger.info("Fitting scaler...")
        X_scaled = scaler.fit_transform(X_dummy)
        logger.info(f"Scaled data shape: {X_scaled.shape}")
        
        # Fit model
        logger.info("Fitting model...")
        model.fit(X_scaled, y_dummy)
        logger.info("Model fitted successfully")
        
        # Test prediction
        logger.info("Testing model prediction...")
        test_pred = model.predict(X_scaled[:1])
        test_proba = model.predict_proba(X_scaled[:1])
        logger.info(f"Test prediction: {test_pred}, Test probability: {test_proba}")
        
        # Set globals
        MODEL = model
        SCALER = scaler
        MODEL_LOADED = True
        
        logger.info("Model initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        MODEL_LOADED = False
        return False

# Initialize model on import
logger.info("Initializing model on module import...")
initialize_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction with detailed logging"""
    logger.info(f"Received prediction request for user: {request.user_id}")
    logger.info(f"Features: {request.features}")
    
    try:
        # Check if model is loaded
        logger.info(f"Model loaded status: {MODEL_LOADED}")
        if not MODEL_LOADED or MODEL is None or SCALER is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        logger.info("Model is loaded, proceeding with prediction")
        
        # Simple A/B assignment
        user_hash = hash(request.user_id) % 100
        group = "treatment" if user_hash < 50 else "control"
        logger.info(f"User assigned to group: {group}")
        
        # Prepare features
        logger.info("Preparing features...")
        features = np.array([request.features])
        logger.info(f"Features array shape: {features.shape}")
        
        # Scale features
        logger.info("Scaling features...")
        features_scaled = SCALER.transform(features)
        logger.info(f"Scaled features shape: {features_scaled.shape}")
        
        # Make prediction
        logger.info("Making prediction...")
        prediction = MODEL.predict(features_scaled)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Get probability
        logger.info("Getting probability...")
        probability = MODEL.predict_proba(features_scaled)[0][1]
        logger.info(f"Raw probability: {probability}")
        
        # Create response
        response = PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version="dummy_v1.0",
            experiment_group=group
        )
        
        logger.info(f"Returning response: {response}")
        return response
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with detailed status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": "Windows",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "model_exists": MODEL is not None,
        "scaler_exists": SCALER is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "A/B Testing API is running on Windows!",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
