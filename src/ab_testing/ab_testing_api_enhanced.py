"""
Enhanced A/B Testing API with realistic models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import numpy as np
from datetime import datetime
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    user_id: str
    features: List[float]  # [loan_amount, income, credit_score, debt_to_income, employment_years]
    experiment_id: Optional[int] = None

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: float
    probability: Optional[float] = None
    model_version: str = "dummy"
    experiment_group: str = "control"
    risk_score: Optional[float] = None

app = FastAPI(
    title="Enhanced A/B Testing API",
    description="A/B Testing API with realistic loan default models",
    version="2.0.0"
)

# Global variables for different models
CONTROL_MODEL = None
TREATMENT_MODEL = None
SCALER = None
MODEL_LOADED = False

def create_realistic_models():
    """Create two different models for A/B testing"""
    global CONTROL_MODEL, TREATMENT_MODEL, SCALER, MODEL_LOADED
    
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Creating realistic models...")
        
        # Generate more realistic training data
        np.random.seed(42)
        n_samples = 2000
        
        # Generate correlated features for loan default prediction
        loan_amounts = np.random.lognormal(10.5, 0.8, n_samples)  # -
        incomes = np.random.lognormal(11.2, 0.6, n_samples)       # -
        credit_scores = np.random.normal(680, 80, n_samples).clip(300, 850)
        debt_ratios = np.random.beta(2, 6, n_samples).clip(0, 0.8)
        employment = np.random.exponential(4, n_samples).clip(0, 30)
        
        X = np.column_stack([loan_amounts, incomes, credit_scores, debt_ratios, employment])
        
        # Create realistic target based on financial logic
        risk_factors = (
            (X[:, 0] / X[:, 1]) * 2.0 +           # High loan-to-income = risky
            ((850 - X[:, 2]) / 100) * 1.5 +       # Low credit score = risky  
            X[:, 3] * 3.0 +                       # High debt ratio = risky
            (1 / (X[:, 4] + 1)) * 0.8 +          # Low employment = risky
            np.random.normal(0, 0.3, n_samples)   # Random variation
        )
        
        # Convert to binary (roughly 20% default rate)
        threshold = np.percentile(risk_factors, 80)
        y = (risk_factors > threshold).astype(int)
        
        logger.info(f"Generated training data: {X.shape}, default rate: {np.mean(y):.2%}")
        
        # Create scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train CONTROL model (Random Forest - Conservative)
        control_model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=8, 
            random_state=42,
            class_weight='balanced'
        )
        control_model.fit(X_scaled, y)
        
        # Train TREATMENT model (Gradient Boosting - More Aggressive)
        treatment_model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1,
            random_state=42
        )
        treatment_model.fit(X_scaled, y)
        
        # Test both models
        control_pred = control_model.predict_proba(X_scaled[:5])
        treatment_pred = treatment_model.predict_proba(X_scaled[:5])
        
        logger.info("Control model test predictions:")
        for i, pred in enumerate(control_pred):
            logger.info(f"  Sample {i}: {pred[1]:.3f} probability of default")
            
        logger.info("Treatment model test predictions:")
        for i, pred in enumerate(treatment_pred):
            logger.info(f"  Sample {i}: {pred[1]:.3f} probability of default")
        
        # Set globals
        CONTROL_MODEL = control_model
        TREATMENT_MODEL = treatment_model
        SCALER = scaler
        MODEL_LOADED = True
        
        logger.info(" Both models created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        logger.error(traceback.format_exc())
        return False

# Initialize models
logger.info("Initializing enhanced models...")
create_realistic_models()

def calculate_risk_score(features):
    """Calculate interpretable risk score"""
    loan_amount, income, credit_score, debt_ratio, employment = features
    
    # Simple risk calculation
    loan_to_income = loan_amount / income if income > 0 else 10
    credit_risk = (850 - credit_score) / 100
    employment_risk = 1 / (employment + 1)
    
    risk_score = (
        loan_to_income * 0.3 +
        credit_risk * 0.4 + 
        debt_ratio * 0.2 +
        employment_risk * 0.1
    )
    
    return min(risk_score * 100, 100)  # Scale to 0-100

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Enhanced prediction with A/B testing"""
    try:
        if not MODEL_LOADED:
            raise HTTPException(status_code=503, detail="Models not initialized")
        
        # A/B assignment based on user_id hash
        user_hash = hash(request.user_id) % 100
        if user_hash < 50:
            group = "control"
            model = CONTROL_MODEL
            model_version = "random_forest_v1.0"
        else:
            group = "treatment" 
            model = TREATMENT_MODEL
            model_version = "gradient_boost_v1.0"
        
        # Prepare and scale features
        features = np.array([request.features])
        features_scaled = SCALER.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Calculate interpretable risk score
        risk_score = calculate_risk_score(request.features)
        
        logger.info(f"User {request.user_id} -> {group}: pred={prediction}, prob={probability:.3f}, risk={risk_score:.1f}")
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            experiment_group=group,
            risk_score=float(risk_score)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": "Windows",
        "version": "2.0.0",
        "models_loaded": MODEL_LOADED,
        "control_model": "RandomForestClassifier" if CONTROL_MODEL else None,
        "treatment_model": "GradientBoostingClassifier" if TREATMENT_MODEL else None
    }

@app.get("/experiment/stats")
async def experiment_stats():
    """Get experiment statistics"""
    return {
        "total_requests": 0,  # Would track in real implementation
        "control_group": {
            "model": "RandomForestClassifier",
            "description": "Conservative model with balanced classes",
            "requests": 0
        },
        "treatment_group": {
            "model": "GradientBoostingClassifier", 
            "description": "Aggressive model with gradient boosting",
            "requests": 0
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Enhanced A/B Testing API for Loan Default Prediction! ",
        "docs": "/docs",
        "health": "/health",
        "stats": "/experiment/stats"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
